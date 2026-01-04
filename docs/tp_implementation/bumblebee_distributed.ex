# Bumblebee Distributed Module
# File: lib/bumblebee/distributed.ex
#
# Main API for tensor parallelism in Bumblebee.

defmodule Bumblebee.Distributed do
  @moduledoc """
  Tensor parallelism support for Bumblebee models.

  This module enables running large language models across multiple GPUs by
  sharding model weights and using collective operations for synchronization.

  ## Overview

  Tensor parallelism (TP) distributes model computation across GPUs by splitting
  weight matrices. For transformer models:

  - **Column-parallel layers**: QKV projections, FFN up/gate - weights split by output dim
  - **Row-parallel layers**: Attention output, FFN down - weights split by input dim
  - **All-reduce**: Synchronization after row-parallel layers

  ## Example

      # Create a device mesh with 2 GPUs for tensor parallelism
      mesh = EXLA.Sharding.mesh(:tp, tp: 2)

      # Load model with tensor parallelism
      {:ok, model_info} = Bumblebee.Distributed.load_model(
        {:hf, "mistralai/Mistral-7B-v0.1"},
        mesh: mesh,
        strategy: :tensor_parallel
      )

      # Load tokenizer and generation config (unchanged)
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "mistralai/Mistral-7B-v0.1"})
      {:ok, gen_config} = Bumblebee.load_generation_config({:hf, "mistralai/Mistral-7B-v0.1"})

      # Create distributed serving
      serving = Bumblebee.Distributed.serving(model_info, tokenizer, gen_config)

      # Run inference (automatically uses all GPUs)
      result = Nx.Serving.run(serving, "The capital of France is")

  ## Supported Models

  Currently supports Llama-style architectures:
  - Mistral
  - Llama 2/3
  - (Others can be added by implementing TP transformer blocks)

  ## Requirements

  - Multiple NVIDIA GPUs with NCCL support
  - EXLA with PR #1646 (sharding POC) + collective operations
  """

  alias Bumblebee.Distributed.ShardedLoader
  alias Bumblebee.Distributed.TPTransformer

  @type mesh :: EXLA.Sharding.DeviceMesh.t()
  @type strategy :: :tensor_parallel | :pipeline_parallel | :layer_sharding

  @type distributed_model_info :: %{
    model: Axon.t(),
    params: map(),
    spec: struct(),
    mesh: mesh(),
    strategy: strategy()
  }

  @doc """
  Loads a model with weights distributed across devices.

  This function:
  1. Loads the model specification from the repository
  2. Builds a tensor-parallel version of the model
  3. Loads parameters sharded according to TP patterns

  ## Options

    * `:mesh` - Device mesh specifying GPU arrangement (required)
    * `:strategy` - Parallelism strategy (default: `:tensor_parallel`)
      * `:tensor_parallel` - vLLM-style column/row parallel layers
      * `:pipeline_parallel` - Layer-by-layer across GPUs (future)
      * `:layer_sharding` - llama.cpp-style full layers per GPU (future)
    * `:spec_overrides` - Model spec overrides
    * `:backend` - Backend for parameter loading (default: derives from mesh)

  ## Returns

    * `{:ok, model_info}` - Map containing:
      * `:model` - The Axon model graph with TP layers
      * `:params` - Sharded parameters
      * `:spec` - Model specification
      * `:mesh` - Device mesh used
      * `:strategy` - Parallelism strategy used

    * `{:error, reason}` - If loading fails

  ## Example

      mesh = EXLA.Sharding.mesh(:tp, tp: 2)

      {:ok, model_info} = Bumblebee.Distributed.load_model(
        {:hf, "mistralai/Mistral-7B-v0.1"},
        mesh: mesh
      )

      # model_info.params contains sharded weights
      # Each GPU holds only its portion
  """
  @spec load_model(Bumblebee.repository(), keyword()) ::
    {:ok, distributed_model_info()} | {:error, term()}
  def load_model(repository, opts \\ []) do
    mesh = Keyword.fetch!(opts, :mesh)
    strategy = Keyword.get(opts, :strategy, :tensor_parallel)

    with :ok <- validate_mesh(mesh, strategy),
         {:ok, base_info} <- load_base_model_info(repository, opts),
         model <- build_distributed_model(base_info.spec, mesh, strategy),
         {:ok, params} <- ShardedLoader.load_sharded_params(model, repository, mesh, opts) do
      {:ok, %{
        model: model,
        params: params,
        spec: base_info.spec,
        mesh: mesh,
        strategy: strategy
      }}
    end
  end

  @doc """
  Creates a serving configured for tensor-parallel inference.

  The serving will automatically:
  - Compile the model for distributed execution
  - Handle batching across the device mesh
  - Manage KV cache sharding

  ## Options

    * `:compile` - Compilation options (batch_size, sequence_length)
    * `:defn_options` - Additional defn compiler options
    * All other options are passed to `Bumblebee.Text.generation/4`

  ## Example

      serving = Bumblebee.Distributed.serving(
        model_info,
        tokenizer,
        generation_config,
        compile: [batch_size: 1, sequence_length: 512]
      )

      Nx.Serving.run(serving, "Hello, world!")
  """
  @spec serving(distributed_model_info(), Bumblebee.Tokenizer.t(), map(), keyword()) ::
    Nx.Serving.t()
  def serving(model_info, tokenizer, generation_config, opts \\ []) do
    %{model: model, params: params, spec: spec, mesh: mesh} = model_info

    # Merge mesh into defn_options
    defn_options = opts
    |> Keyword.get(:defn_options, [])
    |> Keyword.merge([
      compiler: EXLA,
      mesh: mesh,
      # Enable SPMD compilation
      num_partitions: mesh_size(mesh)
    ])

    opts = Keyword.put(opts, :defn_options, defn_options)

    # Use standard generation serving with TP model
    Bumblebee.Text.generation(
      %{model: model, params: params, spec: spec},
      tokenizer,
      generation_config,
      opts
    )
  end

  @doc """
  Creates a text generation serving with tensor parallelism.

  Convenience function that combines `load_model/2` and `serving/4`.

  ## Example

      mesh = EXLA.Sharding.mesh(:tp, tp: 2)

      serving = Bumblebee.Distributed.text_generation(
        {:hf, "mistralai/Mistral-7B-v0.1"},
        mesh: mesh
      )
  """
  @spec text_generation(Bumblebee.repository(), keyword()) ::
    {:ok, Nx.Serving.t()} | {:error, term()}
  def text_generation(repository, opts \\ []) do
    mesh = Keyword.fetch!(opts, :mesh)
    serving_opts = Keyword.get(opts, :serving_options, [])

    with {:ok, model_info} <- load_model(repository, opts),
         {:ok, tokenizer} <- Bumblebee.load_tokenizer(repository),
         {:ok, generation_config} <- Bumblebee.load_generation_config(repository) do
      serving = serving(model_info, tokenizer, generation_config, serving_opts)
      {:ok, serving}
    end
  end

  # Private functions

  defp validate_mesh(mesh, :tensor_parallel) do
    tp_size = mesh.axes[:tp] || 1

    if tp_size < 1 do
      {:error, "tensor parallel size must be at least 1"}
    else
      :ok
    end
  end

  defp validate_mesh(_mesh, _strategy), do: :ok

  defp load_base_model_info(repository, opts) do
    # Load just the spec, not the full params
    spec_opts = Keyword.take(opts, [:spec, :spec_overrides, :module, :architecture])

    case Bumblebee.load_spec(repository, spec_opts) do
      {:ok, spec} -> {:ok, %{spec: spec}}
      error -> error
    end
  end

  defp build_distributed_model(spec, mesh, :tensor_parallel) do
    TPTransformer.build_model(spec, mesh)
  end

  defp build_distributed_model(_spec, _mesh, strategy) do
    raise "Strategy #{inspect(strategy)} not yet implemented"
  end

  defp mesh_size(%{axes: axes}) do
    Enum.reduce(axes, 1, fn {_name, size}, acc -> acc * size end)
  end
end
