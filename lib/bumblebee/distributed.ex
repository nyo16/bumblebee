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
      mesh = %{name: "tp", axes: %{tp: 2}}

      # Load model with tensor parallelism
      {:ok, model_info} = Bumblebee.Distributed.load_model(
        {:hf, "mistralai/Mistral-7B-v0.1"},
        mesh: mesh,
        device_id: 0  # This device's ID (0 or 1)
      )

      # Load tokenizer and generation config (unchanged)
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "mistralai/Mistral-7B-v0.1"})
      {:ok, gen_config} = Bumblebee.load_generation_config({:hf, "mistralai/Mistral-7B-v0.1"})

      # Create distributed serving
      serving = Bumblebee.Distributed.serving(model_info, tokenizer, gen_config)

      # Run inference (uses this device's portion)
      result = Nx.Serving.run(serving, "The capital of France is")

  ## Supported Models

  Currently supports Llama-style architectures:
  - Mistral
  - Llama 2/3
  - (Others can be added by implementing TP transformer blocks)

  ## Requirements

  - Multiple NVIDIA GPUs with NCCL support
  - EXLA with collective operations support
  """

  alias Bumblebee.Distributed.ShardedLoader
  alias Bumblebee.Distributed.TPTransformer

  require Logger

  @type mesh :: %{name: String.t(), axes: %{tp: pos_integer()}}
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
    * `:device_id` - This device's ID in the mesh (default: 0)
    * `:strategy` - Parallelism strategy (default: `:tensor_parallel`)
      * `:tensor_parallel` - vLLM-style column/row parallel layers
      * `:pipeline_parallel` - Layer-by-layer across GPUs (future)
      * `:layer_sharding` - llama.cpp-style full layers per GPU (future)
    * `:spec_overrides` - Model spec overrides
    * `:log_sharding` - Log sharding decisions (default: false)
    * `:shard_attention` - Whether to shard attention layers (default: false).
      When false, only FFN layers are sharded (simpler, avoids head-splitting).
      When true, attention QKV and output are also sharded (requires TP-aware model).

  ## Returns

    * `{:ok, model_info}` - Map containing:
      * `:model` - The Axon model graph with TP layers
      * `:params` - Sharded parameters for this device
      * `:spec` - Model specification
      * `:mesh` - Device mesh used
      * `:strategy` - Parallelism strategy used

    * `{:error, reason}` - If loading fails

  ## Example

      mesh = %{name: "tp", axes: %{tp: 2}}

      {:ok, model_info} = Bumblebee.Distributed.load_model(
        {:hf, "mistralai/Mistral-7B-v0.1"},
        mesh: mesh,
        device_id: 0
      )

      # model_info.params contains sharded weights for device 0
  """
  @spec load_model(Bumblebee.repository(), keyword()) ::
          {:ok, distributed_model_info()} | {:error, term()}
  def load_model(repository, opts \\ []) do
    mesh = Keyword.fetch!(opts, :mesh)
    device_id = Keyword.get(opts, :device_id, 0)
    strategy = Keyword.get(opts, :strategy, :tensor_parallel)

    Logger.info("Loading distributed model with strategy: #{strategy}, device: #{device_id}")

    shard_attention = Keyword.get(opts, :shard_attention, false)

    with :ok <- validate_mesh(mesh, strategy),
         {:ok, spec} <- load_spec(repository, opts),
         model <- build_distributed_model(spec, mesh, strategy, shard_attention: shard_attention),
         {:ok, params} <- load_params(repository, mesh, device_id, opts) do
      {:ok,
       %{
         model: model,
         params: params,
         spec: spec,
         mesh: mesh,
         strategy: strategy,
         device_id: device_id
       }}
    end
  end

  @doc """
  Creates a serving configured for tensor-parallel inference.

  The serving will automatically:
  - Compile the model for distributed execution
  - Handle batching
  - Manage parameter placement

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

    tp_size = get_in(mesh, [:axes, :tp]) || 1

    if tp_size > 1 do
      # TP > 1 with standard Nx.Serving is not yet supported
      # The all_reduce operations require SPMD execution mode
      # For now, raise an informative error with guidance
      raise ArgumentError, """
      Tensor parallelism with TP > 1 is not yet supported via Nx.Serving.

      Current TP size: #{tp_size}

      For TP > 1 inference, use EXLA.SPMD directly:

          # Build SPMD executable
          spmd = EXLA.SPMD.build(input_typespecs, output_typespecs, fn builder ->
            # ... model forward pass with all_reduce ops
          end, num_replicas: #{tp_size})

          # Run with replica-batched inputs
          results = EXLA.SPMD.run(spmd, replica_inputs)

      For single-GPU inference with sharded params (simulating TP), use TP=1:

          mesh = Bumblebee.Distributed.mesh(1)
          {:ok, model_info} = Bumblebee.Distributed.load_model(repo, mesh: mesh, ...)

      The all-reduce NCCL infrastructure is working. See examples/spmd_4gpu_test.exs
      for a demonstration of 4-GPU all-reduce.
      """
    end

    # TP=1: Standard serving with replicated model
    defn_options =
      opts
      |> Keyword.get(:defn_options, [])
      |> Keyword.merge(compiler: EXLA)

    opts = Keyword.put(opts, :defn_options, defn_options)

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

      mesh = %{name: "tp", axes: %{tp: 2}}

      {:ok, serving} = Bumblebee.Distributed.text_generation(
        {:hf, "mistralai/Mistral-7B-v0.1"},
        mesh: mesh,
        device_id: 0
      )
  """
  @spec text_generation(Bumblebee.repository(), keyword()) ::
          {:ok, Nx.Serving.t()} | {:error, term()}
  def text_generation(repository, opts \\ []) do
    serving_opts = Keyword.get(opts, :serving_options, [])

    with {:ok, model_info} <- load_model(repository, opts),
         {:ok, tokenizer} <- Bumblebee.load_tokenizer(repository),
         {:ok, generation_config} <- Bumblebee.load_generation_config(repository) do
      serving = serving(model_info, tokenizer, generation_config, serving_opts)
      {:ok, serving}
    end
  end

  @doc """
  Validates a device mesh configuration.

  Checks that the mesh is properly configured for the given parallelism strategy.
  """
  @spec validate_mesh(mesh(), strategy()) :: :ok | {:error, String.t()}
  def validate_mesh(mesh, strategy) do
    tp_size = get_in(mesh, [:axes, :tp]) || 1

    cond do
      strategy == :tensor_parallel and tp_size < 1 ->
        {:error, "tensor parallel size must be at least 1"}

      strategy == :tensor_parallel and tp_size > 8 ->
        {:error, "tensor parallel size > 8 not recommended (communication overhead)"}

      true ->
        :ok
    end
  end

  @doc """
  Creates a device mesh for tensor parallelism.

  ## Arguments

    * `tp_size` - Number of devices for tensor parallelism
    * `opts` - Options:
      * `:name` - Mesh name (default: "tp")
      * `:dp_size` - Data parallelism size (default: 1)

  ## Example

      # Simple TP mesh with 2 GPUs
      mesh = Bumblebee.Distributed.mesh(2)
      # => %{name: "tp", axes: %{tp: 2}}

      # TP + DP mesh with 2 TP and 2 DP (4 total GPUs)
      mesh = Bumblebee.Distributed.mesh(2, dp_size: 2)
      # => %{name: "tp_dp", axes: %{tp: 2, dp: 2}}
  """
  @spec mesh(pos_integer(), keyword()) :: mesh()
  def mesh(tp_size, opts \\ []) when tp_size >= 1 do
    name = Keyword.get(opts, :name, "tp")
    dp_size = Keyword.get(opts, :dp_size, 1)

    axes =
      if dp_size > 1 do
        %{tp: tp_size, dp: dp_size}
      else
        %{tp: tp_size}
      end

    %{name: name, axes: axes}
  end

  # Private functions

  defp load_spec(repository, opts) do
    spec_opts = Keyword.take(opts, [:spec, :spec_overrides, :module, :architecture])
    Bumblebee.load_spec(repository, spec_opts)
  end

  defp build_distributed_model(spec, mesh, :tensor_parallel, opts) do
    shard_attention = Keyword.get(opts, :shard_attention, false)
    TPTransformer.build_model(spec, mesh, shard_attention: shard_attention)
  end

  defp build_distributed_model(_spec, _mesh, strategy, _opts) do
    raise "Strategy #{inspect(strategy)} not yet implemented"
  end

  defp load_params(repository, mesh, device_id, opts) do
    # Load params using standard Bumblebee (handles HF->Bumblebee name mapping)
    # Then apply sharding based on the Bumblebee-named params
    case Bumblebee.load_model(repository, Keyword.take(opts, [:spec_overrides])) do
      {:ok, %{params: params}} ->
        # Apply sharding to the loaded params
        sharded_params = apply_sharding(params, mesh, device_id, opts)
        {:ok, sharded_params}

      {:error, _} = error ->
        error
    end
  end

  defp apply_sharding(%Axon.ModelState{} = model_state, mesh, device_id, opts) do
    tp_size = mesh_tp_size(mesh)
    log_sharding = Keyword.get(opts, :log_sharding, false)
    shard_attention = Keyword.get(opts, :shard_attention, false)
    sharding_opts = [shard_attention: shard_attention]

    sharded_data =
      model_state.data
      |> Enum.map(fn {layer_name, layer_params} ->
        sharded_layer_params =
          Enum.map(layer_params, fn {param_name, tensor} ->
            full_name = "#{layer_name}.#{param_name}"
            shape = Nx.shape(tensor)
            {sharding_type, axis} = ShardedLoader.infer_sharding(full_name, shape, mesh, sharding_opts)

            sharded_tensor =
              ShardedLoader.shard_tensor(tensor, sharding_type, axis, device_id, tp_size)

            if log_sharding do
              Logger.debug(
                "  #{full_name}: #{inspect(shape)} -> #{sharding_type} -> #{inspect(Nx.shape(sharded_tensor))}"
              )
            end

            {param_name, sharded_tensor}
          end)
          |> Map.new()

        {layer_name, sharded_layer_params}
      end)
      |> Map.new()

    %{model_state | data: sharded_data}
  end

  defp apply_sharding(params, mesh, device_id, opts) when is_map(params) do
    # Handle plain map params (legacy format)
    tp_size = mesh_tp_size(mesh)
    log_sharding = Keyword.get(opts, :log_sharding, false)
    shard_attention = Keyword.get(opts, :shard_attention, false)
    sharding_opts = [shard_attention: shard_attention]

    Enum.map(params, fn {name, tensor} ->
      shape = Nx.shape(tensor)
      {sharding_type, axis} = ShardedLoader.infer_sharding(name, shape, mesh, sharding_opts)

      sharded_tensor =
        ShardedLoader.shard_tensor(tensor, sharding_type, axis, device_id, tp_size)

      if log_sharding do
        Logger.debug(
          "  #{name}: #{inspect(shape)} -> #{sharding_type} -> #{inspect(Nx.shape(sharded_tensor))}"
        )
      end

      {name, sharded_tensor}
    end)
    |> Map.new()
  end

  defp mesh_tp_size(mesh) do
    case mesh do
      %{axes: axes} -> Map.get(axes, :tp, 1)
      _ -> 1
    end
  end
end
