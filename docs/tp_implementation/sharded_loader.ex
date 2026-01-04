# Sharded Parameter Loader
# File: lib/bumblebee/distributed/sharded_loader.ex
#
# Loads model parameters sharded across devices according to tensor parallelism patterns.

defmodule Bumblebee.Distributed.ShardedLoader do
  @moduledoc """
  Loads model parameters sharded across devices for tensor parallelism.

  This module handles the mapping from full model weights to sharded weights
  distributed across GPUs. Each GPU loads only its portion of each parameter.

  ## Sharding Rules

  For Llama/Mistral-style models:

  | Parameter Pattern | Sharding | Description |
  |-------------------|----------|-------------|
  | `*.query.kernel` | Column-parallel | Split output dim |
  | `*.key.kernel` | Column-parallel | Split output dim |
  | `*.value.kernel` | Column-parallel | Split output dim |
  | `*.output.kernel` (attention) | Row-parallel | Split input dim |
  | `*.gate.kernel` | Column-parallel | Split output dim |
  | `*.intermediate.kernel` | Column-parallel | Split output dim |
  | `*.output.kernel` (FFN) | Row-parallel | Split input dim |
  | `embedder.*.kernel` | Column-parallel | Vocab parallel |
  | Other | Replicated | Full copy on each GPU |

  ## Example

      mesh = EXLA.Sharding.mesh(:tp, tp: 2)

      {:ok, params} = ShardedLoader.load_sharded_params(
        model,
        {:hf, "mistralai/Mistral-7B-v0.1"},
        mesh,
        []
      )

      # params contains sharded weights for current device
  """

  require Logger

  @type sharding_type :: :column_parallel | :row_parallel | :replicated
  @type sharding_spec :: {sharding_type(), non_neg_integer()}

  @doc """
  Loads parameters with appropriate sharding for tensor parallelism.

  ## Arguments
    * `model` - The Axon model (used for parameter shape inference)
    * `repository` - Bumblebee repository tuple
    * `mesh` - Device mesh
    * `opts` - Loading options

  ## Returns
    `{:ok, params}` where params is a map of sharded parameters
  """
  @spec load_sharded_params(Axon.t(), Bumblebee.repository(), map(), keyword()) ::
    {:ok, map()} | {:error, term()}
  def load_sharded_params(model, repository, mesh, opts) do
    with {:ok, paths} <- download_params(repository, opts) do
      params = load_with_sharding(paths, mesh)
      {:ok, params}
    end
  end

  @doc """
  Determines sharding strategy for a parameter based on its name and shape.

  ## Arguments
    * `param_name` - Full parameter name (e.g., "decoder.blocks.0.self_attention.query.kernel")
    * `shape` - Parameter shape tuple
    * `mesh` - Device mesh

  ## Returns
    Tuple of `{sharding_type, local_size}` where:
    - `sharding_type` is `:column_parallel`, `:row_parallel`, or `:replicated`
    - `local_size` is the size of the local shard (or full size if replicated)
  """
  @spec infer_sharding(String.t(), tuple(), map()) :: sharding_spec()
  def infer_sharding(param_name, shape, mesh) do
    tp_size = mesh_tp_size(mesh)
    name_lower = String.downcase(param_name)

    cond do
      # QKV projections - column parallel (split output dimension)
      is_qkv_projection?(name_lower) and is_kernel?(name_lower) ->
        {:column_parallel, div(elem(shape, 1), tp_size)}

      # Attention output projection - row parallel (split input dimension)
      is_attention_output?(name_lower) and is_kernel?(name_lower) ->
        {:row_parallel, div(elem(shape, 0), tp_size)}

      # FFN gate and up projections - column parallel
      is_ffn_up_gate?(name_lower) and is_kernel?(name_lower) ->
        {:column_parallel, div(elem(shape, 1), tp_size)}

      # FFN down projection - row parallel
      is_ffn_down?(name_lower) and is_kernel?(name_lower) ->
        {:row_parallel, div(elem(shape, 0), tp_size)}

      # Embedding layer - column parallel (vocabulary parallel)
      is_embedding?(name_lower) and is_kernel?(name_lower) ->
        {:column_parallel, div(elem(shape, 1), tp_size)}

      # LM head (output projection) - column parallel
      is_lm_head?(name_lower) and is_kernel?(name_lower) ->
        {:column_parallel, div(elem(shape, 1), tp_size)}

      # Everything else - replicated (LayerNorm, biases, etc.)
      true ->
        {:replicated, shape}
    end
  end

  @doc """
  Returns parameter name patterns and their sharding strategies.

  Useful for debugging and understanding the sharding scheme.
  """
  @spec sharding_patterns() :: list({Regex.t(), sharding_type(), String.t()})
  def sharding_patterns do
    [
      {~r/query.*kernel$/i, :column_parallel, "QKV projections"},
      {~r/key.*kernel$/i, :column_parallel, "QKV projections"},
      {~r/value.*kernel$/i, :column_parallel, "QKV projections"},
      {~r/self_attention.*output.*kernel$/i, :row_parallel, "Attention output"},
      {~r/gate.*kernel$/i, :column_parallel, "FFN gate"},
      {~r/intermediate.*kernel$/i, :column_parallel, "FFN up"},
      {~r/ffn.*output.*kernel$/i, :row_parallel, "FFN down"},
      {~r/embedder.*kernel$/i, :column_parallel, "Embeddings"},
      {~r/lm_head.*kernel$/i, :column_parallel, "LM head"}
    ]
  end

  # Private implementation

  defp download_params(repository, opts) do
    # Use Bumblebee's existing download infrastructure
    # This handles both single files and sharded checkpoints
    Bumblebee.download_params(repository, opts)
  end

  defp load_with_sharding(paths, mesh) do
    device_id = get_local_device_id()
    tp_size = mesh_tp_size(mesh)

    Logger.info("Loading sharded params for device #{device_id} (TP size: #{tp_size})")

    paths
    |> List.wrap()
    |> Enum.reduce(%{}, fn path, acc ->
      Logger.debug("Loading from: #{path}")

      # Load the PyTorch/SafeTensors checkpoint
      pytorch_state = Bumblebee.Conversion.PyTorchLoader.load!(path)

      Enum.reduce(pytorch_state, acc, fn {name, tensor}, params ->
        shape = Nx.shape(tensor)
        {sharding_type, _local_size} = infer_sharding(name, shape, mesh)

        sharded_tensor = case sharding_type do
          :column_parallel ->
            slice_column_parallel(tensor, device_id, tp_size)

          :row_parallel ->
            slice_row_parallel(tensor, device_id, tp_size)

          :replicated ->
            tensor
        end

        Logger.debug("  #{name}: #{inspect(shape)} -> #{sharding_type} -> #{inspect(Nx.shape(sharded_tensor))}")
        Map.put(params, name, sharded_tensor)
      end)
    end)
  end

  # Column-parallel: split along axis 1 (output dimension)
  defp slice_column_parallel(tensor, device_id, tp_size) do
    {_rows, cols} = get_2d_shape(tensor)
    local_cols = div(cols, tp_size)
    start_col = device_id * local_cols

    Nx.slice_along_axis(tensor, start_col, local_cols, axis: 1)
  end

  # Row-parallel: split along axis 0 (input dimension)
  defp slice_row_parallel(tensor, device_id, tp_size) do
    {rows, _cols} = get_2d_shape(tensor)
    local_rows = div(rows, tp_size)
    start_row = device_id * local_rows

    Nx.slice_along_axis(tensor, start_row, local_rows, axis: 0)
  end

  defp get_2d_shape(tensor) do
    shape = Nx.shape(tensor)
    case tuple_size(shape) do
      2 -> {elem(shape, 0), elem(shape, 1)}
      1 -> {elem(shape, 0), 1}
      _ -> raise "Expected 1D or 2D tensor for sharding, got shape: #{inspect(shape)}"
    end
  end

  # Pattern matching helpers
  defp is_qkv_projection?(name) do
    String.contains?(name, "query") or
    String.contains?(name, "key") or
    String.contains?(name, "value") or
    String.contains?(name, "q_proj") or
    String.contains?(name, "k_proj") or
    String.contains?(name, "v_proj")
  end

  defp is_attention_output?(name) do
    (String.contains?(name, "output") or String.contains?(name, "o_proj")) and
    (String.contains?(name, "self_attention") or String.contains?(name, "attention"))
  end

  defp is_ffn_up_gate?(name) do
    String.contains?(name, "gate") or
    String.contains?(name, "intermediate") or
    String.contains?(name, "up_proj") or
    String.contains?(name, "gate_proj") or
    (String.contains?(name, "mlp") and String.contains?(name, "up"))
  end

  defp is_ffn_down?(name) do
    (String.contains?(name, "ffn") or String.contains?(name, "mlp")) and
    (String.contains?(name, "output") or String.contains?(name, "down_proj"))
  end

  defp is_embedding?(name) do
    String.contains?(name, "embedder") or
    String.contains?(name, "embed_tokens") or
    String.contains?(name, "wte")
  end

  defp is_lm_head?(name) do
    String.contains?(name, "lm_head") or
    String.contains?(name, "output_projection")
  end

  defp is_kernel?(name) do
    String.contains?(name, "kernel") or
    String.contains?(name, "weight")
  end

  defp mesh_tp_size(mesh) do
    mesh.axes[:tp] || 1
  end

  defp get_local_device_id do
    # Get current device ID from EXLA execution context
    # This is set when running in SPMD mode
    Process.get(:exla_device_id, 0)
  end
end
