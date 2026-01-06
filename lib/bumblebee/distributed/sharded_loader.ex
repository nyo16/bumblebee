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
  | `*.self_attention.output.kernel` | Row-parallel | Split input dim |
  | `*.gate.kernel` | Column-parallel | Split output dim |
  | `*.intermediate.kernel` | Column-parallel | Split output dim |
  | `*.ffn.output.kernel` | Row-parallel | Split input dim |
  | Other | Replicated | Full copy on each GPU |

  ## Example

      mesh = %{axes: %{tp: 2}}
      device_id = 0  # or 1

      {:ok, params} = ShardedLoader.load_sharded_params(
        {:hf, "mistralai/Mistral-7B-v0.1"},
        mesh,
        device_id: device_id
      )

      # params contains sharded weights for this device
  """

  require Logger

  @type sharding_type :: :column_parallel | :row_parallel | :replicated
  @type sharding_spec :: {sharding_type(), non_neg_integer() | tuple()}

  @doc """
  Loads parameters with appropriate sharding for tensor parallelism.

  ## Arguments

    * `repository` - Bumblebee repository tuple (e.g., `{:hf, "mistralai/Mistral-7B-v0.1"}`)
    * `mesh` - Device mesh with `:tp` axis for tensor parallelism size
    * `opts` - Loading options:
      * `:device_id` - The device ID for this shard (0 to tp_size-1)
      * `:params_mapping` - Optional custom parameter mapping

  ## Returns

    `{:ok, params}` where params is a map of sharded parameters
  """
  @spec load_sharded_params(Bumblebee.repository(), map(), keyword()) ::
          {:ok, map()} | {:error, term()}
  def load_sharded_params(repository, mesh, opts \\ []) do
    device_id = Keyword.get(opts, :device_id, 0)
    tp_size = mesh_tp_size(mesh)

    Logger.info("Loading sharded params for device #{device_id} (TP size: #{tp_size})")

    with {:ok, params_info} <- download_params(repository, opts) do
      params = load_with_sharding(params_info, mesh, device_id, opts)
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

    Tuple of `{sharding_type, axis}` where:
    - `sharding_type` is `:column_parallel`, `:row_parallel`, or `:replicated`
    - `axis` is the axis to shard along (0 or 1), or `nil` if replicated
  """
  @spec infer_sharding(String.t(), tuple(), map()) :: {sharding_type(), non_neg_integer() | nil}
  def infer_sharding(param_name, shape, mesh, opts \\ []) do
    tp_size = mesh_tp_size(mesh)
    name_lower = String.downcase(param_name)
    shard_attention = Keyword.get(opts, :shard_attention, true)

    cond do
      # Skip if tensor parallelism is disabled
      tp_size <= 1 ->
        {:replicated, nil}

      # QKV projections - column parallel (split output dimension)
      # Only shard if shard_attention is true
      is_qkv_projection?(name_lower) and is_kernel?(name_lower) and tuple_size(shape) == 2 ->
        if shard_attention, do: {:column_parallel, 1}, else: {:replicated, nil}

      # Attention output projection - row parallel (split input dimension)
      # Only shard if shard_attention is true
      is_attention_output?(name_lower) and is_kernel?(name_lower) and tuple_size(shape) == 2 ->
        if shard_attention, do: {:row_parallel, 0}, else: {:replicated, nil}

      # FFN gate and up projections - column parallel
      is_ffn_up_gate?(name_lower) and is_kernel?(name_lower) and tuple_size(shape) == 2 ->
        {:column_parallel, 1}

      # FFN down projection - row parallel
      is_ffn_down?(name_lower) and is_kernel?(name_lower) and tuple_size(shape) == 2 ->
        {:row_parallel, 0}

      # Everything else - replicated (LayerNorm, biases, embeddings, etc.)
      true ->
        {:replicated, nil}
    end
  end

  @doc """
  Shards a single tensor according to the sharding strategy.

  ## Arguments

    * `tensor` - The full tensor to shard
    * `sharding_type` - One of `:column_parallel`, `:row_parallel`, `:replicated`
    * `axis` - The axis to shard along (for parallel types)
    * `device_id` - The device ID (0 to tp_size-1)
    * `tp_size` - Total number of tensor parallel devices

  ## Returns

    The sharded tensor for this device.
  """
  @spec shard_tensor(Nx.Tensor.t(), sharding_type(), non_neg_integer() | nil, non_neg_integer(), pos_integer()) ::
          Nx.Tensor.t()
  def shard_tensor(tensor, :replicated, _axis, _device_id, _tp_size) do
    tensor
  end

  def shard_tensor(tensor, :column_parallel, axis, device_id, tp_size) do
    slice_along_axis(tensor, axis, device_id, tp_size)
  end

  def shard_tensor(tensor, :row_parallel, axis, device_id, tp_size) do
    slice_along_axis(tensor, axis, device_id, tp_size)
  end

  @doc """
  Returns parameter name patterns and their sharding strategies.

  Useful for debugging and understanding the sharding scheme.
  """
  @spec sharding_patterns() :: list({Regex.t(), sharding_type(), String.t()})
  def sharding_patterns do
    [
      {~r/query.*kernel$/i, :column_parallel, "Q projection"},
      {~r/key.*kernel$/i, :column_parallel, "K projection"},
      {~r/value.*kernel$/i, :column_parallel, "V projection"},
      {~r/q_proj.*weight$/i, :column_parallel, "Q projection (HF naming)"},
      {~r/k_proj.*weight$/i, :column_parallel, "K projection (HF naming)"},
      {~r/v_proj.*weight$/i, :column_parallel, "V projection (HF naming)"},
      {~r/self_attention.*output.*kernel$/i, :row_parallel, "Attention output"},
      {~r/o_proj.*weight$/i, :row_parallel, "Attention output (HF naming)"},
      {~r/gate.*kernel$/i, :column_parallel, "FFN gate"},
      {~r/gate_proj.*weight$/i, :column_parallel, "FFN gate (HF naming)"},
      {~r/intermediate.*kernel$/i, :column_parallel, "FFN up"},
      {~r/up_proj.*weight$/i, :column_parallel, "FFN up (HF naming)"},
      {~r/ffn.*output.*kernel$/i, :row_parallel, "FFN down"},
      {~r/down_proj.*weight$/i, :row_parallel, "FFN down (HF naming)"}
    ]
  end

  # Private implementation

  defp download_params(repository, opts) do
    # For now, use a simplified approach that works with local files
    # or downloads using Bumblebee's existing infrastructure
    case repository do
      {:local, dir} ->
        find_local_params(dir)

      {:hf, repository_id} ->
        download_hf_params(repository_id, [], opts)

      {:hf, repository_id, hf_opts} ->
        download_hf_params(repository_id, hf_opts, opts)
    end
  end

  defp find_local_params(dir) do
    # Look for safetensors or pytorch files in the local directory
    safetensors =
      dir
      |> Path.join("*.safetensors")
      |> Path.wildcard()
      |> Enum.reject(&String.contains?(&1, "onnx"))

    pytorch =
      dir
      |> Path.join("*.bin")
      |> Path.wildcard()
      |> Enum.filter(&String.contains?(&1, "pytorch_model"))

    paths = if length(safetensors) > 0, do: safetensors, else: pytorch

    if length(paths) > 0 do
      {:ok, %{paths: paths}}
    else
      {:error, "No parameter files found in #{dir}"}
    end
  end

  defp download_hf_params(repository_id, hf_opts, _opts) do
    alias Bumblebee.HuggingFace.Hub

    revision = Keyword.get(hf_opts, :revision)
    auth_token = Keyword.get(hf_opts, :auth_token)

    # First, try to get the model index to find sharded files
    index_filenames = [
      "model.safetensors.index.json",
      "pytorch_model.bin.index.json"
    ]

    # Try to download index file
    index_result =
      Enum.reduce_while(index_filenames, nil, fn filename, _acc ->
        url = Hub.file_url(repository_id, filename, revision)
        case Hub.cached_download(url, auth_token: auth_token) do
          {:ok, path} ->
            # Parse the index to get shard filenames
            case File.read(path) do
              {:ok, content} ->
                case Jason.decode(content) do
                  {:ok, %{"weight_map" => weight_map}} ->
                    shard_files = weight_map |> Map.values() |> Enum.uniq()
                    {:halt, {:ok, shard_files}}
                  _ -> {:cont, nil}
                end
              _ -> {:cont, nil}
            end
          {:error, _} -> {:cont, nil}
        end
      end)

    # If we found an index, download all shards
    case index_result do
      {:ok, shard_files} ->
        download_shard_files(repository_id, shard_files, revision, auth_token)

      nil ->
        # Fall back to single-file model names
        single_filenames = [
          "model.safetensors",
          "pytorch_model.bin"
        ]

        result =
          Enum.reduce_while(single_filenames, {:error, "No parameter files found"}, fn filename, _acc ->
            url = Hub.file_url(repository_id, filename, revision)
            case Hub.cached_download(url, auth_token: auth_token) do
              {:ok, path} -> {:halt, {:ok, [path]}}
              {:error, _} -> {:cont, {:error, "No parameter files found"}}
            end
          end)

        case result do
          {:ok, paths} -> {:ok, %{paths: paths}}
          error -> error
        end
    end
  end

  defp download_shard_files(repository_id, shard_files, revision, auth_token) do
    alias Bumblebee.HuggingFace.Hub

    paths =
      Enum.reduce_while(shard_files, [], fn filename, acc ->
        url = Hub.file_url(repository_id, filename, revision)
        case Hub.cached_download(url, auth_token: auth_token) do
          {:ok, path} -> {:cont, [path | acc]}
          {:error, reason} -> {:halt, {:error, "Failed to download #{filename}: #{inspect(reason)}"}}
        end
      end)

    case paths do
      {:error, _} = error -> error
      paths when is_list(paths) -> {:ok, %{paths: Enum.reverse(paths)}}
    end
  end

  defp load_with_sharding(%{paths: paths}, mesh, device_id, opts) do
    tp_size = mesh_tp_size(mesh)

    # Load each file and apply sharding
    paths
    |> Enum.reduce(%{}, fn path, acc ->
      Logger.debug("Loading from: #{path}")

      # Determine loader based on file extension
      loader_fun = get_loader_fun(path)
      pytorch_state = loader_fun.(path)

      Enum.reduce(pytorch_state, acc, fn {name, tensor}, params ->
        shape = Nx.shape(tensor)
        {sharding_type, axis} = infer_sharding(name, shape, mesh)

        sharded_tensor = shard_tensor(tensor, sharding_type, axis, device_id, tp_size)

        if opts[:log_sharding] do
          Logger.debug(
            "  #{name}: #{inspect(shape)} -> #{sharding_type} -> #{inspect(Nx.shape(sharded_tensor))}"
          )
        end

        Map.put(params, name, sharded_tensor)
      end)
    end)
  end

  defp get_loader_fun(path) do
    cond do
      String.ends_with?(path, ".safetensors") ->
        &Safetensors.read!/1

      String.ends_with?(path, ".bin") or String.ends_with?(path, ".pt") ->
        &Bumblebee.Conversion.PyTorchLoader.load!/1

      true ->
        # Try SafeTensors first, fall back to PyTorch
        &Safetensors.read!/1
    end
  end

  defp slice_along_axis(tensor, axis, device_id, tp_size) do
    shape = Nx.shape(tensor)
    dim_size = elem(shape, axis)
    local_size = div(dim_size, tp_size)
    start_idx = device_id * local_size

    Nx.slice_along_axis(tensor, start_idx, local_size, axis: axis)
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
      (String.contains?(name, "self_attention") or String.contains?(name, "attention") or
         String.contains?(name, "self_attn"))
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

  defp is_kernel?(name) do
    String.contains?(name, "kernel") or String.contains?(name, "weight")
  end

  defp mesh_tp_size(mesh) do
    case mesh do
      %{axes: axes} -> Map.get(axes, :tp, 1)
      _ -> 1
    end
  end
end
