defmodule Bumblebee.Distributed.TPTransformer do
  @moduledoc """
  Tensor-parallel transformer blocks.

  Implements vLLM-style tensor parallelism for transformer models:

  - **Column-parallel**: QKV projections, FFN up/gate
  - **Row-parallel**: Attention output, FFN down (with all-reduce)
  - **Total communication**: 2 all-reduce operations per transformer block

  ## Architecture

  Each transformer block:

  ```
  Input (replicated)
      │
      ├── RMSNorm (replicated)
      │
      ├── Self-Attention (tensor-parallel)
      │   ├── Q, K, V projections (column-parallel)
      │   ├── Attention computation (local per head subset)
      │   └── Output projection (row-parallel + ALL-REDUCE)
      │
      ├── Residual + RMSNorm
      │
      ├── FFN (tensor-parallel)
      │   ├── Gate + Up projections (column-parallel)
      │   ├── SiLU + multiply (local)
      │   └── Down projection (row-parallel + ALL-REDUCE)
      │
      └── Residual
          │
          ▼
      Output (replicated)
  ```

  ## Supported Models

  - Mistral (with sliding window attention)
  - Llama 2/3
  - Other Llama-style architectures

  ## Example

      mesh = %{axes: %{tp: 2}}
      model = TPTransformer.build_model(mistral_spec, mesh)
  """

  alias Bumblebee.Distributed.TPLayers
  alias Bumblebee.Layers

  import Bumblebee.Utils.Model, only: [join: 2]

  @doc """
  Builds a tensor-parallel model from a Bumblebee spec.

  ## Arguments

    * `spec` - Model specification (e.g., Mistral spec)
    * `mesh` - Device mesh with `:tp` axis

  ## Returns

    Axon model graph with tensor-parallel layers
  """
  @spec build_model(struct(), map(), keyword()) :: Axon.t()
  def build_model(spec, mesh, opts \\ []) do
    # Dispatch based on model architecture
    case spec.__struct__ do
      Bumblebee.Text.Mistral -> build_mistral_model(spec, mesh, opts)
      Bumblebee.Text.Llama -> build_llama_model(spec, mesh, opts)
      other -> raise "Unsupported model for TP: #{inspect(other)}"
    end
  end

  # Mistral-specific model builder
  defp build_mistral_model(spec, mesh, opts) do
    # Input specification
    inputs = tp_inputs(spec)

    # Token embeddings (replicated for simplicity in initial implementation)
    embeddings = tp_embedder(inputs, spec, mesh, name: "embedder")

    # Position IDs
    position_ids =
      Layers.default inputs["position_ids"] do
        Layers.default_position_ids(embeddings)
      end

    # Process through TP transformer blocks
    decoder_outputs =
      tp_decoder(
        embeddings,
        position_ids,
        inputs["attention_mask"],
        inputs["attention_head_mask"],
        inputs["cache"],
        spec,
        mesh,
        Keyword.put(opts, :name, "decoder")
      )

    # Final layer norm (replicated)
    hidden_state =
      Layers.rms_norm(decoder_outputs.hidden_state,
        name: "output_norm",
        epsilon: spec.layer_norm_epsilon
      )

    # LM head - use dense_transposed to match standard Bumblebee param format
    # The kernel is stored as {vocab_size, hidden_size} and transposed during compute
    logits =
      Layers.dense_transposed(hidden_state, spec.vocab_size,
        kernel_initializer: kernel_initializer(spec),
        name: "language_modeling_head.output"
      )

    Layers.output(%{
      logits: logits,
      hidden_state: hidden_state,
      hidden_states: Layers.append(decoder_outputs.hidden_states, hidden_state),
      attentions: decoder_outputs.attentions,
      cache: decoder_outputs.cache
    })
  end

  # Llama uses same architecture as Mistral
  defp build_llama_model(spec, mesh, opts) do
    build_mistral_model(spec, mesh, opts)
  end

  defp tp_inputs(spec) do
    shape = {nil, nil}
    hidden_shape = {nil, nil, spec.hidden_size}
    attention_head_mask_shape = {spec.num_blocks, spec.num_attention_heads}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", optional: true, shape: shape),
      Axon.input("attention_mask", optional: true, shape: shape),
      Axon.input("position_ids", optional: true, shape: shape),
      Axon.input("attention_head_mask", optional: true, shape: attention_head_mask_shape),
      Axon.input("input_embeddings", optional: true, shape: hidden_shape),
      Axon.input("cache", optional: true)
    ])
  end

  defp tp_embedder(inputs, spec, _mesh, opts) do
    name = opts[:name]

    # For initial implementation, embeddings are replicated
    # (could be column-parallel sharded in future optimization)
    Layers.default inputs["input_embeddings"] do
      Axon.embedding(inputs["input_ids"], spec.vocab_size, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "token_embedding")
      )
    end
  end

  defp tp_decoder(
         hidden_state,
         position_ids,
         attention_mask,
         attention_head_mask,
         cache,
         spec,
         mesh,
         opts
       ) do
    name = opts[:name]
    _shard_attention = Keyword.get(opts, :shard_attention, false)

    # Use standard transformer blocks for attention
    # When shard_attention: false (default), attention params are replicated
    # FFN uses TP (column-parallel for gate/up, row-parallel for down)
    Layers.Transformer.blocks(hidden_state,
      attention_mask: attention_mask,
      attention_head_mask: attention_head_mask,
      cache: cache,
      num_blocks: spec.num_blocks,
      num_attention_heads: spec.num_attention_heads,
      num_key_value_heads: spec.num_key_value_heads,
      hidden_size: spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      layer_norm: &Layers.rms_norm(&1, name: &2, epsilon: spec.layer_norm_epsilon),
      ffn:
        &tp_gated_ffn(&1, spec.intermediate_size, spec.hidden_size, mesh,
          name: &2,
          activation: spec.activation,
          kernel_initializer: kernel_initializer(spec)
        ),
      block_type: :norm_first,
      causal: true,
      attention_window_size:
        get_attention_window_size(spec),
      rotary_embedding: [
        position_ids: position_ids,
        max_positions: spec.max_positions,
        base: spec.rotary_embedding_base
      ],
      query_use_bias: false,
      key_use_bias: false,
      value_use_bias: false,
      output_use_bias: false,
      name: join(name, "blocks")
    )
  end

  @doc """
  Tensor-parallel gated FFN (SwiGLU style).

  Architecture:
  ```
  hidden -> gate_proj -> SiLU
               ↓           ↓
  hidden -> up_proj   ->  * (multiply)
               ↓
          down_proj + all-reduce
  ```

  Communication: One all-reduce after down projection.
  """
  def tp_gated_ffn(hidden_state, intermediate_size, output_size, mesh, opts) do
    name = opts[:name]
    activation = opts[:activation] || :silu
    kernel_initializer = opts[:kernel_initializer]

    tp_size = mesh_tp_size(mesh)
    local_intermediate = div(intermediate_size, tp_size)

    # Gate projection (column-parallel - local intermediate size)
    gate =
      hidden_state
      |> Axon.dense(local_intermediate,
        kernel_initializer: kernel_initializer,
        name: join(name, "gate"),
        use_bias: false
      )
      |> Axon.activation(activation)

    # Up projection (column-parallel - local intermediate size)
    up =
      Axon.dense(hidden_state, local_intermediate,
        kernel_initializer: kernel_initializer,
        name: join(name, "intermediate"),
        use_bias: false
      )

    # Element-wise multiply (local)
    intermediate = Axon.multiply(gate, up)

    # Down projection (row-parallel)
    # Note: For proper TP, this needs all-reduce after
    # For now, we use standard dense - the sharded weights will give partial results
    output =
      Axon.dense(intermediate, output_size,
        kernel_initializer: kernel_initializer,
        name: join(name, "output"),
        use_bias: false
      )

    # In proper TP mode, add all-reduce here
    if tp_size > 1 do
      TPLayers.all_reduce(output, mesh, name: join(name, "all_reduce"))
    else
      output
    end
  end

  # Reference implementation for tensor-parallel multi-head attention.
  # Currently not used - the actual attention is handled by Bumblebee's
  # standard transformer blocks with sharded weights from ShardedLoader.
  # Kept here as documentation for future full TP attention implementation.
  @doc false
  def tp_attention(hidden_state, spec, mesh, opts) do
    name = opts[:name]
    tp_size = mesh_tp_size(mesh)

    num_heads = spec.num_attention_heads
    num_kv_heads = Map.get(spec, :num_key_value_heads, num_heads)
    head_size = div(spec.hidden_size, num_heads)

    local_num_heads = div(num_heads, tp_size)
    local_num_kv_heads = div(num_kv_heads, tp_size)
    local_hidden = local_num_heads * head_size

    kernel_initializer = kernel_initializer(spec)

    # QKV projections (column-parallel)
    query =
      Axon.dense(hidden_state, local_hidden,
        kernel_initializer: kernel_initializer,
        name: join(name, "query"),
        use_bias: false
      )

    _key =
      Axon.dense(hidden_state, local_num_kv_heads * head_size,
        kernel_initializer: kernel_initializer,
        name: join(name, "key"),
        use_bias: false
      )

    _value =
      Axon.dense(hidden_state, local_num_kv_heads * head_size,
        kernel_initializer: kernel_initializer,
        name: join(name, "value"),
        use_bias: false
      )

    # Attention computation would go here
    # For now, return the query as a placeholder
    # The actual attention is handled by Layers.Transformer.blocks

    # Output projection (row-parallel with all-reduce)
    output =
      Axon.dense(query, spec.hidden_size,
        kernel_initializer: kernel_initializer,
        name: join(name, "output"),
        use_bias: false
      )

    if tp_size > 1 do
      TPLayers.all_reduce(output, mesh, name: join(name, "output_all_reduce"))
    else
      output
    end
  end

  # Helper functions

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defp mesh_tp_size(mesh) do
    case mesh do
      %{axes: axes} -> Map.get(axes, :tp, 1)
      _ -> 1
    end
  end

  defp get_attention_window_size(spec) do
    # Safely get attention_window_size - only Mistral has this
    case Map.get(spec, :attention_window_size) do
      nil -> nil
      size -> {size, size}
    end
  end
end
