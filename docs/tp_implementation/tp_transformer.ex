# Tensor-Parallel Transformer Implementation
# File: lib/bumblebee/distributed/tp_transformer.ex
#
# Full tensor-parallel transformer blocks for Llama/Mistral-style models.

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

      mesh = EXLA.Sharding.mesh(:tp, tp: 2)
      model = TPTransformer.build_model(mistral_spec, mesh)
  """

  alias Bumblebee.Distributed.TPLayers
  alias Bumblebee.Layers

  @doc """
  Builds a tensor-parallel model from a Bumblebee spec.

  ## Arguments
    * `spec` - Model specification (e.g., Mistral spec)
    * `mesh` - Device mesh

  ## Returns
    Axon model graph with tensor-parallel layers
  """
  @spec build_model(struct(), map()) :: Axon.t()
  def build_model(spec, mesh) do
    # Dispatch based on model architecture
    case spec.__struct__ do
      Bumblebee.Text.Mistral -> build_mistral_model(spec, mesh)
      Bumblebee.Text.Llama -> build_llama_model(spec, mesh)
      other -> raise "Unsupported model for TP: #{inspect(other)}"
    end
  end

  # Mistral-specific model builder
  defp build_mistral_model(spec, mesh) do
    # Input specification
    inputs = %{
      "input_ids" => Axon.input("input_ids", shape: {nil, nil}),
      "attention_mask" => Axon.input("attention_mask", shape: {nil, nil}),
      "position_ids" => Axon.input("position_ids", shape: {nil, nil})
    }

    # Token embeddings (column-parallel)
    embeddings = TPLayers.parallel_embedding(
      inputs["input_ids"],
      spec.vocab_size,
      spec.hidden_size,
      mesh: mesh,
      name: "embedder.token_embedding"
    )

    # Process through transformer blocks
    {hidden_state, _cache} = Enum.reduce(
      0..(spec.num_hidden_layers - 1),
      {embeddings, nil},
      fn idx, {state, cache} ->
        tp_decoder_block(state, spec, mesh,
          block_idx: idx,
          cache: cache
        )
      end
    )

    # Final layer norm (replicated)
    hidden_state = Layers.rms_norm(hidden_state,
      name: "output_norm",
      epsilon: spec.layer_norm_epsilon
    )

    # LM head (column-parallel on vocab)
    logits = TPLayers.parallel_lm_head(
      hidden_state,
      spec.vocab_size,
      mesh: mesh,
      name: "lm_head"
    )

    Axon.container(%{
      logits: logits,
      hidden_states: hidden_state
    })
  end

  # Llama uses same architecture as Mistral
  defp build_llama_model(spec, mesh) do
    build_mistral_model(spec, mesh)
  end

  @doc """
  Single tensor-parallel decoder block.

  Implements pre-norm transformer block with:
  - TP self-attention
  - TP feed-forward network
  - Residual connections

  ## Arguments
    * `hidden_state` - Input hidden states (replicated)
    * `spec` - Model specification
    * `mesh` - Device mesh
    * `opts` - Options including :block_idx, :cache

  ## Returns
    `{output_hidden_state, updated_cache}`
  """
  @spec tp_decoder_block(Axon.t(), struct(), map(), keyword()) :: {Axon.t(), any()}
  def tp_decoder_block(hidden_state, spec, mesh, opts \\ []) do
    block_idx = Keyword.fetch!(opts, :block_idx)
    name = "decoder.blocks.#{block_idx}"

    # Pre-attention layer norm
    normed = Layers.rms_norm(hidden_state,
      name: "#{name}.self_attention_norm",
      epsilon: spec.layer_norm_epsilon
    )

    # Tensor-parallel self-attention
    {attn_output, cache} = tp_attention(normed, spec, mesh,
      name: "#{name}.self_attention",
      cache: opts[:cache]
    )

    # Residual connection
    hidden_state = Axon.add(hidden_state, attn_output)

    # Pre-FFN layer norm
    normed = Layers.rms_norm(hidden_state,
      name: "#{name}.ffn_norm",
      epsilon: spec.layer_norm_epsilon
    )

    # Tensor-parallel FFN
    ffn_output = tp_ffn(normed, spec, mesh,
      name: "#{name}.ffn"
    )

    # Residual connection
    hidden_state = Axon.add(hidden_state, ffn_output)

    {hidden_state, cache}
  end

  @doc """
  Tensor-parallel multi-head attention.

  Shards attention heads across GPUs:
  - Each GPU handles `num_heads / tp_size` query heads
  - Each GPU handles `num_kv_heads / tp_size` key/value heads (for GQA)

  Communication: One all-reduce after output projection.

  ## Arguments
    * `hidden_state` - Input (replicated)
    * `spec` - Model specification
    * `mesh` - Device mesh
    * `opts` - Options including :name, :cache

  ## Returns
    `{attention_output, updated_cache}`
  """
  @spec tp_attention(Axon.t(), struct(), map(), keyword()) :: {Axon.t(), any()}
  def tp_attention(hidden_state, spec, mesh, opts \\ []) do
    name = Keyword.get(opts, :name, "self_attention")
    tp_size = mesh.axes[:tp] || 1

    # Calculate local head counts
    num_heads = spec.num_attention_heads
    num_kv_heads = Map.get(spec, :num_key_value_heads, num_heads)
    head_size = div(spec.hidden_size, num_heads)

    local_num_heads = div(num_heads, tp_size)
    local_num_kv_heads = div(num_kv_heads, tp_size)
    local_qkv_size = local_num_heads * head_size

    # QKV projections (column-parallel, no all-reduce)
    query = hidden_state
    |> TPLayers.column_parallel_dense(num_heads * head_size,
        mesh: mesh,
        name: "#{name}.query")
    |> reshape_for_attention(local_num_heads, head_size)

    key = hidden_state
    |> TPLayers.column_parallel_dense(num_kv_heads * head_size,
        mesh: mesh,
        name: "#{name}.key")
    |> reshape_for_attention(local_num_kv_heads, head_size)

    value = hidden_state
    |> TPLayers.column_parallel_dense(num_kv_heads * head_size,
        mesh: mesh,
        name: "#{name}.value")
    |> reshape_for_attention(local_num_kv_heads, head_size)

    # Apply rotary position embeddings
    # TODO: Add RoPE implementation
    # query = apply_rope(query, position_ids, spec)
    # key = apply_rope(key, position_ids, spec)

    # Expand KV heads for GQA if needed
    {key, value} = if local_num_kv_heads < local_num_heads do
      repeat_factor = div(local_num_heads, local_num_kv_heads)
      {repeat_kv(key, repeat_factor), repeat_kv(value, repeat_factor)}
    else
      {key, value}
    end

    # Attention computation (local per GPU, no communication)
    attention_output = compute_attention(query, key, value, spec)

    # Reshape back: {batch, seq, local_heads, head_size} -> {batch, seq, local_hidden}
    attention_output = flatten_heads(attention_output)

    # Output projection (row-parallel with all-reduce)
    attention_output = TPLayers.row_parallel_dense(attention_output, spec.hidden_size,
      mesh: mesh,
      name: "#{name}.output"
    )

    {attention_output, nil}  # TODO: KV cache
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
  @spec tp_ffn(Axon.t(), struct(), map(), keyword()) :: Axon.t()
  def tp_ffn(hidden_state, spec, mesh, opts \\ []) do
    name = Keyword.get(opts, :name, "ffn")
    intermediate_size = spec.intermediate_size

    # Gate projection (column-parallel)
    gate = hidden_state
    |> TPLayers.column_parallel_dense(intermediate_size,
        mesh: mesh,
        name: "#{name}.gate")
    |> Axon.activation(:silu)

    # Up projection (column-parallel)
    up = TPLayers.column_parallel_dense(hidden_state, intermediate_size,
      mesh: mesh,
      name: "#{name}.intermediate"
    )

    # Element-wise multiply (local)
    hidden_state = Axon.multiply(gate, up)

    # Down projection (row-parallel with all-reduce)
    TPLayers.row_parallel_dense(hidden_state, spec.hidden_size,
      mesh: mesh,
      name: "#{name}.output"
    )
  end

  # Helper functions

  # Reshape for multi-head attention: {batch, seq, hidden} -> {batch, seq, heads, head_size}
  defp reshape_for_attention(tensor, num_heads, head_size) do
    Axon.nx(tensor, fn x ->
      {batch, seq, _} = Nx.shape(x)
      Nx.reshape(x, {batch, seq, num_heads, head_size})
    end)
  end

  # Flatten heads: {batch, seq, heads, head_size} -> {batch, seq, hidden}
  defp flatten_heads(tensor) do
    Axon.nx(tensor, fn x ->
      {batch, seq, heads, head_size} = Nx.shape(x)
      Nx.reshape(x, {batch, seq, heads * head_size})
    end)
  end

  # Repeat KV heads for grouped-query attention
  defp repeat_kv(tensor, repeat_factor) do
    Axon.nx(tensor, fn x ->
      {batch, seq, heads, head_size} = Nx.shape(x)
      x
      |> Nx.reshape({batch, seq, heads, 1, head_size})
      |> Nx.tile([1, 1, 1, repeat_factor, 1])
      |> Nx.reshape({batch, seq, heads * repeat_factor, head_size})
    end)
  end

  # Standard scaled dot-product attention
  defp compute_attention(query, key, value, spec) do
    # query/key/value shape: {batch, seq, heads, head_size}
    head_size = div(spec.hidden_size, spec.num_attention_heads)
    scale = :math.sqrt(head_size)

    # Get window size for sliding window attention (Mistral)
    window_size = Map.get(spec, :sliding_window, nil)

    Axon.layer(
      fn q, k, v, _opts ->
        # Transpose to {batch, heads, seq, head_size}
        q = Nx.transpose(q, axes: [0, 2, 1, 3])
        k = Nx.transpose(k, axes: [0, 2, 1, 3])
        v = Nx.transpose(v, axes: [0, 2, 1, 3])

        # Compute attention scores
        scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1])
        scores = Nx.divide(scores, scale)

        # Apply causal mask
        {_batch, _heads, seq_len, _} = Nx.shape(scores)
        causal_mask = create_causal_mask(seq_len, window_size)
        scores = Nx.select(causal_mask, scores, Nx.Constants.neg_infinity())

        # Softmax
        weights = Nx.softmax(scores, axis: -1)

        # Apply attention to values
        output = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

        # Transpose back to {batch, seq, heads, head_size}
        Nx.transpose(output, axes: [0, 2, 1, 3])
      end,
      [query, key, value],
      name: "attention"
    )
  end

  # Create causal attention mask (optionally with sliding window)
  defp create_causal_mask(seq_len, window_size) do
    # Standard causal mask
    mask = Nx.greater_equal(
      Nx.iota({seq_len, seq_len}, axis: 0),
      Nx.iota({seq_len, seq_len}, axis: 1)
    )

    # Apply sliding window if specified
    if window_size do
      window_mask = Nx.less_equal(
        Nx.subtract(
          Nx.iota({seq_len, seq_len}, axis: 0),
          Nx.iota({seq_len, seq_len}, axis: 1)
        ),
        window_size
      )
      Nx.logical_and(mask, window_mask)
    else
      mask
    end
  end
end
