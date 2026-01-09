# How Tensor Parallelism Works

This document provides a detailed walkthrough of the implementation.

## Overview

The TP implementation has two phases:
1. **Prefill**: Process the entire prompt, initialize KV cache
2. **Decode**: Generate one token at a time using cached K/V

```
┌─────────────────────────────────────────────────────────────────┐
│                        GENERATION FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Prompt: "What is the capital of France?"                       │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  PREFILL PHASE                                          │    │
│  │  - Process all prompt tokens at once                    │    │
│  │  - Initialize KV cache with prompt's K/V                │    │
│  │  - Output: first generated token + KV cache            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  DECODE PHASE (repeated)                                │    │
│  │  - Process single new token                             │    │
│  │  - Attend to all cached K/V + new K/V                  │    │
│  │  - Update cache with new K/V                           │    │
│  │  - Output: next token                                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  Output: "The capital of France is Paris."                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. EXLA.SPMD Module

The SPMD module enables building and running multi-replica executables:

```elixir
# Build an SPMD executable
spmd = EXLA.SPMD.build(
  input_typespecs,    # List of input type specifications
  output_typespecs,   # List of output type specifications
  fn builder ->
    # MLIR builder function
    args = Function.get_arguments(builder)
    # ... build computation graph ...
    [result]
  end,
  num_replicas: 4,    # Number of GPUs
  client: :cuda       # GPU client
)

# Run on all replicas
# replica_inputs is a list of input lists, one per replica
results = EXLA.SPMD.run(spmd, replica_inputs)
```

### 2. Weight Sharding

Weights are extracted and sharded for each GPU:

```elixir
# Extract layer parameters for each GPU shard
extract_layer_params = fn param_data, layer_idx, shard_idx ->
  layer_prefix = "decoder.blocks.#{layer_idx}"

  # Attention weights - shard by heads
  q_proj = param_data["#{layer_prefix}.self_attention.query"]["kernel"]
  k_proj = param_data["#{layer_prefix}.self_attention.key"]["kernel"]
  v_proj = param_data["#{layer_prefix}.self_attention.value"]["kernel"]
  o_proj = param_data["#{layer_prefix}.self_attention.output"]["kernel"]

  # Calculate shard dimensions
  local_heads = num_heads / tp_size
  local_kv_heads = num_kv_heads / tp_size

  # Shard Q projection (column-parallel)
  q_start = shard_idx * local_heads * head_dim
  q_end = (shard_idx + 1) * local_heads * head_dim
  q_shard = Nx.slice(q_proj, [0, q_start], [hidden_size, local_heads * head_dim])

  # Shard K projection
  k_start = shard_idx * local_kv_heads * head_dim
  k_end = (shard_idx + 1) * local_kv_heads * head_dim
  k_shard = Nx.slice(k_proj, [0, k_start], [hidden_size, local_kv_heads * head_dim])

  # Similarly for V, O, FFN weights...
  %{q: q_shard, k: k_shard, v: v_shard, ...}
end
```

### 3. Prefill SPMD Builder

The prefill phase processes all prompt tokens at once:

```elixir
build_prefill_spmd = fn batch_size, prompt_len, max_seq_len ->
  # Define input typespecs
  input_ids_typespec = EXLA.Typespec.tensor({:s, 32}, {batch_size, prompt_len})
  embed_typespec = EXLA.Typespec.tensor({:f, 32}, {vocab_size, hidden_size})
  # ... more typespecs for weights, norms, etc.

  EXLA.SPMD.build(
    [input_ids_typespec, embed_typespec, ...],
    output_typespecs,
    fn builder ->
      args = Function.get_arguments(builder)
      [input_ids, embed_weights, ...] = args

      # 1. Embedding lookup
      hidden_states = Value.gather(
        embed_weights,
        input_ids,
        2,                           # index_vector_dim
        [1, hidden_size],            # slice_sizes
        [prompt_len],                # offset_dims
        [0],                         # collapsed_slice_dims
        [0],                         # start_index_map
        output_typespec
      )

      # 2. For each layer
      {hidden_states, k_caches, v_caches} =
        Enum.reduce(0..(num_layers - 1), {hidden_states, [], []}, fn layer_idx, {h, ks, vs} ->
          # RMSNorm
          h_normed = rms_norm.(h, layer_weights.input_norm)

          # Attention with KV cache initialization
          {attn_out, k_cache, v_cache} = attention_prefill.(h_normed, layer_weights)

          # Residual + RMSNorm + FFN + Residual
          h = Value.add(h, attn_out, hidden_typespec)
          h_normed = rms_norm.(h, layer_weights.post_attn_norm)
          ffn_out = ffn.(h_normed, layer_weights)
          h = Value.add(h, ffn_out, hidden_typespec)

          {h, ks ++ [k_cache], vs ++ [v_cache]}
        end)

      # 3. Final norm + LM head
      h_final = rms_norm.(hidden_states, final_norm)
      logits = lm_head.(h_final)

      [logits | k_caches ++ v_caches]
    end,
    num_replicas: tp_size
  )
end
```

### 4. RMSNorm Implementation

Real RMSNorm using MLIR reduce operations:

```elixir
rms_norm = fn x, weight, typespec ->
  {batch, seq_len, hidden} = typespec.shape
  scalar_typespec = EXLA.Typespec.tensor({:f, 32}, {})
  reduce_typespec = EXLA.Typespec.tensor({:f, 32}, {batch, seq_len})

  # x²
  x_squared = Value.multiply(x, x, typespec)

  # Build reduction region for sum
  {region, [lhs, rhs]} = Function.push_region(builder, [scalar_typespec, scalar_typespec])
  sum_result = Value.add(lhs, rhs, scalar_typespec)
  Value.return(builder, [sum_result])
  Function.pop_region(builder)

  # Reduce sum over hidden dimension
  zero = Value.constant(builder, [0.0], scalar_typespec)
  [sum_squared] = Value.reduce(region, [zero], [x_squared], [2], [reduce_typespec])

  # Mean and rsqrt
  hidden_const = Value.constant(builder, [hidden * 1.0], scalar_typespec)
  hidden_broadcast = Value.broadcast_in_dim(hidden_const, [], reduce_typespec)
  mean_squared = Value.divide(sum_squared, hidden_broadcast, reduce_typespec)

  epsilon = Value.constant(builder, [1.0e-6], scalar_typespec)
  epsilon_broadcast = Value.broadcast_in_dim(epsilon, [], reduce_typespec)
  mean_plus_eps = Value.add(mean_squared, epsilon_broadcast, reduce_typespec)

  rsqrt = Value.rsqrt(mean_plus_eps, reduce_typespec)

  # Normalize and scale
  rsqrt_3d = Value.reshape(rsqrt, EXLA.Typespec.tensor({:f, 32}, {batch, seq_len, 1}))
  rsqrt_broadcast = Value.broadcast_in_dim(rsqrt_3d, [0, 1, 2], typespec)
  normalized = Value.multiply(x, rsqrt_broadcast, typespec)

  weight_broadcast = Value.broadcast_in_dim(weight, [2], typespec)
  Value.multiply(normalized, weight_broadcast, typespec)
end
```

### 5. Attention with All-Reduce

Attention computation with tensor parallelism:

```elixir
attention = fn hidden, weights, k_cache_in, v_cache_in, position ->
  # Q, K, V projections (column-parallel - each GPU has local heads)
  q = Value.dot_general(hidden, weights.q, {[2], [0]}, :default, q_typespec)
  k = Value.dot_general(hidden, weights.k, {[2], [0]}, :default, k_typespec)
  v = Value.dot_general(hidden, weights.v, {[2], [0]}, :default, v_typespec)

  # Reshape to [batch, local_heads, seq, head_dim]
  q = Value.reshape(q, q_4d_typespec)
  k = Value.reshape(k, k_4d_typespec)
  v = Value.reshape(v, v_4d_typespec)

  # Apply RoPE (Rotary Position Embeddings)
  q = apply_rope.(q, rope_cos, rope_sin)
  k = apply_rope.(k, rope_cos, rope_sin)

  # Update KV cache
  k_cache = Value.dynamic_update_slice(k_cache_in, k, [0, 0, position, 0], k_cache_typespec)
  v_cache = Value.dynamic_update_slice(v_cache_in, v, [0, 0, position, 0], v_cache_typespec)

  # Attention scores: Q @ K^T
  k_transposed = Value.transpose(k_cache, [0, 1, 3, 2], k_transposed_typespec)
  scores = Value.dot_general(q, k_transposed, {[3], [0, 1], [2], [0, 1]}, :default, scores_typespec)

  # Scale
  scale = 1.0 / :math.sqrt(head_dim)
  scores = Value.multiply(scores, scale_broadcast, scores_typespec)

  # Causal mask
  scores = Value.add(scores, causal_mask, scores_typespec)

  # Softmax (using reduce for numerical stability)
  attn_weights = softmax.(scores)

  # Attention output: weights @ V
  attn_out = Value.dot_general(attn_weights, v_cache, {[3], [0, 1], [2], [0, 1]}, :default, attn_4d_typespec)

  # Reshape back to [batch, seq, local_heads * head_dim]
  attn_out = Value.reshape(attn_out, attn_2d_typespec)

  # O projection (row-parallel)
  output = Value.dot_general(attn_out, weights.o, {[2], [0]}, :default, output_typespec)

  # All-reduce to combine partial results from all GPUs
  output = Value.all_reduce(output, :sum, [[0, 1, 2, 3]], channel_id, output_typespec)

  {output, k_cache, v_cache}
end
```

### 6. FFN (SwiGLU) with All-Reduce

```elixir
ffn = fn hidden, weights ->
  # Column-parallel: gate and up projections
  gate = Value.dot_general(hidden, weights.gate, {[2], [0]}, :default, gate_typespec)
  up = Value.dot_general(hidden, weights.up, {[2], [0]}, :default, up_typespec)

  # SiLU activation on gate
  gate_sigmoid = Value.logistic(gate, gate_typespec)
  gate_silu = Value.multiply(gate, gate_sigmoid, gate_typespec)

  # Element-wise multiply
  intermediate = Value.multiply(gate_silu, up, gate_typespec)

  # Row-parallel: down projection
  output = Value.dot_general(intermediate, weights.down, {[2], [0]}, :default, output_typespec)

  # All-reduce to combine partial results
  Value.all_reduce(output, :sum, [[0, 1, 2, 3]], channel_id, output_typespec)
end
```

### 7. Decode Phase

The decode phase processes one token at a time:

```elixir
build_decode_spmd = fn batch_size, max_seq_len ->
  # Position input (which cache slot to update)
  position_typespec = EXLA.Typespec.tensor({:s, 32}, {})

  # Single token input
  token_typespec = EXLA.Typespec.tensor({:s, 32}, {batch_size, 1})

  EXLA.SPMD.build(
    [token_typespec, position_typespec, k_cache_typespec, v_cache_typespec, ...],
    output_typespecs,
    fn builder ->
      args = Function.get_arguments(builder)
      [token, position, k_caches, v_caches, ...] = args

      # Embedding lookup for single token
      hidden = Value.gather(embed_weights, token, ...)

      # Process through layers, updating cache
      {hidden, new_k_caches, new_v_caches} =
        Enum.reduce(layers, {hidden, [], []}, fn layer_idx, {h, ks, vs} ->
          k_cache = Enum.at(k_caches, layer_idx)
          v_cache = Enum.at(v_caches, layer_idx)

          # Attention with cache update
          {attn_out, new_k_cache, new_v_cache} =
            attention_decode.(h, weights, k_cache, v_cache, position)

          # ... rest of layer ...

          {h, ks ++ [new_k_cache], vs ++ [new_v_cache]}
        end)

      # Return logits and updated caches
      [logits | new_k_caches ++ new_v_caches]
    end,
    num_replicas: tp_size
  )
end
```

### 8. RoPE Implementation

Rotary Position Embeddings:

```elixir
apply_rope = fn x, cos, sin, typespec ->
  # x shape: [batch, heads, seq, head_dim]
  # cos/sin shape: [1, 1, seq, head_dim]

  # Split into pairs: [x0, x1, x2, x3, ...] -> [x0, x2, ...], [x1, x3, ...]
  x_even = Value.slice(x, [0, 0, 0, 0], [batch, heads, seq, head_dim/2], ...)
  x_odd = Value.slice(x, [0, 0, 0, head_dim/2], [batch, heads, seq, head_dim/2], ...)

  # Rotate: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
  x_even_rot = Value.subtract(
    Value.multiply(x_even, cos_even, ...),
    Value.multiply(x_odd, sin_even, ...),
    ...
  )

  x_odd_rot = Value.add(
    Value.multiply(x_even, sin_odd, ...),
    Value.multiply(x_odd, cos_odd, ...),
    ...
  )

  # Interleave back
  Value.concatenate([x_even_rot, x_odd_rot], 3, typespec)
end
```

## Generation Loop

```elixir
# 1. Build SPMDs
prefill_spmd = build_prefill_spmd.(batch_size, prompt_len, max_seq_len)
decode_spmd = build_decode_spmd.(batch_size, max_seq_len)

# 2. Prefill phase
replica_inputs = for shard_idx <- 0..(tp_size - 1) do
  [input_ids, embed_weights, layer_params[shard_idx], ...]
end
[logits | caches] = EXLA.SPMD.run(prefill_spmd, replica_inputs)

# 3. Sample first token
first_token = sample_token.(logits, temperature, top_k, top_p)
generated_tokens = [first_token]
current_position = prompt_len

# 4. Decode loop
{final_tokens, final_caches} = Enum.reduce_while(
  1..(max_new_tokens - 1),
  {generated_tokens, caches, current_position},
  fn _i, {tokens, caches, position} ->
    # Prepare single token input
    token_tensor = Nx.tensor([[List.last(tokens)]])

    # Run decode
    replica_inputs = for shard_idx <- 0..(tp_size - 1) do
      [token_tensor, position, caches[shard_idx], ...]
    end
    [logits | new_caches] = EXLA.SPMD.run(decode_spmd, replica_inputs)

    # Sample next token
    next_token = sample_token.(logits, temperature, top_k, top_p)

    # Check for EOS
    if next_token == eos_token_id do
      {:halt, {tokens, new_caches, position}}
    else
      {:cont, {tokens ++ [next_token], new_caches, position + 1}}
    end
  end
)

# 5. Decode tokens to text
Bumblebee.Tokenizer.decode(tokenizer, final_tokens)
```

## Summary

The implementation follows these key patterns:
1. **SPMD execution** - Same code runs on all GPUs
2. **Column→Row parallelism** - Minimize all-reduce calls
3. **Pre-allocated KV cache** - No recompilation per token
4. **Proper operations** - Real RMSNorm, softmax, RoPE via MLIR
5. **Position-based masking** - Causal attention with cache
