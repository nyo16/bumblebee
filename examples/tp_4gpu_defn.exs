# 4-GPU Tensor Parallel Text Generation using Nx.Defn
#
# This version uses Nx.Defn for automatic compilation caching:
# - First call: JIT compiles (~4-5s per unique shape)
# - Subsequent calls: Uses cached compilation (~6ms per layer)
#
# Key difference from tp_4gpu_qwen3.exs:
# - Uses Nx.Defn instead of manual MLIR/SPMD building
# - Automatic caching by input shapes
# - Cleaner, more maintainable code
#
# Usage:
#   LAYERS=36 TOKENS=20 mix run examples/tp_4gpu_defn.exs

IO.puts("=" |> String.duplicate(70))
IO.puts("4-GPU TP Generation with Nx.Defn (Auto-Caching)")
IO.puts("=" |> String.duplicate(70))

# Configure EXLA
Nx.default_backend({EXLA.Backend, client: :cuda})

# =============================================================================
# TRANSFORMER OPERATIONS - All defined with Nx.Defn for auto-caching
# =============================================================================

defmodule TPTransformer do
  @moduledoc """
  Tensor Parallel Transformer operations using Nx.Defn.

  Each function is JIT compiled on first call with a specific shape,
  then cached for subsequent calls with the same shape.
  """

  import Nx.Defn

  # -------------------------------------------------------------------------
  # RMSNorm - Layer normalization without mean centering
  # -------------------------------------------------------------------------
  defn rms_norm(x, weight, eps \\ 1.0e-6) do
    # x: {batch, seq, hidden}
    # weight: {hidden}
    variance = Nx.mean(x * x, axes: [-1], keep_axes: true)
    x * Nx.rsqrt(variance + eps) * weight
  end

  # -------------------------------------------------------------------------
  # Rotary Position Embeddings (RoPE)
  # -------------------------------------------------------------------------
  defn apply_rope(x, cos, sin) do
    # x: {batch, heads, seq, head_dim}
    # cos, sin: {1, 1, seq, head_dim}
    {_batch, _heads, _seq, head_dim} = Nx.shape(x)
    half_dim = div(head_dim, 2)

    # Split into first and second half
    x1 = Nx.slice_along_axis(x, 0, half_dim, axis: 3)
    x2 = Nx.slice_along_axis(x, half_dim, half_dim, axis: 3)

    cos1 = Nx.slice_along_axis(cos, 0, half_dim, axis: 3)
    cos2 = Nx.slice_along_axis(cos, half_dim, half_dim, axis: 3)
    sin1 = Nx.slice_along_axis(sin, 0, half_dim, axis: 3)
    sin2 = Nx.slice_along_axis(sin, half_dim, half_dim, axis: 3)

    # Apply rotation
    rotated_x1 = x1 * cos1 - x2 * sin1
    rotated_x2 = x1 * sin2 + x2 * cos2

    Nx.concatenate([rotated_x1, rotated_x2], axis: 3)
  end

  # -------------------------------------------------------------------------
  # Softmax with numerical stability
  # -------------------------------------------------------------------------
  defn stable_softmax(x) do
    max_val = Nx.reduce_max(x, axes: [-1], keep_axes: true)
    exp_x = Nx.exp(x - max_val)
    sum_exp = Nx.sum(exp_x, axes: [-1], keep_axes: true)
    exp_x / sum_exp
  end

  # -------------------------------------------------------------------------
  # Attention with causal mask (for prefill)
  # -------------------------------------------------------------------------
  defn attention_prefill(q, k, v, scale, opts \\ []) do
    # q: {batch, local_heads, seq, head_dim}
    # k, v: {batch, local_kv_heads, seq, head_dim}
    # For GQA: local_heads = local_kv_heads * num_groups

    num_groups = opts[:num_groups]
    {batch, local_kv_heads, seq, head_dim} = Nx.shape(k)
    local_heads = local_kv_heads * num_groups

    # Expand K, V for grouped query attention
    k_expanded = Nx.broadcast(
      Nx.reshape(k, {batch, local_kv_heads, 1, seq, head_dim}),
      {batch, local_kv_heads, num_groups, seq, head_dim}
    )
    k_expanded = Nx.reshape(k_expanded, {batch, local_heads, seq, head_dim})

    v_expanded = Nx.broadcast(
      Nx.reshape(v, {batch, local_kv_heads, 1, seq, head_dim}),
      {batch, local_kv_heads, num_groups, seq, head_dim}
    )
    v_expanded = Nx.reshape(v_expanded, {batch, local_heads, seq, head_dim})

    # Q @ K^T
    k_t = Nx.transpose(k_expanded, axes: [0, 1, 3, 2])
    scores = Nx.dot(q, [3], [0, 1], k_t, [2], [0, 1]) * scale

    # Causal mask
    rows = Nx.iota({seq, 1}, type: :s32)
    cols = Nx.iota({1, seq}, type: :s32)
    mask = Nx.select(Nx.greater_equal(rows, cols), 0.0, -1.0e9)
    mask = Nx.broadcast(mask, {batch, local_heads, seq, seq})
    scores = scores + mask

    # Softmax
    weights = stable_softmax(scores)

    # Attention @ V
    Nx.dot(weights, [3], [0, 1], v_expanded, [2], [0, 1])
  end

  # -------------------------------------------------------------------------
  # Attention with KV cache (for decode - single token)
  # -------------------------------------------------------------------------
  defn attention_decode(q, k_cache, v_cache, k_new, v_new, position, scale, opts \\ []) do
    # q: {batch, local_heads, 1, head_dim}
    # k_cache, v_cache: {batch, local_kv_heads, max_seq, head_dim}
    # k_new, v_new: {batch, local_kv_heads, 1, head_dim}
    # position: scalar (current position index)

    num_groups = opts[:num_groups]
    {batch, local_kv_heads, max_seq, head_dim} = Nx.shape(k_cache)
    local_heads = local_kv_heads * num_groups

    # Update cache at position
    indices = [0, 0, position, 0]
    k_cache_updated = Nx.put_slice(k_cache, indices, k_new)
    v_cache_updated = Nx.put_slice(v_cache, indices, v_new)

    # Expand K, V for GQA
    k_expanded = Nx.broadcast(
      Nx.reshape(k_cache_updated, {batch, local_kv_heads, 1, max_seq, head_dim}),
      {batch, local_kv_heads, num_groups, max_seq, head_dim}
    )
    k_expanded = Nx.reshape(k_expanded, {batch, local_heads, max_seq, head_dim})

    v_expanded = Nx.broadcast(
      Nx.reshape(v_cache_updated, {batch, local_kv_heads, 1, max_seq, head_dim}),
      {batch, local_kv_heads, num_groups, max_seq, head_dim}
    )
    v_expanded = Nx.reshape(v_expanded, {batch, local_heads, max_seq, head_dim})

    # Q @ K^T  (only attend to positions 0..position)
    k_t = Nx.transpose(k_expanded, axes: [0, 1, 3, 2])
    scores = Nx.dot(q, [3], [0, 1], k_t, [2], [0, 1]) * scale

    # Position-based mask (mask future positions)
    positions = Nx.iota({max_seq}, type: :s32)
    valid_mask = Nx.less_equal(positions, position)
    mask = Nx.select(valid_mask, 0.0, -1.0e9)
    mask = Nx.reshape(mask, {1, 1, 1, max_seq})
    scores = scores + mask

    # Softmax and output
    weights = stable_softmax(scores)
    output = Nx.dot(weights, [3], [0, 1], v_expanded, [2], [0, 1])

    {output, k_cache_updated, v_cache_updated}
  end

  # -------------------------------------------------------------------------
  # FFN with SiLU activation (Qwen3 style)
  # -------------------------------------------------------------------------
  defn ffn(x, gate_w, up_w, down_w) do
    # x: {batch, seq, hidden}
    # gate_w, up_w: {hidden, intermediate}
    # down_w: {intermediate, hidden}

    gate = Nx.dot(x, [2], gate_w, [0])
    up = Nx.dot(x, [2], up_w, [0])

    # SiLU activation: x * sigmoid(x)
    hidden = Nx.sigmoid(gate) * gate * up

    Nx.dot(hidden, [2], down_w, [0])
  end

  # -------------------------------------------------------------------------
  # Full transformer layer (prefill)
  # -------------------------------------------------------------------------
  defn transformer_layer_prefill(
    hidden,
    input_norm_w,
    q_w, k_w, v_w, o_w,
    q_norm_w, k_norm_w,
    post_attn_norm_w,
    gate_w, up_w, down_w,
    rope_cos, rope_sin,
    opts
  ) do
    head_dim = opts[:head_dim]
    local_heads = opts[:local_heads]
    local_kv_heads = opts[:local_kv_heads]
    num_groups = div(local_heads, local_kv_heads)
    eps = opts[:eps]
    scale = 1.0 / Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(hidden)))

    {batch, seq, _hidden} = Nx.shape(hidden)

    # Input normalization
    normed = rms_norm(hidden, input_norm_w, eps)

    # QKV projections
    q = Nx.dot(normed, [2], q_w, [0])
    k = Nx.dot(normed, [2], k_w, [0])
    v = Nx.dot(normed, [2], v_w, [0])

    # Reshape for attention
    q = Nx.reshape(q, {batch, seq, local_heads, head_dim})
    k = Nx.reshape(k, {batch, seq, local_kv_heads, head_dim})
    v = Nx.reshape(v, {batch, seq, local_kv_heads, head_dim})

    # Transpose to [batch, heads, seq, head_dim]
    q = Nx.transpose(q, axes: [0, 2, 1, 3])
    k = Nx.transpose(k, axes: [0, 2, 1, 3])
    v = Nx.transpose(v, axes: [0, 2, 1, 3])

    # QK Norm (Qwen3 specific)
    q = rms_norm_per_head(q, q_norm_w, eps)
    k = rms_norm_per_head(k, k_norm_w, eps)

    # Apply RoPE
    q = apply_rope(q, rope_cos, rope_sin)
    k = apply_rope(k, rope_cos, rope_sin)

    # Attention
    attn_out = attention_prefill(q, k, v, scale, num_groups: num_groups)

    # Reshape back
    attn_out = Nx.transpose(attn_out, axes: [0, 2, 1, 3])
    attn_out = Nx.reshape(attn_out, {batch, seq, local_heads * head_dim})

    # Output projection
    attn_out = Nx.dot(attn_out, [2], o_w, [0])

    # Residual connection
    hidden = hidden + attn_out

    # Post-attention norm and FFN
    normed = rms_norm(hidden, post_attn_norm_w, eps)
    ffn_out = ffn(normed, gate_w, up_w, down_w)

    # Final residual
    hidden + ffn_out
  end

  # Helper: RMSNorm applied per-head (for QK norm)
  defn rms_norm_per_head(x, weight, eps) do
    # x: {batch, heads, seq, head_dim}
    # weight: {head_dim}
    variance = Nx.mean(x * x, axes: [-1], keep_axes: true)
    x * Nx.rsqrt(variance + eps) * weight
  end

  # -------------------------------------------------------------------------
  # Compute RoPE frequencies
  # -------------------------------------------------------------------------
  def compute_rope_frequencies(seq_len, head_dim, theta \\ 10000.0, type \\ :f32) do
    half_dim = div(head_dim, 2)

    # Compute inverse frequencies
    inv_freq = for i <- 0..(half_dim - 1) do
      1.0 / :math.pow(theta, i / half_dim)
    end
    inv_freq = Nx.tensor(inv_freq, type: type)

    # Compute position indices
    positions = Nx.iota({seq_len}, type: type)

    # Outer product: positions × inv_freq
    freqs = Nx.outer(positions, inv_freq)

    # Duplicate for full head_dim
    cos = Nx.cos(freqs)
    sin = Nx.sin(freqs)

    cos_full = Nx.concatenate([cos, cos], axis: 1)
    sin_full = Nx.concatenate([sin, sin], axis: 1)

    # Reshape to {1, 1, seq, head_dim} for broadcasting
    cos_full = Nx.reshape(cos_full, {1, 1, seq_len, head_dim})
    sin_full = Nx.reshape(sin_full, {1, 1, seq_len, head_dim})

    {cos_full, sin_full}
  end
end

# =============================================================================
# MAIN SCRIPT
# =============================================================================

# Configuration
tp_size = 4
num_layers = String.to_integer(System.get_env("LAYERS", "2"))
max_new_tokens = String.to_integer(System.get_env("TOKENS", "10"))
max_seq_len = String.to_integer(System.get_env("MAX_SEQ", "64"))
batch_size = 1

IO.puts("\nConfiguration:")
IO.puts("  TP size: #{tp_size}")
IO.puts("  Layers: #{num_layers}")
IO.puts("  Max new tokens: #{max_new_tokens}")
IO.puts("  Max sequence length: #{max_seq_len}")

# Load model
IO.puts("\n" <> "-" |> String.duplicate(70))
IO.puts("Loading Qwen3-4B-Instruct model...")
IO.puts("-" |> String.duplicate(70))

model_id = "Qwen/Qwen3-4B-Instruct-2507"
{:ok, %{params: params_state, spec: spec}} = Bumblebee.load_model({:hf, model_id})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, model_id})

# Bumblebee uses flattened param keys like "decoder.blocks.0.self_attention.query"
param_data = params_state.data

# Model dimensions
hidden_size = spec.hidden_size
num_heads = spec.num_attention_heads
num_kv_heads = spec.num_key_value_heads
head_dim = spec.attention_head_size
intermediate_size = spec.intermediate_size
vocab_size = spec.vocab_size
rope_theta = spec.rotary_embedding_base || 5_000_000.0

# Local dimensions for TP
local_heads = div(num_heads, tp_size)
local_kv_heads = div(num_kv_heads, tp_size)
local_q_size = local_heads * head_dim
local_kv_size = local_kv_heads * head_dim
local_intermediate = div(intermediate_size, tp_size)

IO.puts("  Hidden size: #{hidden_size}")
IO.puts("  Heads: #{num_heads} (#{local_heads} per GPU)")
IO.puts("  KV heads: #{num_kv_heads} (#{local_kv_heads} per GPU)")
IO.puts("  Head dim: #{head_dim}")
IO.puts("  Intermediate: #{intermediate_size} (#{local_intermediate} per GPU)")

# Extract and shard weights for GPU 0 (for single-GPU test)
IO.puts("\n" <> "-" |> String.duplicate(70))
IO.puts("Extracting weights for GPU 0...")
IO.puts("-" |> String.duplicate(70))

gpu_id = 0

# Helper to shard weights
shard_weight = fn weight, dim, shard_size, shard_idx ->
  start = shard_idx * shard_size
  case dim do
    0 -> Nx.slice(weight, [start, 0], [shard_size, elem(Nx.shape(weight), 1)])
    1 -> Nx.slice(weight, [0, start], [elem(Nx.shape(weight), 0), shard_size])
  end
end

# Extract layer weights using flattened keys
layer_weights = for layer_idx <- 0..(num_layers - 1) do
  prefix = "decoder.blocks.#{layer_idx}"

  %{
    input_norm: param_data["#{prefix}.self_attention_norm"]["weight"] |> Nx.backend_transfer(),
    q_w: shard_weight.(param_data["#{prefix}.self_attention.query"]["kernel"] |> Nx.backend_transfer(), 1, local_q_size, gpu_id),
    k_w: shard_weight.(param_data["#{prefix}.self_attention.key"]["kernel"] |> Nx.backend_transfer(), 1, local_kv_size, gpu_id),
    v_w: shard_weight.(param_data["#{prefix}.self_attention.value"]["kernel"] |> Nx.backend_transfer(), 1, local_kv_size, gpu_id),
    o_w: shard_weight.(param_data["#{prefix}.self_attention.output"]["kernel"] |> Nx.backend_transfer(), 0, local_q_size, gpu_id),
    q_norm: param_data["#{prefix}.self_attention.query_norm"]["weight"] |> Nx.backend_transfer(),
    k_norm: param_data["#{prefix}.self_attention.key_norm"]["weight"] |> Nx.backend_transfer(),
    post_attn_norm: param_data["#{prefix}.output_norm"]["weight"] |> Nx.backend_transfer(),
    gate_w: shard_weight.(param_data["#{prefix}.ffn.gate"]["kernel"] |> Nx.backend_transfer(), 1, local_intermediate, gpu_id),
    up_w: shard_weight.(param_data["#{prefix}.ffn.intermediate"]["kernel"] |> Nx.backend_transfer(), 1, local_intermediate, gpu_id),
    down_w: shard_weight.(param_data["#{prefix}.ffn.output"]["kernel"] |> Nx.backend_transfer(), 0, local_intermediate, gpu_id),
  }
end

embed_w = param_data["embedder.token_embedding"]["kernel"] |> Nx.backend_transfer()
final_norm_w = param_data["output_norm"]["weight"] |> Nx.backend_transfer()
# Qwen3 ties embeddings - use embed_w transposed for LM head
lm_head_w = embed_w

IO.puts("  Extracted #{num_layers} layers")

# Tokenize prompt
prompt = System.get_env("PROMPT", "The capital of France is")
IO.puts("\n" <> "-" |> String.duplicate(70))
IO.puts("Prompt: \"#{prompt}\"")
IO.puts("-" |> String.duplicate(70))

inputs = Bumblebee.apply_tokenizer(tokenizer, prompt)
input_ids = inputs["input_ids"] |> Nx.backend_transfer()
{1, prompt_len} = Nx.shape(input_ids)
IO.puts("  Token IDs: #{inspect(Nx.to_flat_list(input_ids))}")
IO.puts("  Prompt length: #{prompt_len}")

# Compute RoPE frequencies
{rope_cos, rope_sin} = TPTransformer.compute_rope_frequencies(max_seq_len, head_dim, rope_theta, :f32)

# =============================================================================
# PREFILL PHASE
# =============================================================================
IO.puts("\n" <> "-" |> String.duplicate(70))
IO.puts("Prefill Phase (first call includes JIT compilation)")
IO.puts("-" |> String.duplicate(70))

# Embedding lookup
hidden = Nx.take(embed_w, Nx.flatten(input_ids), axis: 0)
hidden = Nx.reshape(hidden, {batch_size, prompt_len, hidden_size})

# Slice RoPE for prompt length
rope_cos_prefill = Nx.slice(rope_cos, [0, 0, 0, 0], [1, 1, prompt_len, head_dim])
rope_sin_prefill = Nx.slice(rope_sin, [0, 0, 0, 0], [1, 1, prompt_len, head_dim])

IO.puts("  Running #{num_layers} transformer layers...")
start_time = System.monotonic_time(:millisecond)

# Run through layers
hidden = Enum.reduce(Enum.with_index(layer_weights), hidden, fn {weights, idx}, h ->
  if rem(idx, 4) == 0, do: IO.write("  Layer #{idx}...")

  result = TPTransformer.transformer_layer_prefill(
    h,
    weights.input_norm,
    weights.q_w, weights.k_w, weights.v_w, weights.o_w,
    weights.q_norm, weights.k_norm,
    weights.post_attn_norm,
    weights.gate_w, weights.up_w, weights.down_w,
    rope_cos_prefill, rope_sin_prefill,
    head_dim: head_dim,
    local_heads: local_heads,
    local_kv_heads: local_kv_heads,
    eps: 1.0e-6
  )

  if rem(idx, 4) == 0, do: IO.puts(" done")
  result
end)

# Final norm and LM head
hidden = TPTransformer.rms_norm(hidden, final_norm_w)
# Take last token's hidden state
last_hidden = Nx.slice(hidden, [0, prompt_len - 1, 0], [batch_size, 1, hidden_size])
last_hidden = Nx.reshape(last_hidden, {batch_size, hidden_size})
# Qwen3 ties embeddings: embed_w is {vocab, hidden}, so contract on axis 1 of both
# last_hidden {batch, hidden} @ embed_w^T {hidden, vocab} -> {batch, vocab}
logits = Nx.dot(last_hidden, [1], lm_head_w, [1])

# Get next token
next_token_tensor = Nx.argmax(logits, axis: 1) |> Nx.backend_transfer()
next_token = next_token_tensor[0] |> Nx.to_number()

prefill_time = System.monotonic_time(:millisecond) - start_time
IO.puts("\n  Prefill completed in #{prefill_time} ms")
IO.puts("  First token: #{next_token}")

# Decode token
token_str = Bumblebee.Tokenizer.decode(tokenizer, [next_token])
IO.puts("  Decoded: \"#{token_str}\"")

# =============================================================================
# DECODE PHASE (with caching)
# =============================================================================
IO.puts("\n" <> "-" |> String.duplicate(70))
IO.puts("Decode Phase (subsequent calls use cached compilation)")
IO.puts("-" |> String.duplicate(70))

# Run a few more tokens with the same function (should be cached now)
IO.puts("  Generating #{max_new_tokens - 1} more tokens...")

{generated_tokens, decode_times} = Enum.reduce(1..(max_new_tokens - 1), {[next_token], []}, fn i, {tokens, times} ->
  start = System.monotonic_time(:millisecond)

  # Simple decode: just run prefill again for demonstration
  # (In production, would use KV cache)
  current_token = List.last(tokens)
  current_ids = Nx.tensor([[current_token]])
  hidden = Nx.take(embed_w, Nx.flatten(current_ids), axis: 0)
  hidden = Nx.reshape(hidden, {batch_size, 1, hidden_size})

  # Use position-specific RoPE (simplified - just position i + prompt_len)
  pos = prompt_len + i - 1
  rope_cos_decode = Nx.slice(rope_cos, [0, 0, pos, 0], [1, 1, 1, head_dim])
  rope_sin_decode = Nx.slice(rope_sin, [0, 0, pos, 0], [1, 1, 1, head_dim])

  hidden = Enum.reduce(layer_weights, hidden, fn weights, h ->
    TPTransformer.transformer_layer_prefill(
      h,
      weights.input_norm,
      weights.q_w, weights.k_w, weights.v_w, weights.o_w,
      weights.q_norm, weights.k_norm,
      weights.post_attn_norm,
      weights.gate_w, weights.up_w, weights.down_w,
      rope_cos_decode, rope_sin_decode,
      head_dim: head_dim,
      local_heads: local_heads,
      local_kv_heads: local_kv_heads,
      eps: 1.0e-6
    )
  end)

  hidden = TPTransformer.rms_norm(hidden, final_norm_w)
  hidden = Nx.reshape(hidden, {batch_size, hidden_size})
  logits = Nx.dot(hidden, [1], lm_head_w, [1])

  next_token_tensor = Nx.argmax(logits, axis: 1) |> Nx.backend_transfer()
  new_token = next_token_tensor[0] |> Nx.to_number()

  elapsed = System.monotonic_time(:millisecond) - start
  IO.puts("  Token #{i + 1}: #{elapsed} ms")

  {tokens ++ [new_token], times ++ [elapsed]}
end)

# =============================================================================
# SUMMARY
# =============================================================================
IO.puts("\n" <> "=" |> String.duplicate(70))
IO.puts("SUMMARY")
IO.puts("=" |> String.duplicate(70))

avg_decode = if length(decode_times) > 0, do: Enum.sum(decode_times) / length(decode_times), else: 0

IO.puts("  Prefill time (includes compilation): #{prefill_time} ms")
IO.puts("  Avg decode time per token: #{Float.round(avg_decode, 1)} ms")
IO.puts("  Estimated tokens/second: #{Float.round(1000 / max(avg_decode, 1), 1)}")
IO.puts("")
IO.puts("  Generated tokens: #{inspect(generated_tokens)}")
full_text = Bumblebee.Tokenizer.decode(tokenizer, generated_tokens)
IO.puts("  Full output: \"#{prompt}#{full_text}\"")
IO.puts("")
IO.puts("Note: This is single-GPU (GPU 0 shard only).")
IO.puts("Full TP would require all-reduce between layers.")
