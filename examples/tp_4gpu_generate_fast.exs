# 4-GPU Tensor Parallel Text Generation - FAST VERSION
#
# This version uses PADDED SEQUENCES for fast generation:
# - Build decode SPMD once with fixed max_seq_len
# - Use dynamic_update_slice to update KV cache
# - Reuse compiled SPMD for all tokens (~100x faster)
#
# Usage:
#   LAYERS=2 mix run examples/tp_4gpu_generate_fast.exs

Nx.default_backend(EXLA.Backend)

IO.puts("=" |> String.duplicate(70))
IO.puts("4-GPU TP Generation - FAST VERSION (Padded Sequences)")
IO.puts("=" |> String.duplicate(70))

alias EXLA.MLIR.{Function, Value}

# Configuration
tp_size = 4
max_new_tokens = String.to_integer(System.get_env("TOKENS", "20"))
max_seq_len = String.to_integer(System.get_env("MAX_SEQ", "128"))

# Model size
num_layers = String.to_integer(System.get_env("LAYERS", "2"))

IO.puts("\nConfiguration:")
IO.puts("  TP size: #{tp_size}")
IO.puts("  Max new tokens: #{max_new_tokens}")
IO.puts("  Max sequence length: #{max_seq_len}")

# ----------------------------------------------------------
# Step 1: Load model and tokenizer
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 1: Loading model and tokenizer")
IO.puts("-" |> String.duplicate(70))

IO.puts("  Loading Mistral 7B...")
{:ok, %{params: params, spec: spec}} = Bumblebee.load_model({:hf, "mistralai/Mistral-7B-v0.1"})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "mistralai/Mistral-7B-v0.1"})

vocab_size = spec.vocab_size
hidden_size = spec.hidden_size
intermediate_size = spec.intermediate_size
num_heads = spec.num_attention_heads
num_kv_heads = spec.num_key_value_heads
head_dim = div(hidden_size, num_heads)
rms_norm_eps = spec.layer_norm_epsilon
rope_theta = spec.rotary_embedding_base || 10_000.0

# Tensor parallelism configuration
local_heads = div(num_heads, tp_size)
local_kv_heads = div(num_kv_heads, tp_size)
kv_head_repeat = div(local_heads, local_kv_heads)
local_q_size = local_heads * head_dim
local_kv_size = local_kv_heads * head_dim
local_intermediate = div(intermediate_size, tp_size)

IO.puts("  Model loaded!")
IO.puts("  Using #{num_layers} layers")

# Precompute RoPE embeddings for max_seq_len
compute_rope_embeddings = fn max_positions ->
  half_dim = div(head_dim, 2)
  inv_freq = for i <- 0..(half_dim - 1) do
    1.0 / :math.pow(rope_theta, 2 * i / head_dim)
  end
  inv_freq_tensor = Nx.tensor(inv_freq, type: :f32)
  positions = Nx.iota({max_positions}, type: :f32)
  freqs = Nx.outer(positions, inv_freq_tensor)
  cos_freqs = Nx.cos(freqs)
  sin_freqs = Nx.sin(freqs)
  cos_embed = Nx.concatenate([cos_freqs, cos_freqs], axis: 1)
  sin_embed = Nx.concatenate([sin_freqs, sin_freqs], axis: 1)
  {cos_embed, sin_embed}
end

{rope_cos_full, rope_sin_full} = compute_rope_embeddings.(max_seq_len)

# ----------------------------------------------------------
# Step 2: Extract and shard parameters
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 2: Extracting and sharding parameters")
IO.puts("-" |> String.duplicate(70))

param_data = params.data

embed_tokens = param_data["embedder.token_embedding"]["kernel"]
final_norm_weight = param_data["output_norm"]["weight"]
lm_head_kernel = param_data["language_modeling_head.output"]["kernel"]

layer_params = for layer_idx <- 0..(num_layers - 1) do
  layer_prefix = "decoder.blocks.#{layer_idx}"

  sa_norm = param_data["#{layer_prefix}.self_attention_norm"]["weight"]
  ffn_norm = param_data["#{layer_prefix}.output_norm"]["weight"]

  q_kernel = param_data["#{layer_prefix}.self_attention.query"]["kernel"]
  k_kernel = param_data["#{layer_prefix}.self_attention.key"]["kernel"]
  v_kernel = param_data["#{layer_prefix}.self_attention.value"]["kernel"]
  o_kernel = param_data["#{layer_prefix}.self_attention.output"]["kernel"]

  q_shards = for i <- 0..(tp_size - 1), do: Nx.slice(q_kernel, [0, i * local_q_size], [hidden_size, local_q_size])
  k_shards = for i <- 0..(tp_size - 1), do: Nx.slice(k_kernel, [0, i * local_kv_size], [hidden_size, local_kv_size])
  v_shards = for i <- 0..(tp_size - 1), do: Nx.slice(v_kernel, [0, i * local_kv_size], [hidden_size, local_kv_size])
  o_shards = for i <- 0..(tp_size - 1), do: Nx.slice(o_kernel, [i * local_q_size, 0], [local_q_size, hidden_size])

  gate_kernel = param_data["#{layer_prefix}.ffn.gate"]["kernel"]
  up_kernel = param_data["#{layer_prefix}.ffn.intermediate"]["kernel"]
  down_kernel = param_data["#{layer_prefix}.ffn.output"]["kernel"]

  gate_shards = for i <- 0..(tp_size - 1), do: Nx.slice(gate_kernel, [0, i * local_intermediate], [hidden_size, local_intermediate])
  up_shards = for i <- 0..(tp_size - 1), do: Nx.slice(up_kernel, [0, i * local_intermediate], [hidden_size, local_intermediate])
  down_shards = for i <- 0..(tp_size - 1), do: Nx.slice(down_kernel, [i * local_intermediate, 0], [local_intermediate, hidden_size])

  %{
    sa_norm: sa_norm, ffn_norm: ffn_norm,
    q_shards: q_shards, k_shards: k_shards, v_shards: v_shards, o_shards: o_shards,
    gate_shards: gate_shards, up_shards: up_shards, down_shards: down_shards
  }
end

IO.puts("  Extracted #{num_layers} layer parameters")

# ----------------------------------------------------------
# Step 3: Build FAST decode SPMD (compile once, reuse)
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 3: Building FAST decode SPMD (padded sequences)")
IO.puts("-" |> String.duplicate(70))

batch_size = 1

# Build decode SPMD with FIXED max_seq_len - compile once, reuse for all tokens
build_fast_decode_spmd = fn ->
  # Inputs
  input_ids_typespec = EXLA.Typespec.tensor({:s, 32}, {batch_size, 1})
  position_typespec = EXLA.Typespec.tensor({:s, 32}, {})  # Scalar position
  embed_typespec = EXLA.Typespec.tensor({:f, 32}, {vocab_size, hidden_size})
  norm_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size})
  lm_head_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, vocab_size})

  # RoPE for single position
  rope_cos_typespec = EXLA.Typespec.tensor({:f, 32}, {1, head_dim})
  rope_sin_typespec = EXLA.Typespec.tensor({:f, 32}, {1, head_dim})

  q_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_q_size})
  k_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_kv_size})
  v_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_kv_size})
  o_typespec = EXLA.Typespec.tensor({:f, 32}, {local_q_size, hidden_size})
  gate_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_intermediate})
  up_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_intermediate})
  down_typespec = EXLA.Typespec.tensor({:f, 32}, {local_intermediate, hidden_size})

  layer_param_typespecs = List.duplicate([
    norm_typespec, norm_typespec,
    q_typespec, k_typespec, v_typespec, o_typespec,
    gate_typespec, up_typespec, down_typespec
  ], num_layers) |> List.flatten()

  # FIXED size KV cache - this is the key optimization!
  k_cache_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, max_seq_len, head_dim})
  v_cache_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, max_seq_len, head_dim})
  cache_typespecs = List.duplicate([k_cache_typespec, v_cache_typespec], num_layers) |> List.flatten()

  input_typespecs = [input_ids_typespec, position_typespec, embed_typespec, norm_typespec, lm_head_typespec,
                    rope_cos_typespec, rope_sin_typespec] ++ layer_param_typespecs ++ cache_typespecs

  # Outputs: logits + updated caches (same shape)
  output_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, vocab_size})
  output_typespecs = [output_typespec] ++ cache_typespecs

  replica_groups = [Enum.to_list(0..(tp_size - 1))]

  EXLA.SPMD.build(input_typespecs, output_typespecs, fn builder ->
    args = Function.get_arguments(builder)
    [input_ids, position, embed_w, final_norm_w, lm_head_w, rope_cos, rope_sin | rest_args] = args

    num_param_args = num_layers * 9
    {layer_param_args, cache_args} = Enum.split(rest_args, num_param_args)

    layer_weights = Enum.chunk_every(layer_param_args, 9)
    |> Enum.map(fn [sa_norm, ffn_norm, q, k, v, o, gate, up, down] ->
      %{sa_norm: sa_norm, ffn_norm: ffn_norm, q: q, k: k, v: v, o: o, gate: gate, up: up, down: down}
    end)

    cache_pairs = Enum.chunk_every(cache_args, 2)

    layer_weights_and_caches = Enum.zip(layer_weights, cache_pairs)
    |> Enum.map(fn {weights, [k_cache, v_cache]} -> {weights, k_cache, v_cache} end)

    hidden_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, hidden_size})

    # Embedding lookup
    hidden_states = Value.gather(
      embed_w, input_ids, 2, [1, hidden_size], [2], [0], [0], hidden_typespec
    )

    # RoPE helper for decode
    apply_rope = fn x, cos_embed, sin_embed, x_typespec ->
      {batch_local, num_heads_local, seq_len_local, _} = x_typespec.shape
      cos_4d = Value.reshape(cos_embed, EXLA.Typespec.tensor({:f, 32}, {1, 1, 1, head_dim}))
      sin_4d = Value.reshape(sin_embed, EXLA.Typespec.tensor({:f, 32}, {1, 1, 1, head_dim}))
      cos_broadcast = Value.broadcast_in_dim(cos_4d, [0, 1, 2, 3], x_typespec)
      sin_broadcast = Value.broadcast_in_dim(sin_4d, [0, 1, 2, 3], x_typespec)

      half_dim = div(head_dim, 2)
      first_half_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_local, num_heads_local, seq_len_local, half_dim})
      second_half_typespec = first_half_typespec

      x_first = Value.slice(x, [0, 0, 0, 0], [batch_local, num_heads_local, seq_len_local, half_dim], [1, 1, 1, 1], first_half_typespec)
      x_second = Value.slice(x, [0, 0, 0, half_dim], [batch_local, num_heads_local, seq_len_local, head_dim], [1, 1, 1, 1], second_half_typespec)

      neg_one = Value.constant(builder, [-1.0], EXLA.Typespec.tensor({:f, 32}, {}))
      neg_one_broadcast = Value.broadcast_in_dim(neg_one, [], second_half_typespec)
      x_second_neg = Value.multiply(x_second, neg_one_broadcast, second_half_typespec)

      x_rotated = Value.concatenate([x_second_neg, x_first], 3, x_typespec)
      x_cos = Value.multiply(x, cos_broadcast, x_typespec)
      x_rot_sin = Value.multiply(x_rotated, sin_broadcast, x_typespec)
      Value.add(x_cos, x_rot_sin, x_typespec)
    end

    # RMSNorm helper
    rms_norm = fn x, weight, typespec ->
      scalar_typespec = EXLA.Typespec.tensor({:f, 32}, {})
      reduce_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1})

      x_squared = Value.multiply(x, x, typespec)

      {region, [lhs, rhs]} = Function.push_region(builder, [scalar_typespec, scalar_typespec])
      sum_result = Value.add(lhs, rhs, scalar_typespec)
      Value.return(builder, [sum_result])
      Function.pop_region(builder)

      zero = Value.constant(builder, [0.0], scalar_typespec)
      [sum_squared] = Value.reduce(region, [zero], [x_squared], [2], [reduce_typespec])

      hidden_size_c = Value.constant(builder, [hidden_size * 1.0], scalar_typespec)
      hidden_size_b = Value.broadcast_in_dim(hidden_size_c, [], reduce_typespec)
      mean_squared = Value.divide(sum_squared, hidden_size_b, reduce_typespec)

      eps_c = Value.constant(builder, [rms_norm_eps], scalar_typespec)
      eps_b = Value.broadcast_in_dim(eps_c, [], reduce_typespec)
      mean_squared_eps = Value.add(mean_squared, eps_b, reduce_typespec)

      rsqrt = Value.rsqrt(mean_squared_eps, reduce_typespec)
      rsqrt_3d = Value.reshape(rsqrt, EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, 1}))
      rsqrt_broadcast = Value.broadcast_in_dim(rsqrt_3d, [0, 1, 2], typespec)

      normalized = Value.multiply(x, rsqrt_broadcast, typespec)
      weight_broadcast = Value.broadcast_in_dim(weight, [2], typespec)
      Value.multiply(normalized, weight_broadcast, typespec)
    end

    # Create position indices for dynamic slicing
    zero_idx = Value.constant(builder, [0], EXLA.Typespec.tensor({:s, 32}, {}))

    # Process layers
    {final_hidden, updated_caches} = Enum.reduce(layer_weights_and_caches, {hidden_states, []}, fn {weights, k_cache, v_cache}, {hidden, acc_caches} ->
      normed = rms_norm.(hidden, weights.sa_norm, hidden_typespec)

      q_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_q_size})
      k_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_kv_size})
      v_proj_typespec = k_proj_typespec

      q = Value.dot_general(normed, weights.q, {[2], [], [0], []}, :default, q_proj_typespec)
      k_new = Value.dot_general(normed, weights.k, {[2], [], [0], []}, :default, k_proj_typespec)
      v_new = Value.dot_general(normed, weights.v, {[2], [], [0], []}, :default, v_proj_typespec)

      # Reshape to [batch, heads, 1, head_dim]
      q_4d_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_heads, 1, head_dim})
      k_4d_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, head_dim})
      v_4d_typespec = k_4d_typespec

      q_reshaped = Value.reshape(q, EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_heads, head_dim}))
      k_reshaped = Value.reshape(k_new, EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_kv_heads, head_dim}))
      v_reshaped = Value.reshape(v_new, EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_kv_heads, head_dim}))

      q_transposed = Value.transpose(q_reshaped, [0, 2, 1, 3], q_4d_typespec)
      k_transposed = Value.transpose(k_reshaped, [0, 2, 1, 3], k_4d_typespec)
      v_transposed = Value.transpose(v_reshaped, [0, 2, 1, 3], v_4d_typespec)

      # Apply RoPE
      q_rope = apply_rope.(q_transposed, rope_cos, rope_sin, q_4d_typespec)
      k_rope = apply_rope.(k_transposed, rope_cos, rope_sin, k_4d_typespec)

      # Update KV cache at current position using dynamic_update_slice
      k_cache_updated = Value.dynamic_update_slice(k_cache, k_rope, [zero_idx, zero_idx, position, zero_idx], k_cache_typespec)
      v_cache_updated = Value.dynamic_update_slice(v_cache, v_transposed, [zero_idx, zero_idx, position, zero_idx], v_cache_typespec)

      # For attention, we need to attend to positions 0..position (inclusive)
      # Create attention scores with full cache
      k_for_attn = Value.transpose(k_cache_updated, [0, 1, 3, 2], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, head_dim, max_seq_len}))

      q_grouped = Value.reshape(q_rope, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, 1, head_dim}))
      k_expanded = Value.reshape(k_for_attn, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, head_dim, max_seq_len}))
      v_expanded = Value.reshape(v_cache_updated, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, max_seq_len, head_dim}))

      k_broadcast = Value.broadcast_in_dim(k_expanded, [0, 1, 2, 3, 4], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, head_dim, max_seq_len}))
      v_broadcast = Value.broadcast_in_dim(v_expanded, [0, 1, 2, 3, 4], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, max_seq_len, head_dim}))

      scores_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, 1, max_seq_len})
      scores = Value.dot_general(q_grouped, k_broadcast, {[4], [0, 1, 2], [3], [0, 1, 2]}, :default, scores_typespec)

      scale_value = 1.0 / :math.sqrt(head_dim)
      scale_c = Value.constant(builder, [scale_value], EXLA.Typespec.tensor({:f, 32}, {}))
      scale_b = Value.broadcast_in_dim(scale_c, [], scores_typespec)
      scores_scaled = Value.multiply(scores, scale_b, scores_typespec)

      # Create causal mask: mask out positions > current_position
      # position_indices from 0 to max_seq_len-1
      pos_indices = Value.iota(builder, 0, EXLA.Typespec.tensor({:s, 32}, {max_seq_len}))
      # Map operand dim 0 (128) to result dim 4 (128), broadcast 1s in dims 0-3
      pos_indices_broadcast = Value.broadcast_in_dim(pos_indices, [4], EXLA.Typespec.tensor({:s, 32}, {1, 1, 1, 1, max_seq_len}))
      pos_indices_full = Value.broadcast_in_dim(pos_indices_broadcast, [0, 1, 2, 3, 4], EXLA.Typespec.tensor({:s, 32}, {batch_size, local_kv_heads, kv_head_repeat, 1, max_seq_len}))

      # Broadcast current position
      position_broadcast = Value.broadcast_in_dim(position, [], EXLA.Typespec.tensor({:s, 32}, {batch_size, local_kv_heads, kv_head_repeat, 1, max_seq_len}))

      # Mask: attend if pos_indices <= position
      mask_pred = Value.less_equal(pos_indices_full, position_broadcast, EXLA.Typespec.tensor({:pred, 8}, {batch_size, local_kv_heads, kv_head_repeat, 1, max_seq_len}))

      neg_inf = Value.constant(builder, [-1.0e9], EXLA.Typespec.tensor({:f, 32}, {}))
      zero_f = Value.constant(builder, [0.0], EXLA.Typespec.tensor({:f, 32}, {}))
      neg_inf_b = Value.broadcast_in_dim(neg_inf, [], scores_typespec)
      zero_b = Value.broadcast_in_dim(zero_f, [], scores_typespec)

      mask_values = Value.select(mask_pred, zero_b, neg_inf_b, scores_typespec)
      scores_masked = Value.add(scores_scaled, mask_values, scores_typespec)

      # Softmax
      scalar_typespec = EXLA.Typespec.tensor({:f, 32}, {})
      reduce_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, 1})

      {region, [lhs, rhs]} = Function.push_region(builder, [scalar_typespec, scalar_typespec])
      max_result = Value.max(lhs, rhs, scalar_typespec)
      Value.return(builder, [max_result])
      Function.pop_region(builder)

      neg_inf_init = Value.constant(builder, [-1.0e9], scalar_typespec)
      [max_scores] = Value.reduce(region, [neg_inf_init], [scores_masked], [4], [reduce_typespec])

      max_expanded = Value.reshape(max_scores, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, 1, 1}))
      max_broadcast = Value.broadcast_in_dim(max_expanded, [0, 1, 2, 3, 4], scores_typespec)
      scores_shifted = Value.subtract(scores_masked, max_broadcast, scores_typespec)
      scores_exp = Value.exp(scores_shifted, scores_typespec)

      {region, [lhs, rhs]} = Function.push_region(builder, [scalar_typespec, scalar_typespec])
      sum_result = Value.add(lhs, rhs, scalar_typespec)
      Value.return(builder, [sum_result])
      Function.pop_region(builder)

      zero_init = Value.constant(builder, [0.0], scalar_typespec)
      [sum_exp] = Value.reduce(region, [zero_init], [scores_exp], [4], [reduce_typespec])

      sum_expanded = Value.reshape(sum_exp, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, 1, 1}))
      sum_broadcast = Value.broadcast_in_dim(sum_expanded, [0, 1, 2, 3, 4], scores_typespec)
      attention_weights = Value.divide(scores_exp, sum_broadcast, scores_typespec)

      # Attention output
      attn_output_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, 1, head_dim})
      attn_output = Value.dot_general(attention_weights, v_broadcast, {[4], [0, 1, 2], [3], [0, 1, 2]}, :default, attn_output_typespec)

      attn_merged = Value.reshape(attn_output, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_heads, 1, head_dim}))
      attn_transposed = Value.transpose(attn_merged, [0, 2, 1, 3], EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_heads, head_dim}))
      attn_flat = Value.reshape(attn_transposed, EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_q_size}))

      attn_partial = Value.dot_general(attn_flat, weights.o, {[2], [], [0], []}, :default, hidden_typespec)
      attn_result = Value.all_reduce(attn_partial, :sum, replica_groups, hidden_typespec)

      after_attn = Value.add(hidden, attn_result, hidden_typespec)

      # FFN
      normed_ffn = rms_norm.(after_attn, weights.ffn_norm, hidden_typespec)

      intermediate_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_intermediate})

      gate_out = Value.dot_general(normed_ffn, weights.gate, {[2], [], [0], []}, :default, intermediate_typespec)
      up_out = Value.dot_general(normed_ffn, weights.up, {[2], [], [0], []}, :default, intermediate_typespec)

      gate_sigmoid = Value.sigmoid(gate_out, intermediate_typespec)
      gate_silu = Value.multiply(gate_out, gate_sigmoid, intermediate_typespec)
      combined = Value.multiply(gate_silu, up_out, intermediate_typespec)

      ffn_partial = Value.dot_general(combined, weights.down, {[2], [], [0], []}, :default, hidden_typespec)
      ffn_result = Value.all_reduce(ffn_partial, :sum, replica_groups, hidden_typespec)

      layer_output = Value.add(after_attn, ffn_result, hidden_typespec)

      {layer_output, acc_caches ++ [k_cache_updated, v_cache_updated]}
    end)

    # Final norm + LM head
    normed_output = rms_norm.(final_hidden, final_norm_w, hidden_typespec)
    last_hidden_2d = Value.reshape(normed_output, EXLA.Typespec.tensor({:f, 32}, {batch_size, hidden_size}))
    logits = Value.dot_general(last_hidden_2d, lm_head_w, {[1], [], [0], []}, :default, output_typespec)

    [logits] ++ updated_caches
  end, num_replicas: tp_size, client: :cuda)
end

IO.puts("  Building decode SPMD (this will be reused for all tokens)...")
{compile_time, decode_spmd} = :timer.tc(fn -> build_fast_decode_spmd.() end)
IO.puts("  Decode SPMD compiled in #{Float.round(compile_time / 1_000_000, 2)} seconds")

# ----------------------------------------------------------
# Step 4: Initialize KV caches (zeros)
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 4: Initializing KV caches")
IO.puts("-" |> String.duplicate(70))

initial_caches = for _layer <- 0..(num_layers - 1) do
  k_cache = Nx.broadcast(0.0, {batch_size, local_kv_heads, max_seq_len, head_dim}) |> Nx.as_type(:f32)
  v_cache = Nx.broadcast(0.0, {batch_size, local_kv_heads, max_seq_len, head_dim}) |> Nx.as_type(:f32)
  [k_cache, v_cache]
end |> List.flatten()

IO.puts("  Initialized #{num_layers} layer caches with shape [#{batch_size}, #{local_kv_heads}, #{max_seq_len}, #{head_dim}]")

# ----------------------------------------------------------
# Step 5: Tokenize prompt and run generation
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 5: Generating text")
IO.puts("-" |> String.duplicate(70))

prompt = System.get_env("PROMPT", "The capital of France is")
IO.puts("  Prompt: \"#{prompt}\"")

tokenized = Bumblebee.apply_tokenizer(tokenizer, prompt)
input_ids = tokenized["input_ids"] |> Nx.as_type(:s32)
prompt_tokens = Nx.to_flat_list(input_ids)
prompt_length = length(prompt_tokens)

IO.puts("  Prompt tokens: #{inspect(prompt_tokens)}")

# Prepare inputs function
prepare_inputs = fn token_id, position, caches ->
  token_tensor = Nx.tensor([[token_id]], type: :s32)
  position_tensor = Nx.tensor(position, type: :s32)

  # Get RoPE for this position
  rope_cos_pos = Nx.slice(rope_cos_full, [position, 0], [1, head_dim])
  rope_sin_pos = Nx.slice(rope_sin_full, [position, 0], [1, head_dim])

  for gpu <- 0..(tp_size - 1) do
    layer_params_for_gpu = Enum.flat_map(layer_params, fn layer ->
      [
        layer.sa_norm, layer.ffn_norm,
        Enum.at(layer.q_shards, gpu), Enum.at(layer.k_shards, gpu),
        Enum.at(layer.v_shards, gpu), Enum.at(layer.o_shards, gpu),
        Enum.at(layer.gate_shards, gpu), Enum.at(layer.up_shards, gpu),
        Enum.at(layer.down_shards, gpu)
      ]
    end)

    [token_tensor, position_tensor, embed_tokens, final_norm_weight, lm_head_kernel,
     rope_cos_pos, rope_sin_pos] ++ layer_params_for_gpu ++ caches
  end
end

# Process prompt tokens one by one to fill cache
IO.puts("\n  Processing prompt (#{prompt_length} tokens)...")
current_caches = initial_caches

{prefill_time, {_, current_caches}} = :timer.tc(fn ->
  Enum.reduce(Enum.with_index(prompt_tokens), {nil, current_caches}, fn {token, pos}, {_last_logits, caches} ->
    replica_inputs = prepare_inputs.(token, pos, caches)
    [[logits | new_caches] | _] = EXLA.SPMD.run(decode_spmd, replica_inputs)
    {logits, new_caches}
  end)
end)

IO.puts("  Prefill completed in #{Float.round(prefill_time / 1000, 2)} ms")

# Get first generated token
[[logits | _] | _] = EXLA.SPMD.run(decode_spmd, prepare_inputs.(List.last(prompt_tokens), prompt_length - 1, current_caches))
next_token = Nx.argmax(logits, axis: 1) |> Nx.to_flat_list() |> hd()

generated_tokens = [next_token]
IO.puts("\n  Generating #{max_new_tokens - 1} more tokens...")
IO.write("  Output: #{Bumblebee.Tokenizer.decode(tokenizer, [next_token])}")

# Generate remaining tokens - FAST because SPMD is already compiled!
{gen_time, {generated_tokens, _}} = :timer.tc(fn ->
  Enum.reduce(1..(max_new_tokens - 1), {generated_tokens, current_caches}, fn _i, {tokens, caches} ->
    last_token = List.last(tokens)
    current_pos = prompt_length + length(tokens) - 1

    replica_inputs = prepare_inputs.(last_token, current_pos, caches)
    [[new_logits | new_caches] | _] = EXLA.SPMD.run(decode_spmd, replica_inputs)

    new_token = Nx.argmax(new_logits, axis: 1) |> Nx.to_flat_list() |> hd()
    IO.write(Bumblebee.Tokenizer.decode(tokenizer, [new_token]))

    {tokens ++ [new_token], new_caches}
  end)
end)

IO.puts("")
IO.puts("\n  Generation completed in #{Float.round(gen_time / 1000, 2)} ms")
IO.puts("  Average per token: #{Float.round(gen_time / 1000 / (max_new_tokens - 1), 2)} ms")

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
IO.puts("\n" <> ("=" |> String.duplicate(70)))
IO.puts("Summary")
IO.puts("=" |> String.duplicate(70))

full_text = Bumblebee.Tokenizer.decode(tokenizer, prompt_tokens ++ generated_tokens)

IO.puts("""

✓ FAST Generation with Padded Sequences!

Prompt: "#{prompt}"

Full Generated Text:
#{full_text}

Performance:
  - SPMD Compile: #{Float.round(compile_time / 1_000_000, 2)} seconds (one-time)
  - Prefill: #{Float.round(prefill_time / 1000, 2)} ms for #{prompt_length} tokens
  - Generation: #{Float.round(gen_time / 1000, 2)} ms for #{max_new_tokens - 1} tokens
  - Average per token: #{Float.round(gen_time / 1000 / (max_new_tokens - 1), 2)} ms

Model: Mistral 7B (#{num_layers} layers)
TP Configuration: #{tp_size} GPUs
Max sequence length: #{max_seq_len}
""")
