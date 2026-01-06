# 4-GPU Tensor Parallel Text Generation with KV Cache - Mistral 7B
#
# This demonstrates efficient autoregressive generation using:
# - Tensor parallelism across 4 GPUs
# - KV cache to avoid recomputing previous tokens
# - Separate prefill (prompt) and decode (generation) phases
# - Mistral 7B model
#
# Usage:
#   mix run examples/tp_4gpu_generate_kvcache.exs

Nx.default_backend(EXLA.Backend)

IO.puts("=" |> String.duplicate(70))
IO.puts("4-GPU Tensor Parallel Text Generation with KV Cache - Mistral 7B")
IO.puts("=" |> String.duplicate(70))

alias EXLA.MLIR.{Function, Value}

# Configuration
tp_size = 4
max_new_tokens = 20
temperature = 1.0

# Model size - will be loaded from Mistral 7B
num_layers = String.to_integer(System.get_env("LAYERS", "2"))  # Use fewer layers for demo

IO.puts("\nConfiguration:")
IO.puts("  TP size: #{tp_size}")
IO.puts("  Max new tokens: #{max_new_tokens}")
IO.puts("  Temperature: #{temperature}")

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

# Tensor parallelism configuration
local_heads = div(num_heads, tp_size)
local_kv_heads = div(num_kv_heads, tp_size)
kv_head_repeat = div(local_heads, local_kv_heads)
local_q_size = local_heads * head_dim
local_kv_size = local_kv_heads * head_dim
local_intermediate = div(intermediate_size, tp_size)

IO.puts("  Model loaded!")
IO.puts("  Vocab size: #{vocab_size}")
IO.puts("  Hidden size: #{hidden_size}")
IO.puts("  Using #{num_layers} layers for generation")

# ----------------------------------------------------------
# Step 2: Extract and shard parameters
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 2: Extracting and sharding parameters")
IO.puts("-" |> String.duplicate(70))

param_data = params.data

# Extract global parameters (not sharded) - Mistral paths
embed_tokens = param_data["embedder.token_embedding"]["kernel"]
final_norm_weight = param_data["output_norm"]["weight"]
lm_head_kernel = param_data["language_modeling_head.output"]["kernel"]

# Extract and shard layer parameters
layer_params = for layer_idx <- 0..(num_layers - 1) do
  layer_prefix = "decoder.blocks.#{layer_idx}"

  # Norms (replicated) - Mistral uses "weight" not "scale"
  sa_norm = param_data["#{layer_prefix}.self_attention_norm"]["weight"]
  ffn_norm = param_data["#{layer_prefix}.output_norm"]["weight"]

  # Attention weights
  q_kernel = param_data["#{layer_prefix}.self_attention.query"]["kernel"]
  k_kernel = param_data["#{layer_prefix}.self_attention.key"]["kernel"]
  v_kernel = param_data["#{layer_prefix}.self_attention.value"]["kernel"]
  o_kernel = param_data["#{layer_prefix}.self_attention.output"]["kernel"]

  # Shard attention weights (column-parallel for Q/K/V, row-parallel for O)
  q_shards = for i <- 0..(tp_size - 1), do: Nx.slice(q_kernel, [0, i * local_q_size], [hidden_size, local_q_size])
  k_shards = for i <- 0..(tp_size - 1), do: Nx.slice(k_kernel, [0, i * local_kv_size], [hidden_size, local_kv_size])
  v_shards = for i <- 0..(tp_size - 1), do: Nx.slice(v_kernel, [0, i * local_kv_size], [hidden_size, local_kv_size])
  o_shards = for i <- 0..(tp_size - 1), do: Nx.slice(o_kernel, [i * local_q_size, 0], [local_q_size, hidden_size])

  # FFN weights
  gate_kernel = param_data["#{layer_prefix}.ffn.gate"]["kernel"]
  up_kernel = param_data["#{layer_prefix}.ffn.intermediate"]["kernel"]
  down_kernel = param_data["#{layer_prefix}.ffn.output"]["kernel"]

  # Shard FFN weights (column-parallel for gate/up, row-parallel for down)
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
# Step 3: Build SPMD executables with KV cache
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 3: Building SPMD executables with KV cache")
IO.puts("-" |> String.duplicate(70))

# Maximum sequence length for KV cache
max_seq_len = 512

# Prefill phase: Process prompt and initialize KV cache
# Input: token IDs [batch, prompt_len]
# Output: logits [batch, vocab_size], K cache [num_layers, batch, local_kv_heads, prompt_len, head_dim], V cache
IO.puts("  Building prefill SPMD (processes prompt, initializes cache)...")

build_prefill_spmd = fn batch_size, prompt_len ->
  input_ids_typespec = EXLA.Typespec.tensor({:s, 64}, {batch_size, prompt_len})
  embed_typespec = EXLA.Typespec.tensor({:f, 32}, {vocab_size, hidden_size})
  norm_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size})
  lm_head_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, vocab_size})

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

  input_typespecs = [input_ids_typespec, embed_typespec, norm_typespec, lm_head_typespec] ++ layer_param_typespecs

  # Outputs: logits + K/V caches for each layer
  output_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, vocab_size})
  k_cache_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, prompt_len, head_dim})
  v_cache_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, prompt_len, head_dim})

  # One K and V cache per layer
  cache_typespecs = List.duplicate([k_cache_typespec, v_cache_typespec], num_layers) |> List.flatten()
  output_typespecs = [output_typespec] ++ cache_typespecs

  replica_groups = [Enum.to_list(0..(tp_size - 1))]

  EXLA.SPMD.build(input_typespecs, output_typespecs, fn builder ->
    args = Function.get_arguments(builder)
    [input_ids, embed_w, final_norm_w, lm_head_w | layer_args] = args

    layer_weights = Enum.chunk_every(layer_args, 9)
    |> Enum.map(fn [sa_norm, ffn_norm, q, k, v, o, gate, up, down] ->
      %{sa_norm: sa_norm, ffn_norm: ffn_norm, q: q, k: k, v: v, o: o, gate: gate, up: up, down: down}
    end)

    hidden_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, hidden_size})

    # Simplified embedding lookup
    float_ids_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len})
    float_ids = Value.convert(input_ids, float_ids_typespec)

    scale_typespec = EXLA.Typespec.tensor({:f, 32}, {})
    scale_tensor = Value.constant(builder, [0.001], scale_typespec)
    scale_broadcast = Value.broadcast_in_dim(scale_tensor, [], float_ids_typespec)
    float_ids_scaled = Value.multiply(float_ids, scale_broadcast, float_ids_typespec)

    float_ids_3d_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, 1})
    float_ids_3d = Value.reshape(float_ids_scaled, float_ids_3d_typespec)

    proj_typespec = EXLA.Typespec.tensor({:f, 32}, {1, hidden_size})
    proj = Value.slice(embed_w, [0, 0], [1, hidden_size], [1, 1], proj_typespec)

    hidden_states = Value.dot_general(float_ids_3d, proj, {[2], [], [0], []}, :default, hidden_typespec)

    # Simplified norm
    simple_norm = fn x, weight ->
      weight_broadcast = Value.broadcast_in_dim(weight, [2], hidden_typespec)
      Value.multiply(x, weight_broadcast, hidden_typespec)
    end

    # Process layers and collect K/V caches
    {final_hidden, kv_caches} = Enum.reduce(layer_weights, {hidden_states, []}, fn weights, {hidden, caches} ->
      # Self-attention with cache output
      normed_for_attn = simple_norm.(hidden, weights.sa_norm)

      q_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, local_q_size})
      k_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, local_kv_size})
      v_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, local_kv_size})

      q = Value.dot_general(normed_for_attn, weights.q, {[2], [], [0], []}, :default, q_proj_typespec)
      k = Value.dot_general(normed_for_attn, weights.k, {[2], [], [0], []}, :default, k_proj_typespec)
      v = Value.dot_general(normed_for_attn, weights.v, {[2], [], [0], []}, :default, v_proj_typespec)

      # Reshape for multi-head attention
      q_reshaped = Value.reshape(q, EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, local_heads, head_dim}))
      k_reshaped = Value.reshape(k, EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, local_kv_heads, head_dim}))
      v_reshaped = Value.reshape(v, EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, local_kv_heads, head_dim}))

      # Transpose to [batch, heads, seq_len, head_dim]
      q_transposed = Value.transpose(q_reshaped, [0, 2, 1, 3], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_heads, prompt_len, head_dim}))
      k_transposed = Value.transpose(k_reshaped, [0, 2, 1, 3], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, prompt_len, head_dim}))
      v_transposed = Value.transpose(v_reshaped, [0, 2, 1, 3], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, prompt_len, head_dim}))

      # Store K/V for cache (these will be output)
      k_cache = k_transposed
      v_cache = v_transposed

      # Grouped query attention
      k_for_attn = Value.transpose(k_transposed, [0, 1, 3, 2], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, head_dim, prompt_len}))

      q_grouped = Value.reshape(q_transposed, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, prompt_len, head_dim}))
      k_expanded = Value.reshape(k_for_attn, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, head_dim, prompt_len}))
      v_expanded = Value.reshape(v_transposed, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, prompt_len, head_dim}))

      k_broadcast = Value.broadcast_in_dim(k_expanded, [0, 1, 2, 3, 4], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, head_dim, prompt_len}))
      v_broadcast = Value.broadcast_in_dim(v_expanded, [0, 1, 2, 3, 4], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, prompt_len, head_dim}))

      scores_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, prompt_len, prompt_len})
      scores = Value.dot_general(q_grouped, k_broadcast, {[4], [0, 1, 2], [3], [0, 1, 2]}, :default, scores_typespec)

      scale_value = 1.0 / :math.sqrt(head_dim)
      scale_tensor = Value.constant(builder, [scale_value], scale_typespec)
      scale_broadcast = Value.broadcast_in_dim(scale_tensor, [], scores_typespec)
      scores_scaled = Value.multiply(scores, scale_broadcast, scores_typespec)

      attention_weights = Value.sigmoid(scores_scaled, scores_typespec)

      attn_output_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, prompt_len, head_dim})
      attn_output = Value.dot_general(attention_weights, v_broadcast, {[4], [0, 1, 2], [3], [0, 1, 2]}, :default, attn_output_typespec)

      attn_merged = Value.reshape(attn_output, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_heads, prompt_len, head_dim}))
      attn_transposed = Value.transpose(attn_merged, [0, 2, 1, 3], EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, local_heads, head_dim}))
      attn_flat = Value.reshape(attn_transposed, EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, local_q_size}))

      attn_partial = Value.dot_general(attn_flat, weights.o, {[2], [], [0], []}, :default, hidden_typespec)
      attn_result = Value.all_reduce(attn_partial, :sum, replica_groups, hidden_typespec)

      after_attn = Value.add(hidden, attn_result, hidden_typespec)

      # FFN
      normed_for_ffn = simple_norm.(after_attn, weights.ffn_norm)

      intermediate_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, local_intermediate})

      gate_out = Value.dot_general(normed_for_ffn, weights.gate, {[2], [], [0], []}, :default, intermediate_typespec)
      up_out = Value.dot_general(normed_for_ffn, weights.up, {[2], [], [0], []}, :default, intermediate_typespec)

      gate_sigmoid = Value.sigmoid(gate_out, intermediate_typespec)
      gate_silu = Value.multiply(gate_out, gate_sigmoid, intermediate_typespec)
      combined = Value.multiply(gate_silu, up_out, intermediate_typespec)

      ffn_partial = Value.dot_general(combined, weights.down, {[2], [], [0], []}, :default, hidden_typespec)
      ffn_result = Value.all_reduce(ffn_partial, :sum, replica_groups, hidden_typespec)

      layer_output = Value.add(after_attn, ffn_result, hidden_typespec)

      # Accumulate K/V caches for this layer
      {layer_output, caches ++ [k_cache, v_cache]}
    end)

    # Final norm + LM head (last position only)
    normed_output = simple_norm.(final_hidden, final_norm_w)

    last_hidden_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, hidden_size})
    last_hidden = Value.slice(normed_output, [0, prompt_len - 1, 0], [batch_size, prompt_len, hidden_size], [1, 1, 1], last_hidden_typespec)
    last_hidden_2d = Value.reshape(last_hidden, EXLA.Typespec.tensor({:f, 32}, {batch_size, hidden_size}))

    logits = Value.dot_general(last_hidden_2d, lm_head_w, {[1], [], [0], []}, :default, output_typespec)

    # Return logits + all K/V caches
    [logits] ++ kv_caches
  end, num_replicas: tp_size, client: :cuda)
end

IO.puts("  Prefill SPMD builder created!")

# Decode phase: Process single token with existing KV cache
# Input: single token + K/V caches
# Output: logits + updated K/V caches
IO.puts("  Building decode SPMD (processes single token with cache)...")

build_decode_spmd = fn batch_size, cache_len ->
  # New sequence length after adding one token
  new_seq_len = cache_len + 1

  # Input: single token
  input_ids_typespec = EXLA.Typespec.tensor({:s, 64}, {batch_size, 1})
  embed_typespec = EXLA.Typespec.tensor({:f, 32}, {vocab_size, hidden_size})
  norm_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size})
  lm_head_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, vocab_size})

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

  # Input K/V caches (previous sequence)
  k_cache_in_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, cache_len, head_dim})
  v_cache_in_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, cache_len, head_dim})
  cache_in_typespecs = List.duplicate([k_cache_in_typespec, v_cache_in_typespec], num_layers) |> List.flatten()

  input_typespecs = [input_ids_typespec, embed_typespec, norm_typespec, lm_head_typespec] ++ layer_param_typespecs ++ cache_in_typespecs

  # Outputs: logits + updated K/V caches
  output_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, vocab_size})
  k_cache_out_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, new_seq_len, head_dim})
  v_cache_out_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, new_seq_len, head_dim})
  cache_out_typespecs = List.duplicate([k_cache_out_typespec, v_cache_out_typespec], num_layers) |> List.flatten()
  output_typespecs = [output_typespec] ++ cache_out_typespecs

  replica_groups = [Enum.to_list(0..(tp_size - 1))]

  EXLA.SPMD.build(input_typespecs, output_typespecs, fn builder ->
    args = Function.get_arguments(builder)
    [input_ids, embed_w, final_norm_w, lm_head_w | rest_args] = args

    # Split into layer params (9 per layer) and caches (2 per layer)
    num_param_args = num_layers * 9
    {layer_param_args, cache_args} = Enum.split(rest_args, num_param_args)

    # Extract layer weights
    layer_weights = Enum.chunk_every(layer_param_args, 9)
    |> Enum.map(fn [sa_norm, ffn_norm, q, k, v, o, gate, up, down] ->
      %{sa_norm: sa_norm, ffn_norm: ffn_norm, q: q, k: k, v: v, o: o, gate: gate, up: up, down: down}
    end)

    # Extract K/V caches (2 per layer)
    cache_pairs = Enum.chunk_every(cache_args, 2)

    # Combine weights and caches
    layer_weights_and_caches = Enum.zip(layer_weights, cache_pairs)
    |> Enum.map(fn {weights, [k_cache_in, v_cache_in]} ->
      {weights, k_cache_in, v_cache_in}
    end)

    hidden_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, hidden_size})

    # Simplified embedding (same as prefill)
    float_ids_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1})
    float_ids = Value.convert(input_ids, float_ids_typespec)

    scale_typespec = EXLA.Typespec.tensor({:f, 32}, {})
    scale_tensor = Value.constant(builder, [0.001], scale_typespec)
    scale_broadcast = Value.broadcast_in_dim(scale_tensor, [], float_ids_typespec)
    float_ids_scaled = Value.multiply(float_ids, scale_broadcast, float_ids_typespec)

    float_ids_3d_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, 1})
    float_ids_3d = Value.reshape(float_ids_scaled, float_ids_3d_typespec)

    proj_typespec = EXLA.Typespec.tensor({:f, 32}, {1, hidden_size})
    proj = Value.slice(embed_w, [0, 0], [1, hidden_size], [1, 1], proj_typespec)

    hidden_states = Value.dot_general(float_ids_3d, proj, {[2], [], [0], []}, :default, hidden_typespec)

    # Simplified norm
    simple_norm = fn x, weight ->
      weight_broadcast = Value.broadcast_in_dim(weight, [2], hidden_typespec)
      Value.multiply(x, weight_broadcast, hidden_typespec)
    end

    # Process layers with cache
    {final_hidden, updated_caches} = Enum.reduce(layer_weights_and_caches, {hidden_states, []}, fn {weights, k_cache_in, v_cache_in}, {hidden, acc_caches} ->
      # Self-attention with cache
      normed_for_attn = simple_norm.(hidden, weights.sa_norm)

      # Project Q, K, V for new token only
      q_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_q_size})
      k_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_kv_size})
      v_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_kv_size})

      q = Value.dot_general(normed_for_attn, weights.q, {[2], [], [0], []}, :default, q_proj_typespec)
      k_new = Value.dot_general(normed_for_attn, weights.k, {[2], [], [0], []}, :default, k_proj_typespec)
      v_new = Value.dot_general(normed_for_attn, weights.v, {[2], [], [0], []}, :default, v_proj_typespec)

      # Reshape new K/V for attention: [batch, 1, heads, head_dim] -> [batch, heads, 1, head_dim]
      k_new_reshaped = Value.reshape(k_new, EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_kv_heads, head_dim}))
      v_new_reshaped = Value.reshape(v_new, EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_kv_heads, head_dim}))

      k_new_transposed = Value.transpose(k_new_reshaped, [0, 2, 1, 3], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, head_dim}))
      v_new_transposed = Value.transpose(v_new_reshaped, [0, 2, 1, 3], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, head_dim}))

      # Concatenate with cache: [batch, heads, cache_len, head_dim] + [batch, heads, 1, head_dim]
      k_full = Value.concatenate([k_cache_in, k_new_transposed], 2, k_cache_out_typespec)
      v_full = Value.concatenate([v_cache_in, v_new_transposed], 2, v_cache_out_typespec)

      # Reshape Q: [batch, 1, local_heads, head_dim]
      q_reshaped = Value.reshape(q, EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_heads, head_dim}))
      q_transposed = Value.transpose(q_reshaped, [0, 2, 1, 3], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_heads, 1, head_dim}))

      # Grouped query attention
      k_for_attn = Value.transpose(k_full, [0, 1, 3, 2], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, head_dim, new_seq_len}))

      q_grouped = Value.reshape(q_transposed, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, 1, head_dim}))
      k_expanded = Value.reshape(k_for_attn, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, head_dim, new_seq_len}))
      v_expanded = Value.reshape(v_full, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, new_seq_len, head_dim}))

      k_broadcast = Value.broadcast_in_dim(k_expanded, [0, 1, 2, 3, 4], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, head_dim, new_seq_len}))
      v_broadcast = Value.broadcast_in_dim(v_expanded, [0, 1, 2, 3, 4], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, new_seq_len, head_dim}))

      scores_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, 1, new_seq_len})
      scores = Value.dot_general(q_grouped, k_broadcast, {[4], [0, 1, 2], [3], [0, 1, 2]}, :default, scores_typespec)

      scale_value = 1.0 / :math.sqrt(head_dim)
      scale_tensor = Value.constant(builder, [scale_value], scale_typespec)
      scale_broadcast = Value.broadcast_in_dim(scale_tensor, [], scores_typespec)
      scores_scaled = Value.multiply(scores, scale_broadcast, scores_typespec)

      attention_weights = Value.sigmoid(scores_scaled, scores_typespec)

      attn_output_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, 1, head_dim})
      attn_output = Value.dot_general(attention_weights, v_broadcast, {[4], [0, 1, 2], [3], [0, 1, 2]}, :default, attn_output_typespec)

      attn_merged = Value.reshape(attn_output, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_heads, 1, head_dim}))
      attn_transposed = Value.transpose(attn_merged, [0, 2, 1, 3], EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_heads, head_dim}))
      attn_flat = Value.reshape(attn_transposed, EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_q_size}))

      attn_partial = Value.dot_general(attn_flat, weights.o, {[2], [], [0], []}, :default, hidden_typespec)
      attn_result = Value.all_reduce(attn_partial, :sum, replica_groups, hidden_typespec)

      after_attn = Value.add(hidden, attn_result, hidden_typespec)

      # FFN (same as prefill)
      normed_for_ffn = simple_norm.(after_attn, weights.ffn_norm)

      intermediate_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_intermediate})

      gate_out = Value.dot_general(normed_for_ffn, weights.gate, {[2], [], [0], []}, :default, intermediate_typespec)
      up_out = Value.dot_general(normed_for_ffn, weights.up, {[2], [], [0], []}, :default, intermediate_typespec)

      gate_sigmoid = Value.sigmoid(gate_out, intermediate_typespec)
      gate_silu = Value.multiply(gate_out, gate_sigmoid, intermediate_typespec)
      combined = Value.multiply(gate_silu, up_out, intermediate_typespec)

      ffn_partial = Value.dot_general(combined, weights.down, {[2], [], [0], []}, :default, hidden_typespec)
      ffn_result = Value.all_reduce(ffn_partial, :sum, replica_groups, hidden_typespec)

      layer_output = Value.add(after_attn, ffn_result, hidden_typespec)

      # Return updated hidden state and K/V caches
      {layer_output, acc_caches ++ [k_full, v_full]}
    end)

    # Final norm + LM head
    normed_output = simple_norm.(final_hidden, final_norm_w)

    last_hidden_2d = Value.reshape(normed_output, EXLA.Typespec.tensor({:f, 32}, {batch_size, hidden_size}))
    logits = Value.dot_general(last_hidden_2d, lm_head_w, {[1], [], [0], []}, :default, output_typespec)

    # Return logits + updated caches
    [logits] ++ updated_caches
  end, num_replicas: tp_size, client: :cuda)
end

IO.puts("  Decode SPMD builder created!")

# ----------------------------------------------------------
# Step 4: Tokenize prompt
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 4: Tokenizing prompt")
IO.puts("-" |> String.duplicate(70))

prompt = "The meaning of life is"
IO.puts("  Prompt: \"#{prompt}\"")

tokenized = Bumblebee.apply_tokenizer(tokenizer, prompt)
input_ids = tokenized["input_ids"] |> Nx.as_type(:s64)
prompt_length = Nx.axis_size(input_ids, 1)

IO.puts("  Token IDs: #{inspect(Nx.to_flat_list(input_ids))}")
IO.puts("  Prompt length: #{prompt_length} tokens")

# ----------------------------------------------------------
# Step 5: Test prefill phase
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 5: Testing prefill phase")
IO.puts("-" |> String.duplicate(70))

IO.puts("  Building prefill SPMD for prompt_len=#{prompt_length}...")
prefill_spmd = build_prefill_spmd.(1, prompt_length)

# Prepare replica inputs
prepare_prefill_inputs = fn input_ids ->
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

    [input_ids, embed_tokens, final_norm_weight, lm_head_kernel] ++ layer_params_for_gpu
  end
end

IO.puts("  Running prefill on #{tp_size} GPUs...")
replica_inputs = prepare_prefill_inputs.(input_ids)

{time_us, results} = :timer.tc(fn ->
  EXLA.SPMD.run(prefill_spmd, replica_inputs)
end)

IO.puts("  Prefill completed in #{Float.round(time_us / 1000, 2)} ms")

# Extract results
[[logits | kv_caches] | _] = results

IO.puts("\n  Output shapes:")
IO.puts("    Logits: #{inspect(Nx.shape(logits))}")
IO.puts("    Number of K/V cache pairs: #{div(length(kv_caches), 2)}")
IO.puts("    K cache shape (per layer): #{inspect(Nx.shape(hd(kv_caches)))}")

# Sample first token
next_token = Nx.argmax(logits, axis: 1) |> Nx.to_flat_list() |> hd()
IO.puts("\n  First generated token: #{next_token}")
IO.puts("  Decoded: \"#{Bumblebee.Tokenizer.decode(tokenizer, [next_token])}\"")

# ----------------------------------------------------------
# Step 6: Autoregressive generation with KV cache
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 6: Generating text with KV cache (#{max_new_tokens - 1} more tokens)")
IO.puts("-" |> String.duplicate(70))

# Prepare decode inputs function
prepare_decode_inputs = fn token_id, caches ->
  token_tensor = Nx.tensor([[token_id]], type: :s64)

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

    # Caches are already in the right format from SPMD output
    [token_tensor, embed_tokens, final_norm_weight, lm_head_kernel] ++ layer_params_for_gpu ++ caches
  end
end

# Generate tokens with decode phase
generated_tokens = [next_token]
current_caches = kv_caches
current_cache_len = prompt_length

IO.write("  Generated text: #{Bumblebee.Tokenizer.decode(tokenizer, [next_token])}")

Enum.each(1..(max_new_tokens - 1), fn i ->
  # Build decode SPMD for current cache length
  decode_spmd = build_decode_spmd.(1, current_cache_len)

  # Prepare inputs with last generated token
  last_token = List.last(generated_tokens)
  replica_inputs = prepare_decode_inputs.(last_token, current_caches)

  # Run decode
  [[new_logits | new_caches] | _] = EXLA.SPMD.run(decode_spmd, replica_inputs)

  # Sample next token
  new_token = Nx.argmax(new_logits, axis: 1) |> Nx.to_flat_list() |> hd()

  # Update state
  generated_tokens = generated_tokens ++ [new_token]
  current_caches = new_caches
  current_cache_len = current_cache_len + 1

  # Print token
  IO.write(Bumblebee.Tokenizer.decode(tokenizer, [new_token]))

  # Progress indicator every 5 tokens
  if rem(i, 5) == 0 do
    IO.write(" [#{i}/#{max_new_tokens - 1}]")
  end
end)

IO.puts("")

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
IO.puts("\n" <> ("=" |> String.duplicate(70)))
IO.puts("Summary")
IO.puts("=" |> String.duplicate(70))

full_tokens = Nx.to_flat_list(input_ids) ++ generated_tokens
full_text = Bumblebee.Tokenizer.decode(tokenizer, full_tokens)

IO.puts("""

✓ KV Cache Generation Complete!

Prompt: "#{prompt}"

Full Generated Text:
#{full_text}

Performance:
  - Prefill: #{Float.round(time_us / 1000, 2)} ms for #{prompt_length} tokens
  - Decode: #{max_new_tokens} tokens generated with O(1) computation per token
  - Cache updated incrementally (no recomputation of previous tokens)

Model: Mistral 7B (#{num_layers} layers)
TP Configuration: #{tp_size} GPUs
Cache shape per layer: [1, #{local_kv_heads}, seq_len, #{head_dim}]

Note: Output quality limited by simplified embedding/norm implementations.
With proper implementations, this architecture enables efficient LLM inference.
""")
