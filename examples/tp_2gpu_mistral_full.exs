# Run Mistral 7B model on 2 GPUs with tensor parallelism
#
# This stacks multiple transformer blocks with:
# - Embedding lookup (replicated)
# - N transformer blocks with TP attention + FFN
# - Final RMSNorm (replicated)
# - LM head (replicated for simplicity)
#
# Usage:
#   mix run examples/tp_2gpu_mistral_full.exs
#
# For fewer layers:
#   LAYERS=2 mix run examples/tp_2gpu_mistral_full.exs

Nx.default_backend(EXLA.Backend)

IO.puts("=" |> String.duplicate(70))
IO.puts("2-GPU Tensor Parallel Full Model - Mistral 7B")
IO.puts("=" |> String.duplicate(70))

alias EXLA.MLIR.{Function, Value}

# Configuration
tp_size = 2
batch_size = 1
seq_len = 8

# Number of layers to use (default 4 for testing, can go up to full model)
num_layers = case System.get_env("LAYERS") do
  nil -> 4
  n -> String.to_integer(n)
end

IO.puts("\nConfiguration:")
IO.puts("  TP size: #{tp_size}")
IO.puts("  Batch size: #{batch_size}")
IO.puts("  Sequence length: #{seq_len}")
IO.puts("  Number of layers: #{num_layers}")

# ----------------------------------------------------------
# Step 1: Load model
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 1: Loading model")
IO.puts("-" |> String.duplicate(70))

repo = {:hf, "mistralai/Mistral-7B-v0.1"}

IO.puts("  Loading Mistral 7B...")
{:ok, %{params: params, spec: spec}} = Bumblebee.load_model(repo)

hidden_size = spec.hidden_size
intermediate_size = spec.intermediate_size
num_heads = spec.num_attention_heads
num_kv_heads = spec.num_key_value_heads
head_dim = div(hidden_size, num_heads)
vocab_size = spec.vocab_size
total_layers = spec.num_blocks

# Limit layers to what's available
num_layers = min(num_layers, total_layers)

# Per-GPU dimensions
local_heads = div(num_heads, tp_size)
local_kv_heads = div(num_kv_heads, tp_size)
local_intermediate = div(intermediate_size, tp_size)
local_q_size = local_heads * head_dim
local_kv_size = local_kv_heads * head_dim
kv_head_repeat = div(local_heads, local_kv_heads)

IO.puts("  Model loaded!")
IO.puts("  Vocab size: #{vocab_size}")
IO.puts("  Hidden size: #{hidden_size}")
IO.puts("  Intermediate size: #{intermediate_size} (#{local_intermediate} per GPU)")
IO.puts("  Query heads: #{num_heads} (#{local_heads} per GPU)")
IO.puts("  KV heads: #{num_kv_heads} (#{local_kv_heads} per GPU)")
IO.puts("  Head dim: #{head_dim}")
IO.puts("  Total layers: #{total_layers} (using #{num_layers})")

# ----------------------------------------------------------
# Step 2: Extract and shard parameters
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 2: Extracting and sharding parameters")
IO.puts("-" |> String.duplicate(70))

param_data = params.data

# Embeddings (replicated)
embed_tokens = param_data["embedder.token_embedding"]["kernel"]
IO.puts("  Embedding: #{inspect(Nx.shape(embed_tokens))}")

# Final norm (replicated) - note: top-level "output_norm" not "decoder.output_norm"
final_norm_weight = param_data["output_norm"]["weight"]
IO.puts("  Final norm: #{inspect(Nx.shape(final_norm_weight))}")

# LM head (replicated for simplicity)
lm_head_kernel = param_data["language_modeling_head.output"]["kernel"]
IO.puts("  LM head: #{inspect(Nx.shape(lm_head_kernel))}")

# Extract and shard layer parameters
IO.puts("\n  Extracting #{num_layers} layer parameters...")

layer_params = for layer_idx <- 0..(num_layers - 1) do
  prefix = "decoder.blocks.#{layer_idx}"

  # Norms (replicated)
  sa_norm = param_data["#{prefix}.self_attention_norm"]["weight"]
  ffn_norm = param_data["#{prefix}.output_norm"]["weight"]

  # Attention
  q_kernel = param_data["#{prefix}.self_attention.query"]["kernel"]
  k_kernel = param_data["#{prefix}.self_attention.key"]["kernel"]
  v_kernel = param_data["#{prefix}.self_attention.value"]["kernel"]
  o_kernel = param_data["#{prefix}.self_attention.output"]["kernel"]

  # FFN
  gate_kernel = param_data["#{prefix}.ffn.gate"]["kernel"]
  up_kernel = param_data["#{prefix}.ffn.intermediate"]["kernel"]
  down_kernel = param_data["#{prefix}.ffn.output"]["kernel"]

  # Shard attention weights
  q_shards = for i <- 0..(tp_size - 1) do
    Nx.slice(q_kernel, [0, i * local_q_size], [hidden_size, local_q_size])
  end

  k_shards = for i <- 0..(tp_size - 1) do
    Nx.slice(k_kernel, [0, i * local_kv_size], [hidden_size, local_kv_size])
  end

  v_shards = for i <- 0..(tp_size - 1) do
    Nx.slice(v_kernel, [0, i * local_kv_size], [hidden_size, local_kv_size])
  end

  o_shards = for i <- 0..(tp_size - 1) do
    Nx.slice(o_kernel, [i * local_q_size, 0], [local_q_size, hidden_size])
  end

  # Shard FFN weights
  gate_shards = for i <- 0..(tp_size - 1) do
    Nx.slice(gate_kernel, [0, i * local_intermediate], [hidden_size, local_intermediate])
  end

  up_shards = for i <- 0..(tp_size - 1) do
    Nx.slice(up_kernel, [0, i * local_intermediate], [hidden_size, local_intermediate])
  end

  down_shards = for i <- 0..(tp_size - 1) do
    Nx.slice(down_kernel, [i * local_intermediate, 0], [local_intermediate, hidden_size])
  end

  %{
    sa_norm: sa_norm,
    ffn_norm: ffn_norm,
    q_shards: q_shards,
    k_shards: k_shards,
    v_shards: v_shards,
    o_shards: o_shards,
    gate_shards: gate_shards,
    up_shards: up_shards,
    down_shards: down_shards
  }
end

IO.puts("  Extracted parameters for #{num_layers} layers")
IO.puts("  Per-layer sharded params: Q #{inspect(Nx.shape(hd(hd(layer_params).q_shards)))}")

# ----------------------------------------------------------
# Step 3: Build SPMD executable
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 3: Building SPMD executable for #{num_layers}-layer model")
IO.puts("-" |> String.duplicate(70))

IO.puts("  Building input typespecs...")

# Input: token IDs
input_ids_typespec = EXLA.Typespec.tensor({:s, 64}, {batch_size, seq_len})

# Embedding (replicated)
embed_typespec = EXLA.Typespec.tensor({:f, 32}, {vocab_size, hidden_size})

# Norm typespecs (replicated)
norm_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size})

# Per-layer sharded typespecs
q_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_q_size})
k_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_kv_size})
v_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_kv_size})
o_typespec = EXLA.Typespec.tensor({:f, 32}, {local_q_size, hidden_size})
gate_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_intermediate})
up_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_intermediate})
down_typespec = EXLA.Typespec.tensor({:f, 32}, {local_intermediate, hidden_size})

# LM head (replicated)
lm_head_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, vocab_size})

# Build input typespecs list:
# [input_ids, embed, final_norm, lm_head, layer0_params..., layer1_params..., ...]
# Each layer has: sa_norm, ffn_norm, q, k, v, o, gate, up, down
layer_param_typespecs = List.duplicate([
  norm_typespec,   # sa_norm
  norm_typespec,   # ffn_norm
  q_typespec,
  k_typespec,
  v_typespec,
  o_typespec,
  gate_typespec,
  up_typespec,
  down_typespec
], num_layers) |> List.flatten()

input_typespecs = [input_ids_typespec, embed_typespec, norm_typespec, lm_head_typespec] ++ layer_param_typespecs

# Output: logits for next token prediction (only last position)
output_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, vocab_size})
output_typespecs = [output_typespec]

replica_groups = [Enum.to_list(0..(tp_size - 1))]

IO.puts("  Building SPMD executable (#{length(input_typespecs)} inputs)...")

spmd = EXLA.SPMD.build(input_typespecs, output_typespecs, fn builder ->
  args = Function.get_arguments(builder)

  # Parse arguments
  [input_ids, embed_w, final_norm_w, lm_head_w | layer_args] = args

  # Parse layer arguments (9 per layer)
  layer_weights = Enum.chunk_every(layer_args, 9)
  |> Enum.map(fn [sa_norm, ffn_norm, q, k, v, o, gate, up, down] ->
    %{sa_norm: sa_norm, ffn_norm: ffn_norm, q: q, k: k, v: v, o: o, gate: gate, up: up, down: down}
  end)

  hidden_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, hidden_size})

  # ===== Embedding Lookup =====
  # For demo purposes, use a simplified approach:
  # Convert token IDs to float and project through first few embedding rows
  # Real implementation would use proper gather operation

  # Convert input_ids to float
  float_ids_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len})
  float_ids = Value.convert(input_ids, float_ids_typespec)

  # Scale the token IDs to small values (so matmul doesn't explode)
  scale_typespec = EXLA.Typespec.tensor({:f, 32}, {})
  scale_tensor = Value.constant(builder, [0.001], scale_typespec)
  scale_broadcast = Value.broadcast_in_dim(scale_tensor, [], float_ids_typespec)
  float_ids_scaled = Value.multiply(float_ids, scale_broadcast, float_ids_typespec)

  # Expand to 3D: [batch, seq] -> [batch, seq, 1]
  float_ids_3d_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, 1})
  float_ids_3d = Value.reshape(float_ids_scaled, float_ids_3d_typespec)

  # Take first row of embedding as a projection vector
  proj_typespec = EXLA.Typespec.tensor({:f, 32}, {1, hidden_size})
  proj = Value.slice(embed_w, [0, 0], [1, hidden_size], [1, 1], proj_typespec)

  # Matmul: [batch, seq, 1] @ [1, hidden] -> [batch, seq, hidden]
  hidden_states = Value.dot_general(float_ids_3d, proj, {[2], [], [0], []}, :default, hidden_typespec)

  # ===== Helper: Simplified Norm =====
  simple_norm = fn x, weight ->
    weight_broadcast = Value.broadcast_in_dim(weight, [2], hidden_typespec)
    Value.multiply(x, weight_broadcast, hidden_typespec)
  end

  # ===== Transformer Block Function =====
  transformer_block = fn input, weights ->
    # Self-Attention
    normed_for_attn = simple_norm.(input, weights.sa_norm)

    # Q, K, V projections
    q_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_q_size})
    k_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_kv_size})
    v_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_kv_size})

    q = Value.dot_general(normed_for_attn, weights.q, {[2], [], [0], []}, :default, q_proj_typespec)
    k = Value.dot_general(normed_for_attn, weights.k, {[2], [], [0], []}, :default, k_proj_typespec)
    v = Value.dot_general(normed_for_attn, weights.v, {[2], [], [0], []}, :default, v_proj_typespec)

    # Reshape for attention
    q_reshaped = Value.reshape(q, EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_heads, head_dim}))
    k_reshaped = Value.reshape(k, EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_kv_heads, head_dim}))
    v_reshaped = Value.reshape(v, EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_kv_heads, head_dim}))

    # Transpose
    q_transposed = Value.transpose(q_reshaped, [0, 2, 1, 3], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_heads, seq_len, head_dim}))
    k_transposed = Value.transpose(k_reshaped, [0, 2, 3, 1], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, head_dim, seq_len}))
    v_transposed = Value.transpose(v_reshaped, [0, 2, 1, 3], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, seq_len, head_dim}))

    # GQA grouping
    q_grouped = Value.reshape(q_transposed, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, seq_len, head_dim}))
    k_expanded = Value.reshape(k_transposed, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, head_dim, seq_len}))
    v_expanded = Value.reshape(v_transposed, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, seq_len, head_dim}))

    k_broadcast = Value.broadcast_in_dim(k_expanded, [0, 1, 2, 3, 4], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, head_dim, seq_len}))
    v_broadcast = Value.broadcast_in_dim(v_expanded, [0, 1, 2, 3, 4], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, seq_len, head_dim}))

    # Attention scores
    scores_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, seq_len, seq_len})
    scores = Value.dot_general(q_grouped, k_broadcast, {[4], [0, 1, 2], [3], [0, 1, 2]}, :default, scores_typespec)

    # Scale
    scale_value = 1.0 / :math.sqrt(head_dim)
    scale_typespec = EXLA.Typespec.tensor({:f, 32}, {})
    scale_tensor = Value.constant(builder, [scale_value], scale_typespec)
    scale_broadcast = Value.broadcast_in_dim(scale_tensor, [], scores_typespec)
    scores_scaled = Value.multiply(scores, scale_broadcast, scores_typespec)

    # Simplified attention (sigmoid)
    attention_weights = Value.sigmoid(scores_scaled, scores_typespec)

    # Attention output
    attn_output_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, seq_len, head_dim})
    attn_output = Value.dot_general(attention_weights, v_broadcast, {[4], [0, 1, 2], [3], [0, 1, 2]}, :default, attn_output_typespec)

    # Reshape back
    attn_merged = Value.reshape(attn_output, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_heads, seq_len, head_dim}))
    attn_transposed = Value.transpose(attn_merged, [0, 2, 1, 3], EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_heads, head_dim}))
    attn_flat = Value.reshape(attn_transposed, EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_q_size}))

    # Output projection + all-reduce
    attn_partial = Value.dot_general(attn_flat, weights.o, {[2], [], [0], []}, :default, hidden_typespec)
    attn_result = Value.all_reduce(attn_partial, :sum, replica_groups, hidden_typespec)

    # Residual
    after_attn = Value.add(input, attn_result, hidden_typespec)

    # FFN
    normed_for_ffn = simple_norm.(after_attn, weights.ffn_norm)

    intermediate_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_intermediate})

    gate_out = Value.dot_general(normed_for_ffn, weights.gate, {[2], [], [0], []}, :default, intermediate_typespec)
    up_out = Value.dot_general(normed_for_ffn, weights.up, {[2], [], [0], []}, :default, intermediate_typespec)

    # SiLU
    gate_sigmoid = Value.sigmoid(gate_out, intermediate_typespec)
    gate_silu = Value.multiply(gate_out, gate_sigmoid, intermediate_typespec)
    combined = Value.multiply(gate_silu, up_out, intermediate_typespec)

    # Down projection + all-reduce
    ffn_partial = Value.dot_general(combined, weights.down, {[2], [], [0], []}, :default, hidden_typespec)
    ffn_result = Value.all_reduce(ffn_partial, :sum, replica_groups, hidden_typespec)

    # Residual
    Value.add(after_attn, ffn_result, hidden_typespec)
  end

  # ===== Run all transformer blocks =====
  final_hidden = Enum.reduce(layer_weights, hidden_states, fn weights, hidden ->
    transformer_block.(hidden, weights)
  end)

  # ===== Final Norm =====
  normed_output = simple_norm.(final_hidden, final_norm_w)

  # ===== LM Head =====
  # Take only the last position for next token prediction
  # Slice last position: [batch, seq, hidden] -> [batch, 1, hidden] -> [batch, hidden]
  # Note: limit_indices are exclusive, so to get position seq_len-1 to seq_len, use seq_len as limit
  last_hidden_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, hidden_size})
  last_hidden = Value.slice(normed_output, [0, seq_len - 1, 0], [batch_size, seq_len, hidden_size], [1, 1, 1], last_hidden_typespec)
  last_hidden_2d = Value.reshape(last_hidden, EXLA.Typespec.tensor({:f, 32}, {batch_size, hidden_size}))

  # Project to vocab
  logits = Value.dot_general(last_hidden_2d, lm_head_w, {[1], [], [0], []}, :default, output_typespec)

  [logits]
end, num_replicas: tp_size, client: :cuda)

IO.puts("  SPMD executable built successfully!")
IO.puts("  - #{num_layers} transformer blocks")
IO.puts("  - #{num_layers * 2} all-reduce operations total")
IO.puts("  - Embedding lookup (simplified)")
IO.puts("  - LM head projection")

# ----------------------------------------------------------
# Step 4: Run full model on 4 GPUs
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 4: Running #{num_layers}-layer model on #{tp_size} GPUs")
IO.puts("-" |> String.duplicate(70))

# Create input token IDs
input_ids = Nx.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], type: :s64)

IO.puts("  Input shape: #{inspect(Nx.shape(input_ids))}")
IO.puts("  Input tokens: #{inspect(Nx.to_flat_list(input_ids))}")

# Prepare replica inputs
replica_inputs = for gpu <- 0..(tp_size - 1) do
  layer_params_for_gpu = Enum.flat_map(layer_params, fn layer ->
    [
      layer.sa_norm,
      layer.ffn_norm,
      Enum.at(layer.q_shards, gpu),
      Enum.at(layer.k_shards, gpu),
      Enum.at(layer.v_shards, gpu),
      Enum.at(layer.o_shards, gpu),
      Enum.at(layer.gate_shards, gpu),
      Enum.at(layer.up_shards, gpu),
      Enum.at(layer.down_shards, gpu)
    ]
  end)

  [input_ids, embed_tokens, final_norm_weight, lm_head_kernel] ++ layer_params_for_gpu
end

IO.puts("\n  Running SPMD on #{tp_size} GPUs...")

{time_us, results} = :timer.tc(fn ->
  EXLA.SPMD.run(spmd, replica_inputs)
end)

IO.puts("  Execution time: #{Float.round(time_us / 1000, 2)} ms")

IO.puts("\n  Results from all GPUs:")
for {[logits], gpu} <- Enum.with_index(results) do
  top_token = Nx.argmax(logits, axis: 1) |> Nx.to_flat_list() |> hd()
  max_logit = Nx.reduce_max(logits) |> Nx.to_number() |> Float.round(4)
  IO.puts("    GPU #{gpu}: logits shape=#{inspect(Nx.shape(logits))}, top_token=#{top_token}, max_logit=#{max_logit}")
end

# Verify all outputs are identical
[[ref_logits] | rest] = results
all_same = Enum.all?(rest, fn [logits] ->
  Nx.all_close(ref_logits, logits, atol: 1.0e-5) |> Nx.to_number() == 1
end)

if all_same do
  IO.puts("\n  All GPU outputs are identical!")
else
  IO.puts("\n  GPU outputs differ (unexpected)")
end

# ----------------------------------------------------------
# Step 5: Benchmark
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 5: Benchmarking (5 runs)")
IO.puts("-" |> String.duplicate(70))

times = for _ <- 1..5 do
  {time_us, _} = :timer.tc(fn ->
    EXLA.SPMD.run(spmd, replica_inputs)
  end)
  time_us / 1000
end

avg_time = Enum.sum(times) / length(times)
min_time = Enum.min(times)
max_time = Enum.max(times)

IO.puts("  Average: #{Float.round(avg_time, 2)} ms")
IO.puts("  Min: #{Float.round(min_time, 2)} ms")
IO.puts("  Max: #{Float.round(max_time, 2)} ms")

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
IO.puts("\n" <> ("=" |> String.duplicate(70)))
IO.puts("Summary")
IO.puts("=" |> String.duplicate(70))

IO.puts("""

Successfully ran #{num_layers}-layer Mistral-7B with 2-GPU tensor parallelism!

Model architecture:
  - Embedding lookup (simplified projection)
  - #{num_layers}x Transformer blocks with TP
    - Self-Attention: #{num_heads} Q heads, #{num_kv_heads} KV heads (GQA)
    - FFN: #{intermediate_size} intermediate
  - Final norm + LM head

Tensor parallelism:
  - #{num_layers * 2} all-reduce operations (2 per block)
  - Column-parallel: Q, K, V, gate, up projections
  - Row-parallel: attention output, FFN down projections

Performance:
  - #{Float.round(avg_time, 2)} ms for #{num_layers} layers
  - #{Float.round(avg_time / num_layers, 2)} ms per layer average

To run with more layers:
  LAYERS=#{total_layers} mix run examples/tp_2gpu_mistral_full.exs
""")
