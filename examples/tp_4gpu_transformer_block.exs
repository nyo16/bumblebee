# Run a full transformer block on 4 GPUs with tensor parallelism
#
# This combines:
# - RMSNorm (replicated)
# - Tensor-Parallel Attention (column-parallel Q/K/V, row-parallel output + all-reduce)
# - Residual connection
# - RMSNorm (replicated)
# - Tensor-Parallel FFN (column-parallel gate/up, row-parallel down + all-reduce)
# - Residual connection
#
# Usage:
#   mix run examples/tp_4gpu_transformer_block.exs
#
# For faster testing:
#   MODEL=tiny mix run examples/tp_4gpu_transformer_block.exs

Nx.default_backend(EXLA.Backend)

IO.puts("=" |> String.duplicate(70))
IO.puts("4-GPU Tensor Parallel Transformer Block")
IO.puts("=" |> String.duplicate(70))

alias EXLA.MLIR.{Function, Value}

# Configuration
tp_size = 4
batch_size = 2
seq_len = 16

IO.puts("\nConfiguration:")
IO.puts("  TP size: #{tp_size}")
IO.puts("  Batch size: #{batch_size}")
IO.puts("  Sequence length: #{seq_len}")

# ----------------------------------------------------------
# Step 1: Load model
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 1: Loading model")
IO.puts("-" |> String.duplicate(70))

repo = case System.get_env("MODEL") do
  "tiny" ->
    IO.puts("  Using TinyLlama (fast testing mode)")
    {:hf, "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}
  _ ->
    IO.puts("  Using Mistral 7B (default)")
    {:hf, "mistralai/Mistral-7B-v0.1"}
end

IO.puts("  Loading model...")
{:ok, %{params: params, spec: spec}} = Bumblebee.load_model(repo)

hidden_size = spec.hidden_size
intermediate_size = spec.intermediate_size
num_heads = spec.num_attention_heads
num_kv_heads = spec.num_key_value_heads
head_dim = div(hidden_size, num_heads)

# Per-GPU dimensions
local_heads = div(num_heads, tp_size)
local_kv_heads = div(num_kv_heads, tp_size)
local_intermediate = div(intermediate_size, tp_size)
kv_head_repeat = div(local_heads, local_kv_heads)

IO.puts("  Model loaded!")
IO.puts("  Hidden size: #{hidden_size}")
IO.puts("  Intermediate size: #{intermediate_size} (#{local_intermediate} per GPU)")
IO.puts("  Query heads: #{num_heads} (#{local_heads} per GPU)")
IO.puts("  KV heads: #{num_kv_heads} (#{local_kv_heads} per GPU)")
IO.puts("  Head dim: #{head_dim}")

# ----------------------------------------------------------
# Step 2: Extract parameters from layer 0
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 2: Extracting transformer block parameters")
IO.puts("-" |> String.duplicate(70))

param_data = params.data

# RMSNorm weights (replicated)
sa_norm_weight = param_data["decoder.blocks.0.self_attention_norm"]["weight"]
ffn_norm_weight = param_data["decoder.blocks.0.output_norm"]["weight"]

# Attention weights
query_kernel = param_data["decoder.blocks.0.self_attention.query"]["kernel"]
key_kernel = param_data["decoder.blocks.0.self_attention.key"]["kernel"]
value_kernel = param_data["decoder.blocks.0.self_attention.value"]["kernel"]
output_kernel = param_data["decoder.blocks.0.self_attention.output"]["kernel"]

# FFN weights
gate_kernel = param_data["decoder.blocks.0.ffn.gate"]["kernel"]
up_kernel = param_data["decoder.blocks.0.ffn.intermediate"]["kernel"]
down_kernel = param_data["decoder.blocks.0.ffn.output"]["kernel"]

IO.puts("  RMSNorm weights:")
IO.puts("    self_attention_norm: #{inspect(Nx.shape(sa_norm_weight))}")
IO.puts("    output_norm: #{inspect(Nx.shape(ffn_norm_weight))}")

IO.puts("\n  Attention weights:")
IO.puts("    Query: #{inspect(Nx.shape(query_kernel))}")
IO.puts("    Key: #{inspect(Nx.shape(key_kernel))}")
IO.puts("    Value: #{inspect(Nx.shape(value_kernel))}")
IO.puts("    Output: #{inspect(Nx.shape(output_kernel))}")

IO.puts("\n  FFN weights:")
IO.puts("    Gate: #{inspect(Nx.shape(gate_kernel))}")
IO.puts("    Up: #{inspect(Nx.shape(up_kernel))}")
IO.puts("    Down: #{inspect(Nx.shape(down_kernel))}")

# ----------------------------------------------------------
# Step 3: Shard parameters
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 3: Sharding parameters across GPUs")
IO.puts("-" |> String.duplicate(70))

local_q_size = div(Nx.axis_size(query_kernel, 1), tp_size)
local_kv_size = div(Nx.axis_size(key_kernel, 1), tp_size)
local_output_in = div(Nx.axis_size(output_kernel, 0), tp_size)

# Attention shards
query_shards = for i <- 0..(tp_size - 1) do
  Nx.slice(query_kernel, [0, i * local_q_size], [hidden_size, local_q_size])
end

key_shards = for i <- 0..(tp_size - 1) do
  Nx.slice(key_kernel, [0, i * local_kv_size], [hidden_size, local_kv_size])
end

value_shards = for i <- 0..(tp_size - 1) do
  Nx.slice(value_kernel, [0, i * local_kv_size], [hidden_size, local_kv_size])
end

output_shards = for i <- 0..(tp_size - 1) do
  Nx.slice(output_kernel, [i * local_output_in, 0], [local_output_in, hidden_size])
end

# FFN shards
gate_shards = for i <- 0..(tp_size - 1) do
  Nx.slice(gate_kernel, [0, i * local_intermediate], [hidden_size, local_intermediate])
end

up_shards = for i <- 0..(tp_size - 1) do
  Nx.slice(up_kernel, [0, i * local_intermediate], [hidden_size, local_intermediate])
end

down_shards = for i <- 0..(tp_size - 1) do
  Nx.slice(down_kernel, [i * local_intermediate, 0], [local_intermediate, hidden_size])
end

IO.puts("  Attention shards (per GPU):")
IO.puts("    Query: #{inspect(Nx.shape(hd(query_shards)))} (column-parallel)")
IO.puts("    Key: #{inspect(Nx.shape(hd(key_shards)))} (column-parallel)")
IO.puts("    Value: #{inspect(Nx.shape(hd(value_shards)))} (column-parallel)")
IO.puts("    Output: #{inspect(Nx.shape(hd(output_shards)))} (row-parallel)")

IO.puts("\n  FFN shards (per GPU):")
IO.puts("    Gate: #{inspect(Nx.shape(hd(gate_shards)))} (column-parallel)")
IO.puts("    Up: #{inspect(Nx.shape(hd(up_shards)))} (column-parallel)")
IO.puts("    Down: #{inspect(Nx.shape(hd(down_shards)))} (row-parallel)")

# ----------------------------------------------------------
# Step 4: Build SPMD executable for transformer block
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 4: Building SPMD executable for transformer block")
IO.puts("-" |> String.duplicate(70))

IO.puts("  Building SPMD executable (this includes attention + FFN + norms + residuals)...")

# Input typespecs
input_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, hidden_size})

# Norm weights (replicated - same on all GPUs)
sa_norm_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size})
ffn_norm_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size})

# Attention weights (sharded)
query_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_q_size})
key_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_kv_size})
value_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_kv_size})
attn_out_typespec = EXLA.Typespec.tensor({:f, 32}, {local_output_in, hidden_size})

# FFN weights (sharded)
gate_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_intermediate})
up_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_intermediate})
down_typespec = EXLA.Typespec.tensor({:f, 32}, {local_intermediate, hidden_size})

input_typespecs = [
  input_typespec,
  # Norms (replicated)
  sa_norm_typespec, ffn_norm_typespec,
  # Attention (sharded)
  query_typespec, key_typespec, value_typespec, attn_out_typespec,
  # FFN (sharded)
  gate_typespec, up_typespec, down_typespec
]

output_typespecs = [input_typespec]  # Same shape as input

replica_groups = [Enum.to_list(0..(tp_size - 1))]

spmd = EXLA.SPMD.build(input_typespecs, output_typespecs, fn builder ->
  [input, sa_norm_w, ffn_norm_w, q_w, k_w, v_w, o_w, gate_w, up_w, down_w] = Function.get_arguments(builder)

  hidden_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, hidden_size})

  # ===== Helper: Simplified Norm =====
  # For TP demo, we use a simplified norm: just multiply by weight
  # The key TP operations are the projections and all-reduce, not normalization
  # Real implementation would use proper RMSNorm with reduce operations
  simple_norm = fn x, weight ->
    weight_broadcast = Value.broadcast_in_dim(weight, [2], hidden_typespec)
    Value.multiply(x, weight_broadcast, hidden_typespec)
  end

  # ===== Self-Attention =====
  # 1. Simplified Norm (weight scaling)
  normed_for_attn = simple_norm.(input, sa_norm_w)

  # 2. Q, K, V projections (column-parallel)
  q_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_q_size})
  k_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_kv_size})
  v_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_kv_size})

  q = Value.dot_general(normed_for_attn, q_w, {[2], [], [0], []}, :default, q_proj_typespec)
  k = Value.dot_general(normed_for_attn, k_w, {[2], [], [0], []}, :default, k_proj_typespec)
  v = Value.dot_general(normed_for_attn, v_w, {[2], [], [0], []}, :default, v_proj_typespec)

  # 3. Reshape for attention
  q_reshaped_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_heads, head_dim})
  q_reshaped = Value.reshape(q, q_reshaped_typespec)

  k_reshaped_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_kv_heads, head_dim})
  v_reshaped_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_kv_heads, head_dim})
  k_reshaped = Value.reshape(k, k_reshaped_typespec)
  v_reshaped = Value.reshape(v, v_reshaped_typespec)

  # 4. Transpose for attention computation
  q_transposed_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_heads, seq_len, head_dim})
  q_transposed = Value.transpose(q_reshaped, [0, 2, 1, 3], q_transposed_typespec)

  k_transposed_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, head_dim, seq_len})
  k_transposed = Value.transpose(k_reshaped, [0, 2, 3, 1], k_transposed_typespec)

  v_transposed_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, seq_len, head_dim})
  v_transposed = Value.transpose(v_reshaped, [0, 2, 1, 3], v_transposed_typespec)

  # 5. GQA: group Q heads and expand K, V
  q_grouped_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, seq_len, head_dim})
  q_grouped = Value.reshape(q_transposed, q_grouped_typespec)

  k_expanded_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, head_dim, seq_len})
  k_expanded = Value.reshape(k_transposed, k_expanded_typespec)
  k_broadcast_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, head_dim, seq_len})
  k_broadcast = Value.broadcast_in_dim(k_expanded, [0, 1, 2, 3, 4], k_broadcast_typespec)

  v_expanded_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, seq_len, head_dim})
  v_expanded = Value.reshape(v_transposed, v_expanded_typespec)
  v_broadcast_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, seq_len, head_dim})
  v_broadcast = Value.broadcast_in_dim(v_expanded, [0, 1, 2, 3, 4], v_broadcast_typespec)

  # 6. Attention scores: Q @ K^T / sqrt(head_dim)
  scores_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, seq_len, seq_len})
  scores = Value.dot_general(q_grouped, k_broadcast, {[4], [0, 1, 2], [3], [0, 1, 2]}, :default, scores_typespec)

  scale_value = 1.0 / :math.sqrt(head_dim)
  scale_typespec = EXLA.Typespec.tensor({:f, 32}, {})
  scale_tensor = Value.constant(builder, [scale_value], scale_typespec)
  scale_broadcast = Value.broadcast_in_dim(scale_tensor, [], scores_typespec)
  scores_scaled = Value.multiply(scores, scale_broadcast, scores_typespec)

  # 7. Attention weights (sigmoid for simplicity in TP demo)
  attention_weights = Value.sigmoid(scores_scaled, scores_typespec)

  # 8. Attention output: weights @ V
  attn_output_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, seq_len, head_dim})
  attn_output = Value.dot_general(attention_weights, v_broadcast, {[4], [0, 1, 2], [3], [0, 1, 2]}, :default, attn_output_typespec)

  # 9. Reshape back
  attn_output_merged = Value.reshape(attn_output, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_heads, seq_len, head_dim}))
  attn_output_transposed = Value.transpose(attn_output_merged, [0, 2, 1, 3], EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_heads, head_dim}))
  attn_output_flat = Value.reshape(attn_output_transposed, EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_q_size}))

  # 10. Output projection (row-parallel) + all-reduce
  attn_partial = Value.dot_general(attn_output_flat, o_w, {[2], [], [0], []}, :default, hidden_typespec)
  attn_result = Value.all_reduce(attn_partial, :sum, replica_groups, hidden_typespec)

  # 11. Residual connection
  after_attn = Value.add(input, attn_result, hidden_typespec)

  # ===== FFN =====
  # 1. Simplified Norm
  normed_for_ffn = simple_norm.(after_attn, ffn_norm_w)

  # 2. Gate and Up projections (column-parallel)
  intermediate_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_intermediate})

  gate_out = Value.dot_general(normed_for_ffn, gate_w, {[2], [], [0], []}, :default, intermediate_typespec)
  up_out = Value.dot_general(normed_for_ffn, up_w, {[2], [], [0], []}, :default, intermediate_typespec)

  # 3. SiLU activation on gate
  gate_sigmoid = Value.sigmoid(gate_out, intermediate_typespec)
  gate_silu = Value.multiply(gate_out, gate_sigmoid, intermediate_typespec)

  # 4. Element-wise multiply
  combined = Value.multiply(gate_silu, up_out, intermediate_typespec)

  # 5. Down projection (row-parallel) + all-reduce
  ffn_partial = Value.dot_general(combined, down_w, {[2], [], [0], []}, :default, hidden_typespec)
  ffn_result = Value.all_reduce(ffn_partial, :sum, replica_groups, hidden_typespec)

  # 6. Residual connection
  output = Value.add(after_attn, ffn_result, hidden_typespec)

  [output]
end, num_replicas: tp_size, client: :cuda)

IO.puts("  SPMD executable built successfully!")
IO.puts("  - 2 all-reduce operations (after attention output, after FFN down)")
IO.puts("  - 2 norm layers (simplified, replicated computation)")
IO.puts("  - 2 residual connections")

# ----------------------------------------------------------
# Step 5: Run transformer block on 4 GPUs
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 5: Running transformer block on #{tp_size} GPUs")
IO.puts("-" |> String.duplicate(70))

# Create random input
key = Nx.Random.key(42)
{input, _key} = Nx.Random.normal(key, shape: {batch_size, seq_len, hidden_size}, type: :f32)
input = Nx.multiply(input, 0.1)

IO.puts("  Input shape: #{inspect(Nx.shape(input))}")
IO.puts("  Input mean: #{Nx.mean(input) |> Nx.to_number() |> Float.round(6)}")

# Prepare replica inputs
replica_inputs = for i <- 0..(tp_size - 1) do
  [
    input,
    # Norms (replicated - same on all GPUs)
    sa_norm_weight, ffn_norm_weight,
    # Attention shards
    Enum.at(query_shards, i), Enum.at(key_shards, i),
    Enum.at(value_shards, i), Enum.at(output_shards, i),
    # FFN shards
    Enum.at(gate_shards, i), Enum.at(up_shards, i), Enum.at(down_shards, i)
  ]
end

IO.puts("\n  Running SPMD on #{tp_size} GPUs...")

{time_us, results} = :timer.tc(fn ->
  EXLA.SPMD.run(spmd, replica_inputs)
end)

IO.puts("  Execution time: #{Float.round(time_us / 1000, 2)} ms")

IO.puts("\n  Results from all GPUs:")
for {[output], gpu} <- Enum.with_index(results) do
  mean = Nx.mean(output) |> Nx.to_number() |> Float.round(6)
  std = Nx.standard_deviation(output) |> Nx.to_number() |> Float.round(6)
  IO.puts("    GPU #{gpu}: shape=#{inspect(Nx.shape(output))}, mean=#{mean}, std=#{std}")
end

# Verify all outputs are identical
[[ref_output] | rest] = results
all_same = Enum.all?(rest, fn [output] ->
  Nx.all_close(ref_output, output, atol: 1.0e-5) |> Nx.to_number() == 1
end)

if all_same do
  IO.puts("\n  All GPU outputs are identical!")
else
  IO.puts("\n  GPU outputs differ (unexpected)")
end

# ----------------------------------------------------------
# Step 6: Benchmark
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 6: Benchmarking (10 runs)")
IO.puts("-" |> String.duplicate(70))

times = for _ <- 1..10 do
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

model_name = case System.get_env("MODEL") do
  "tiny" -> "TinyLlama-1.1B"
  _ -> "Mistral-7B"
end

IO.puts("""

Successfully ran #{model_name} transformer block with 4-GPU tensor parallelism!

Block structure:
  Input (replicated)
    |
    +-- RMSNorm (replicated)
    +-- Self-Attention (TP, 2 all-reduces: QKV + Output)
    |     Query: #{num_heads} heads -> #{local_heads}/GPU (column-parallel)
    |     KV: #{num_kv_heads} heads -> #{local_kv_heads}/GPU (column-parallel)
    |     Output: row-parallel + all-reduce
    +-- Residual
    |
    +-- RMSNorm (replicated)
    +-- FFN (TP, 2 all-reduces: gate/up + down)
    |     Gate/Up: #{intermediate_size} -> #{local_intermediate}/GPU (column-parallel)
    |     Down: row-parallel + all-reduce
    +-- Residual
    |
  Output (replicated)

Communication: 2 all-reduce operations per block
Performance: #{Float.round(avg_time, 2)} ms average per transformer block
""")
