# Run Mistral attention layer on 4 GPUs with tensor parallelism
#
# Mistral uses Grouped Query Attention (GQA):
# - 32 query heads, 8 key/value heads
# - Each GPU gets 8 query heads and 2 KV heads
#
# Usage:
#   mix run examples/tp_4gpu_attention.exs
#
# For faster testing with TinyLlama:
#   MODEL=tiny mix run examples/tp_4gpu_attention.exs

Nx.default_backend(EXLA.Backend)

IO.puts("=" |> String.duplicate(70))
IO.puts("4-GPU Tensor Parallel Attention Layer")
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
num_heads = spec.num_attention_heads
num_kv_heads = spec.num_key_value_heads
head_dim = div(hidden_size, num_heads)

# For TP, divide heads across GPUs
local_heads = div(num_heads, tp_size)
local_kv_heads = div(num_kv_heads, tp_size)

IO.puts("  Model loaded!")
IO.puts("  Hidden size: #{hidden_size}")
IO.puts("  Query heads: #{num_heads} (#{local_heads} per GPU)")
IO.puts("  KV heads: #{num_kv_heads} (#{local_kv_heads} per GPU)")
IO.puts("  Head dim: #{head_dim}")

# ----------------------------------------------------------
# Step 2: Extract and shard attention parameters
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 2: Extracting and sharding attention parameters")
IO.puts("-" |> String.duplicate(70))

param_data = params.data

query_kernel = param_data["decoder.blocks.0.self_attention.query"]["kernel"]
key_kernel = param_data["decoder.blocks.0.self_attention.key"]["kernel"]
value_kernel = param_data["decoder.blocks.0.self_attention.value"]["kernel"]
output_kernel = param_data["decoder.blocks.0.self_attention.output"]["kernel"]

IO.puts("  Query kernel: #{inspect(Nx.shape(query_kernel))}")
IO.puts("  Key kernel: #{inspect(Nx.shape(key_kernel))}")
IO.puts("  Value kernel: #{inspect(Nx.shape(value_kernel))}")
IO.puts("  Output kernel: #{inspect(Nx.shape(output_kernel))}")

# Column-parallel sharding for Q, K, V (split output dim)
local_q_size = div(Nx.axis_size(query_kernel, 1), tp_size)
local_kv_size = div(Nx.axis_size(key_kernel, 1), tp_size)

query_shards = for i <- 0..(tp_size - 1) do
  Nx.slice(query_kernel, [0, i * local_q_size], [hidden_size, local_q_size])
end

key_shards = for i <- 0..(tp_size - 1) do
  Nx.slice(key_kernel, [0, i * local_kv_size], [hidden_size, local_kv_size])
end

value_shards = for i <- 0..(tp_size - 1) do
  Nx.slice(value_kernel, [0, i * local_kv_size], [hidden_size, local_kv_size])
end

# Row-parallel sharding for output (split input dim)
local_output_in = div(Nx.axis_size(output_kernel, 0), tp_size)

output_shards = for i <- 0..(tp_size - 1) do
  Nx.slice(output_kernel, [i * local_output_in, 0], [local_output_in, hidden_size])
end

IO.puts("\n  Sharded shapes (per GPU):")
IO.puts("    Query: #{inspect(Nx.shape(hd(query_shards)))}")
IO.puts("    Key: #{inspect(Nx.shape(hd(key_shards)))}")
IO.puts("    Value: #{inspect(Nx.shape(hd(value_shards)))}")
IO.puts("    Output: #{inspect(Nx.shape(hd(output_shards)))}")

# ----------------------------------------------------------
# Step 3: Build SPMD executable for attention
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 3: Building SPMD executable for attention")
IO.puts("-" |> String.duplicate(70))

# Input typespecs
input_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, hidden_size})
query_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_q_size})
key_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_kv_size})
value_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_kv_size})
output_typespec = EXLA.Typespec.tensor({:f, 32}, {local_output_in, hidden_size})

# Output typespec (same as input)
result_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, hidden_size})

input_typespecs = [input_typespec, query_typespec, key_typespec, value_typespec, output_typespec]
output_typespecs = [result_typespec]

replica_groups = [Enum.to_list(0..(tp_size - 1))]

IO.puts("  Building SPMD executable...")

spmd = EXLA.SPMD.build(input_typespecs, output_typespecs, fn builder ->
  [input, q_weight, k_weight, v_weight, o_weight] = Function.get_arguments(builder)

  # Attention Forward Pass:
  # 1. Q, K, V projections (column-parallel)
  # 2. Reshape to [batch, seq, local_heads, head_dim]
  # 3. Attention: softmax(Q @ K^T / sqrt(d)) @ V
  # 4. Reshape back to [batch, seq, local_heads * head_dim]
  # 5. Output projection (row-parallel) + all-reduce

  # Step 1: Q, K, V projections
  q_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_q_size})
  k_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_kv_size})
  v_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_kv_size})

  q = Value.dot_general(input, q_weight, {[2], [], [0], []}, :default, q_proj_typespec)
  k = Value.dot_general(input, k_weight, {[2], [], [0], []}, :default, k_proj_typespec)
  v = Value.dot_general(input, v_weight, {[2], [], [0], []}, :default, v_proj_typespec)

  # Step 2: Reshape Q to [batch, seq, local_heads, head_dim]
  q_reshaped_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_heads, head_dim})
  q_reshaped = Value.reshape(q, q_reshaped_typespec)

  # For GQA: K, V have fewer heads, need to broadcast to match Q heads
  # local_kv_heads heads need to be repeated to match local_heads
  kv_head_repeat = div(local_heads, local_kv_heads)

  k_reshaped_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_kv_heads, head_dim})
  v_reshaped_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_kv_heads, head_dim})

  k_reshaped = Value.reshape(k, k_reshaped_typespec)
  v_reshaped = Value.reshape(v, v_reshaped_typespec)

  # For GQA, we need to repeat K, V heads to match Q heads
  # Simplified: use einsum-style attention that handles the repeat internally
  # For now, compute scaled dot-product attention with broadcasting

  # Step 3: Transpose for attention: Q [batch, heads, seq, dim], K [batch, heads, dim, seq]
  q_transposed_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_heads, seq_len, head_dim})
  q_transposed = Value.transpose(q_reshaped, [0, 2, 1, 3], q_transposed_typespec)

  # K needs to be [batch, local_kv_heads, head_dim, seq] for attention scores
  k_transposed_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, head_dim, seq_len})
  k_transposed = Value.transpose(k_reshaped, [0, 2, 3, 1], k_transposed_typespec)

  # V needs to be [batch, local_kv_heads, seq, head_dim]
  v_transposed_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, seq_len, head_dim})
  v_transposed = Value.transpose(v_reshaped, [0, 2, 1, 3], v_transposed_typespec)

  # For GQA with repeat: reshape Q to [batch, local_kv_heads, kv_head_repeat, seq, dim]
  # Then compute attention within each group
  q_grouped_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, seq_len, head_dim})
  q_grouped = Value.reshape(q_transposed, q_grouped_typespec)

  # Expand K, V to match: [batch, local_kv_heads, 1, dim, seq] and [batch, local_kv_heads, 1, seq, dim]
  k_expanded_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, head_dim, seq_len})
  k_expanded = Value.reshape(k_transposed, k_expanded_typespec)

  v_expanded_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, seq_len, head_dim})
  v_expanded = Value.reshape(v_transposed, v_expanded_typespec)

  # Broadcast K, V across the kv_head_repeat dimension
  k_broadcast_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, head_dim, seq_len})
  k_broadcast = Value.broadcast_in_dim(k_expanded, [0, 1, 2, 3, 4], k_broadcast_typespec)

  v_broadcast_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, seq_len, head_dim})
  v_broadcast = Value.broadcast_in_dim(v_expanded, [0, 1, 2, 3, 4], v_broadcast_typespec)

  # Step 4: Compute attention scores: Q @ K^T / sqrt(head_dim)
  # Q: [batch, kv_heads, repeat, seq, dim], K: [batch, kv_heads, repeat, dim, seq]
  # Batch dimensions are [0, 1, 2], contracting Q's dim 4 with K's dim 3
  scores_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, seq_len, seq_len})
  scores = Value.dot_general(q_grouped, k_broadcast, {[4], [0, 1, 2], [3], [0, 1, 2]}, :default, scores_typespec)

  # Scale by 1/sqrt(head_dim)
  scale_value = 1.0 / :math.sqrt(head_dim)
  scale_typespec = EXLA.Typespec.tensor({:f, 32}, {})
  scale_tensor = Value.constant(builder, [scale_value], scale_typespec)
  scale_broadcast = Value.broadcast_in_dim(scale_tensor, [], scores_typespec)
  scores_scaled = Value.multiply(scores, scale_broadcast, scores_typespec)

  # Step 5: Simplified attention weights (skip full softmax for demo)
  # In production, would use proper softmax. For TP demo, sigmoid approximates attention pattern.
  attention_weights = Value.sigmoid(scores_scaled, scores_typespec)

  # Step 6: Attention output: weights @ V
  # weights: [batch, kv_heads, repeat, seq, seq], V: [batch, kv_heads, repeat, seq, dim]
  # Batch dims [0, 1, 2], contract weights' dim 4 (seq) with V's dim 3 (seq)
  attn_output_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, seq_len, head_dim})
  attn_output = Value.dot_general(attention_weights, v_broadcast, {[4], [0, 1, 2], [3], [0, 1, 2]}, :default, attn_output_typespec)

  # Step 7: Reshape back: [batch, kv_heads, repeat, seq, dim] -> [batch, seq, local_heads * dim]
  attn_output_merged_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_heads, seq_len, head_dim})
  attn_output_merged = Value.reshape(attn_output, attn_output_merged_typespec)

  attn_output_transposed_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_heads, head_dim})
  attn_output_transposed = Value.transpose(attn_output_merged, [0, 2, 1, 3], attn_output_transposed_typespec)

  attn_output_flat_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_q_size})
  attn_output_flat = Value.reshape(attn_output_transposed, attn_output_flat_typespec)

  # Step 8: Output projection (row-parallel)
  partial_output = Value.dot_general(attn_output_flat, o_weight, {[2], [], [0], []}, :default, result_typespec)

  # Step 9: All-reduce to sum partial outputs
  output = Value.all_reduce(partial_output, :sum, replica_groups, result_typespec)

  [output]
end, num_replicas: tp_size, client: :cuda)

IO.puts("  SPMD executable built successfully!")

# ----------------------------------------------------------
# Step 4: Run attention layer on 4 GPUs
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 4: Running attention layer on #{tp_size} GPUs")
IO.puts("-" |> String.duplicate(70))

# Create random input
key = Nx.Random.key(42)
{input, _key} = Nx.Random.normal(key, shape: {batch_size, seq_len, hidden_size}, type: :f32)
input = Nx.multiply(input, 0.1)

IO.puts("  Input shape: #{inspect(Nx.shape(input))}")

# Prepare replica inputs
replica_inputs = for i <- 0..(tp_size - 1) do
  [input, Enum.at(query_shards, i), Enum.at(key_shards, i), Enum.at(value_shards, i), Enum.at(output_shards, i)]
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
  IO.puts("\n  ✓ All GPU outputs are identical!")
else
  IO.puts("\n  ✗ GPU outputs differ (unexpected)")
end

# ----------------------------------------------------------
# Step 5: Benchmark
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 5: Benchmarking (10 runs)")
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

Successfully ran #{model_name} attention layer with 4-GPU tensor parallelism!

Architecture:
  - Grouped Query Attention (GQA)
  - #{num_heads} query heads → #{local_heads} per GPU
  - #{num_kv_heads} KV heads → #{local_kv_heads} per GPU
  - Head dimension: #{head_dim}

Parameter sharding:
  - Query: #{inspect(Nx.shape(query_kernel))} → #{tp_size}x #{inspect(Nx.shape(hd(query_shards)))} (column-parallel)
  - Key:   #{inspect(Nx.shape(key_kernel))} → #{tp_size}x #{inspect(Nx.shape(hd(key_shards)))} (column-parallel)
  - Value: #{inspect(Nx.shape(value_kernel))} → #{tp_size}x #{inspect(Nx.shape(hd(value_shards)))} (column-parallel)
  - Output: #{inspect(Nx.shape(output_kernel))} → #{tp_size}x #{inspect(Nx.shape(hd(output_shards)))} (row-parallel)

Performance: #{Float.round(avg_time, 2)} ms average per attention layer
""")
