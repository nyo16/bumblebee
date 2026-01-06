# Tensor Parallelism Demo with 4 GPUs
#
# This demonstrates real tensor parallelism:
# - Column-parallel matmul (no communication)
# - Row-parallel matmul with all-reduce
#
# The pattern mirrors what happens in a TP transformer FFN layer:
#   Input (replicated) -> Column-parallel -> Activation -> Row-parallel + AllReduce -> Output (replicated)
#
# Usage:
#   mix run examples/tp_4gpu_demo.exs

IO.puts("=" |> String.duplicate(70))
IO.puts("4-GPU Tensor Parallelism Demo")
IO.puts("=" |> String.duplicate(70))

alias EXLA.MLIR.{Function, Value}

# Configuration
tp_size = 4
hidden_size = 1024
intermediate_size = 4096  # 4x hidden for FFN
batch_size = 2
seq_len = 8

IO.puts("\nConfiguration:")
IO.puts("  TP size: #{tp_size}")
IO.puts("  Hidden size: #{hidden_size}")
IO.puts("  Intermediate size: #{intermediate_size} (#{div(intermediate_size, tp_size)} per GPU)")
IO.puts("  Batch size: #{batch_size}")
IO.puts("  Sequence length: #{seq_len}")

# ----------------------------------------------------------
# Step 1: Create sharded weights (simulating what ShardedLoader does)
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 1: Creating sharded weights")
IO.puts("-" |> String.duplicate(70))

# Column-parallel weight: [hidden, intermediate]
# Each GPU gets [hidden, intermediate/tp_size]
# This is the "up" projection in FFN
local_intermediate = div(intermediate_size, tp_size)

# Create different shards for each GPU
up_weights_shards = for i <- 0..(tp_size - 1) do
  # Each shard has shape [hidden_size, local_intermediate]
  # Initialize with distinct values so we can verify correctness
  Nx.broadcast(0.1 * (i + 1), {hidden_size, local_intermediate})
  |> Nx.add(Nx.iota({hidden_size, local_intermediate}) |> Nx.divide(hidden_size * local_intermediate))
end

IO.puts("  Up projection (column-parallel):")
IO.puts("    Full shape: {#{hidden_size}, #{intermediate_size}}")
IO.puts("    Per-GPU shape: {#{hidden_size}, #{local_intermediate}}")

# Row-parallel weight: [intermediate, hidden]
# Each GPU gets [intermediate/tp_size, hidden]
# This is the "down" projection in FFN - needs all-reduce after
down_weights_shards = for i <- 0..(tp_size - 1) do
  Nx.broadcast(0.05 * (i + 1), {local_intermediate, hidden_size})
  |> Nx.add(Nx.iota({local_intermediate, hidden_size}) |> Nx.divide(local_intermediate * hidden_size))
end

IO.puts("  Down projection (row-parallel):")
IO.puts("    Full shape: {#{intermediate_size}, #{hidden_size}}")
IO.puts("    Per-GPU shape: {#{local_intermediate}, #{hidden_size}}")

# ----------------------------------------------------------
# Step 2: Build SPMD executable for FFN forward pass
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 2: Building SPMD executable")
IO.puts("-" |> String.duplicate(70))

# Input typespecs
input_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, hidden_size})
up_weight_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_intermediate})
down_weight_typespec = EXLA.Typespec.tensor({:f, 32}, {local_intermediate, hidden_size})

# Output typespec (same as input - replicated after all-reduce)
output_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, hidden_size})

input_typespecs = [input_typespec, up_weight_typespec, down_weight_typespec]
output_typespecs = [output_typespec]

replica_groups = [Enum.to_list(0..(tp_size - 1))]

spmd = EXLA.SPMD.build(input_typespecs, output_typespecs, fn builder ->
  [input, up_weight, down_weight] = Function.get_arguments(builder)

  # FFN Forward Pass:
  # 1. Up projection (column-parallel): [batch, seq, hidden] x [hidden, local_inter] = [batch, seq, local_inter]
  # 2. Activation (GELU) - local, no communication
  # 3. Down projection (row-parallel): [batch, seq, local_inter] x [local_inter, hidden] = [batch, seq, hidden]
  # 4. All-reduce to sum partial results across GPUs

  # Step 1: Column-parallel matmul
  intermediate_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_intermediate})

  # dot_general dnums: {lhs_contract, lhs_batch, rhs_contract, rhs_batch}
  up_output = Value.dot_general(
    input,
    up_weight,
    {[2], [], [0], []},  # contract input's axis 2 with weight's axis 0, no batch dims
    :default,
    intermediate_typespec
  )

  # Step 2: GELU activation (simplified as tanh for demo)
  activated = Value.tanh(up_output, intermediate_typespec)

  # Step 3: Row-parallel matmul
  partial_output = Value.dot_general(
    activated,
    down_weight,
    {[2], [], [0], []},
    :default,
    output_typespec
  )

  # Step 4: All-reduce - sum partial results from all GPUs
  # This is the KEY operation for tensor parallelism!
  output = Value.all_reduce(partial_output, :sum, replica_groups, output_typespec)

  [output]
end, num_replicas: tp_size, client: :cuda)

IO.puts("  SPMD executable built successfully")
IO.puts("  Operations: dot_general -> tanh -> dot_general -> all_reduce")

# ----------------------------------------------------------
# Step 3: Run SPMD inference
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 3: Running SPMD inference on #{tp_size} GPUs")
IO.puts("-" |> String.duplicate(70))

# Create input (same for all GPUs - replicated)
input = Nx.iota({batch_size, seq_len, hidden_size}, type: :f32) |> Nx.divide(batch_size * seq_len * hidden_size)

IO.puts("  Input shape: #{inspect(Nx.shape(input))}")
IO.puts("  Input sample: #{inspect(Nx.to_flat_list(input[[0, 0, 0..4]]))}")

# Prepare replica inputs: each GPU gets same input but different weight shards
replica_inputs = for i <- 0..(tp_size - 1) do
  [input, Enum.at(up_weights_shards, i), Enum.at(down_weights_shards, i)]
end

IO.puts("\n  Running SPMD...")
:timer.tc(fn ->
  results = EXLA.SPMD.run(spmd, replica_inputs)

  IO.puts("\n  Results from all GPUs (should be IDENTICAL due to all-reduce):")
  for {[output], gpu} <- Enum.with_index(results) do
    sample = Nx.to_flat_list(output[[0, 0, 0..4]])
    IO.puts("    GPU #{gpu}: shape=#{inspect(Nx.shape(output))}, sample=#{inspect(Enum.map(sample, &Float.round(&1, 4)))}")
  end

  # Verify all outputs are identical
  [[ref_output] | rest] = results
  all_same = Enum.all?(rest, fn [output] ->
    Nx.all_close(ref_output, output, atol: 1.0e-5) |> Nx.to_number() == 1
  end)

  if all_same do
    IO.puts("\n  ✓ All GPU outputs are identical (all-reduce working correctly!)")
  else
    IO.puts("\n  ✗ WARNING: GPU outputs differ!")
  end

  results
end)
|> then(fn {time_us, _results} ->
  IO.puts("  Execution time: #{Float.round(time_us / 1000, 2)} ms")
end)

# ----------------------------------------------------------
# Step 4: Compare with single-GPU reference
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 4: Verifying against single-GPU reference")
IO.puts("-" |> String.duplicate(70))

# Reconstruct full weights
full_up_weight = Nx.concatenate(up_weights_shards, axis: 1)
full_down_weight = Nx.concatenate(down_weights_shards, axis: 0)

IO.puts("  Full up weight shape: #{inspect(Nx.shape(full_up_weight))}")
IO.puts("  Full down weight shape: #{inspect(Nx.shape(full_down_weight))}")

# Single-GPU computation
reference_output =
  input
  |> Nx.dot([2], full_up_weight, [0])
  |> Nx.tanh()
  |> Nx.dot([2], full_down_weight, [0])

IO.puts("  Reference output shape: #{inspect(Nx.shape(reference_output))}")
IO.puts("  Reference sample: #{inspect(Enum.map(Nx.to_flat_list(reference_output[[0, 0, 0..4]]), &Float.round(&1, 4)))}")

# Compare with SPMD result
[[spmd_output] | _] = EXLA.SPMD.run(spmd, replica_inputs)

if Nx.all_close(reference_output, spmd_output, atol: 1.0e-4) |> Nx.to_number() == 1 do
  IO.puts("\n  ✓ SPMD output matches single-GPU reference!")
else
  max_diff = Nx.subtract(reference_output, spmd_output) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
  IO.puts("\n  ✗ SPMD output differs from reference (max diff: #{max_diff})")
end

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
IO.puts("\n" <> ("=" |> String.duplicate(70)))
IO.puts("Summary")
IO.puts("=" |> String.duplicate(70))
IO.puts("""
This demo showed real 4-GPU tensor parallelism:

1. Column-parallel layer (up projection):
   - Each GPU has #{local_intermediate}/#{intermediate_size} of the weight columns
   - Input is replicated, output is partitioned
   - No communication needed

2. Row-parallel layer (down projection):
   - Each GPU has #{local_intermediate}/#{intermediate_size} of the weight rows
   - Input is partitioned, output is partial
   - Requires all-reduce after

3. All-reduce:
   - Sums partial results from all #{tp_size} GPUs
   - All GPUs end up with identical output (replicated)
   - Uses NCCL for efficient GPU-to-GPU communication

This is the exact pattern used in tensor-parallel transformers:
- QKV projections: column-parallel
- Attention output projection: row-parallel + all-reduce
- FFN up/gate: column-parallel
- FFN down: row-parallel + all-reduce
""")
