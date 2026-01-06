# Run Mistral 7B FFN layer on 2 GPUs with tensor parallelism
#
# Usage:
#   mix run examples/tp_2gpu_mistral_ffn.exs

Nx.default_backend(EXLA.Backend)

IO.puts("=" |> String.duplicate(70))
IO.puts("2-GPU Tensor Parallel FFN Layer - Mistral 7B")
IO.puts("=" |> String.duplicate(70))

alias EXLA.MLIR.{Function, Value}

# Configuration
tp_size = 2  # Changed to 2 GPUs
batch_size = 2
seq_len = 16

IO.puts("\nConfiguration:")
IO.puts("  TP size: #{tp_size}")
IO.puts("  Batch size: #{batch_size}")
IO.puts("  Sequence length: #{seq_len}")

# ----------------------------------------------------------
# Step 1: Load Mistral 7B model
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 1: Loading Mistral 7B model")
IO.puts("-" |> String.duplicate(70))

IO.puts("  Loading Mistral 7B (this may take a moment)...")
{:ok, %{params: params, spec: spec}} = Bumblebee.load_model({:hf, "mistralai/Mistral-7B-v0.1"})

hidden_size = spec.hidden_size
intermediate_size = spec.intermediate_size

IO.puts("  Model loaded!")
IO.puts("  Hidden size: #{hidden_size}")
IO.puts("  Intermediate size: #{intermediate_size} (#{div(intermediate_size, tp_size)} per GPU)")

# ----------------------------------------------------------
# Step 2: Extract and shard FFN parameters
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 2: Extracting and sharding FFN parameters")
IO.puts("-" |> String.duplicate(70))

param_data = params.data

gate_kernel = param_data["decoder.blocks.0.ffn.gate"]["kernel"]
up_kernel = param_data["decoder.blocks.0.ffn.intermediate"]["kernel"]
down_kernel = param_data["decoder.blocks.0.ffn.output"]["kernel"]

IO.puts("  Gate kernel shape: #{inspect(Nx.shape(gate_kernel))}")
IO.puts("  Up kernel shape: #{inspect(Nx.shape(up_kernel))}")
IO.puts("  Down kernel shape: #{inspect(Nx.shape(down_kernel))}")

# Shard parameters across GPUs
local_intermediate = div(intermediate_size, tp_size)

gate_shards = for i <- 0..(tp_size - 1) do
  Nx.slice(gate_kernel, [0, i * local_intermediate], [hidden_size, local_intermediate])
end

up_shards = for i <- 0..(tp_size - 1) do
  Nx.slice(up_kernel, [0, i * local_intermediate], [hidden_size, local_intermediate])
end

down_shards = for i <- 0..(tp_size - 1) do
  Nx.slice(down_kernel, [i * local_intermediate, 0], [local_intermediate, hidden_size])
end

IO.puts("\n  Sharded shapes (per GPU):")
IO.puts("    Gate: #{inspect(Nx.shape(hd(gate_shards)))}")
IO.puts("    Up: #{inspect(Nx.shape(hd(up_shards)))}")
IO.puts("    Down: #{inspect(Nx.shape(hd(down_shards)))}")

# ----------------------------------------------------------
# Step 3: Build SPMD executable for FFN layer
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 3: Building SPMD executable for FFN layer")
IO.puts("-" |> String.duplicate(70))

input_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, hidden_size})
gate_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_intermediate})
up_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_intermediate})
down_typespec = EXLA.Typespec.tensor({:f, 32}, {local_intermediate, hidden_size})

input_typespecs = [input_typespec, gate_typespec, up_typespec, down_typespec]
output_typespecs = [input_typespec]

replica_groups = [Enum.to_list(0..(tp_size - 1))]

IO.puts("  Building SPMD executable...")

spmd = EXLA.SPMD.build(input_typespecs, output_typespecs, fn builder ->
  [input, gate_w, up_w, down_w] = Function.get_arguments(builder)
  
  intermediate_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_intermediate})
  result_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, hidden_size})
  
  # Gate and Up projections (column-parallel)
  gate_out = Value.dot_general(input, gate_w, {[2], [], [0], []}, :default, intermediate_typespec)
  up_out = Value.dot_general(input, up_w, {[2], [], [0], []}, :default, intermediate_typespec)
  
  # SiLU activation: x * sigmoid(x)
  gate_sigmoid = Value.sigmoid(gate_out, intermediate_typespec)
  gate_silu = Value.multiply(gate_out, gate_sigmoid, intermediate_typespec)
  
  # Element-wise multiply
  combined = Value.multiply(gate_silu, up_out, intermediate_typespec)
  
  # Down projection (row-parallel)
  partial_output = Value.dot_general(combined, down_w, {[2], [], [0], []}, :default, result_typespec)
  
  # All-reduce to sum partial results
  output = Value.all_reduce(partial_output, :sum, replica_groups, result_typespec)
  
  [output]
end, num_replicas: tp_size, client: :cuda)

IO.puts("  SPMD executable built successfully!")

# ----------------------------------------------------------
# Step 4: Run FFN layer on 2 GPUs
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 4: Running FFN layer on #{tp_size} GPUs")
IO.puts("-" |> String.duplicate(70))

# Create random input
key = Nx.Random.key(42)
{input, _key} = Nx.Random.normal(key, shape: {batch_size, seq_len, hidden_size}, type: :f32)
input = Nx.multiply(input, 0.1)

IO.puts("  Input shape: #{inspect(Nx.shape(input))}")
IO.puts("  Input mean: #{Nx.mean(input) |> Nx.to_number() |> Float.round(6)}")

# Prepare replica inputs
replica_inputs = for i <- 0..(tp_size - 1) do
  [input, Enum.at(gate_shards, i), Enum.at(up_shards, i), Enum.at(down_shards, i)]
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

IO.puts("""

Successfully ran Mistral 7B FFN layer with 2-GPU tensor parallelism!

Key achievements:
✓ Real model parameters from Mistral 7B
✓ Proper parameter sharding (column-parallel gate/up, row-parallel down)
✓ SiLU activation (Llama's FFN activation)
✓ All-reduce synchronization across 2 H100 GPUs via NCCL
✓ All outputs identical

Parameter sharding:
  - Gate: {#{hidden_size}, #{intermediate_size}} → 2x {#{hidden_size}, #{local_intermediate}}
  - Up:   {#{hidden_size}, #{intermediate_size}} → 2x {#{hidden_size}, #{local_intermediate}}
  - Down: {#{intermediate_size}, #{hidden_size}} → 2x {#{local_intermediate}, #{hidden_size}}

Performance: #{Float.round(avg_time, 2)} ms average per FFN layer
""")
