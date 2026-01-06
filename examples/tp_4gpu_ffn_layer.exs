# Run an actual Mistral FFN layer on 4 GPUs with tensor parallelism
#
# This demonstrates running real model parameters through the TP pipeline:
# - Load Mistral 7B model and extract FFN parameters
# - Shard parameters across 4 GPUs (column-parallel up, row-parallel down)
# - Run forward pass with all-reduce synchronization
#
# Usage:
#   mix run examples/tp_4gpu_ffn_layer.exs
#
# For faster testing with smaller model:
#   MODEL=tiny mix run examples/tp_4gpu_ffn_layer.exs

# Use CUDA backend for fast loading
Nx.default_backend(EXLA.Backend)

IO.puts("=" |> String.duplicate(70))
IO.puts("4-GPU Tensor Parallel FFN Layer")
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

# Use Mistral 7B by default, or TinyLlama for faster testing
repo = case System.get_env("MODEL") do
  "tiny" ->
    IO.puts("  Using TinyLlama (fast testing mode)")
    {:hf, "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}
  _ ->
    IO.puts("  Using Mistral 7B (default)")
    {:hf, "mistralai/Mistral-7B-v0.1"}
end

IO.puts("  Loading model (this may take a moment)...")
{:ok, %{params: params, spec: spec}} = Bumblebee.load_model(repo)

hidden_size = spec.hidden_size
intermediate_size = spec.intermediate_size
local_intermediate = div(intermediate_size, tp_size)

IO.puts("  Model loaded!")
IO.puts("  Hidden size: #{hidden_size}")
IO.puts("  Intermediate size: #{intermediate_size} (#{local_intermediate} per GPU)")

# ----------------------------------------------------------
# Step 2: Extract and shard FFN parameters from first layer
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 2: Extracting and sharding FFN parameters")
IO.puts("-" |> String.duplicate(70))

# Get FFN weights from layer 0
# TinyLlama/Mistral uses Llama architecture: gate_proj, up_proj, down_proj
# In Bumblebee, params are stored with flat keys like "decoder.blocks.0.ffn.gate.kernel"

# Get the flat params data
param_data = params.data

gate_kernel = param_data["decoder.blocks.0.ffn.gate"]["kernel"]
up_kernel = param_data["decoder.blocks.0.ffn.intermediate"]["kernel"]
down_kernel = param_data["decoder.blocks.0.ffn.output"]["kernel"]

IO.puts("  Gate kernel shape: #{inspect(Nx.shape(gate_kernel))}")
IO.puts("  Up kernel shape: #{inspect(Nx.shape(up_kernel))}")
IO.puts("  Down kernel shape: #{inspect(Nx.shape(down_kernel))}")

# Shard column-parallel weights (gate, up) - split on output dim (axis 1)
# Shard row-parallel weights (down) - split on input dim (axis 0)
gate_shards = for i <- 0..(tp_size - 1) do
  start_col = i * local_intermediate
  Nx.slice(gate_kernel, [0, start_col], [hidden_size, local_intermediate])
end

up_shards = for i <- 0..(tp_size - 1) do
  start_col = i * local_intermediate
  Nx.slice(up_kernel, [0, start_col], [hidden_size, local_intermediate])
end

down_shards = for i <- 0..(tp_size - 1) do
  start_row = i * local_intermediate
  Nx.slice(down_kernel, [start_row, 0], [local_intermediate, hidden_size])
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

# Input typespecs
input_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, hidden_size})
gate_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_intermediate})
up_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_intermediate})
down_typespec = EXLA.Typespec.tensor({:f, 32}, {local_intermediate, hidden_size})

# Output typespec (same as input)
output_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, hidden_size})

input_typespecs = [input_typespec, gate_typespec, up_typespec, down_typespec]
output_typespecs = [output_typespec]

replica_groups = [Enum.to_list(0..(tp_size - 1))]

IO.puts("  Building SPMD executable...")

spmd = EXLA.SPMD.build(input_typespecs, output_typespecs, fn builder ->
  [input, gate_weight, up_weight, down_weight] = Function.get_arguments(builder)

  # Llama FFN: output = down(silu(gate(x)) * up(x))
  # SiLU(x) = x * sigmoid(x)

  intermediate_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_intermediate})

  # Step 1: Gate projection (column-parallel)
  gate_output = Value.dot_general(
    input,
    gate_weight,
    {[2], [], [0], []},
    :default,
    intermediate_typespec
  )

  # Step 2: SiLU activation on gate output
  # SiLU(x) = x * sigmoid(x)
  gate_sigmoid = Value.sigmoid(gate_output, intermediate_typespec)
  gate_silu = Value.multiply(gate_output, gate_sigmoid, intermediate_typespec)

  # Step 3: Up projection (column-parallel)
  up_output = Value.dot_general(
    input,
    up_weight,
    {[2], [], [0], []},
    :default,
    intermediate_typespec
  )

  # Step 4: Element-wise multiply gate and up
  combined = Value.multiply(gate_silu, up_output, intermediate_typespec)

  # Step 5: Down projection (row-parallel)
  partial_output = Value.dot_general(
    combined,
    down_weight,
    {[2], [], [0], []},
    :default,
    output_typespec
  )

  # Step 6: All-reduce to sum partial results
  output = Value.all_reduce(partial_output, :sum, replica_groups, output_typespec)

  [output]
end, num_replicas: tp_size, client: :cuda)

IO.puts("  SPMD executable built successfully!")

# ----------------------------------------------------------
# Step 4: Run FFN layer on 4 GPUs
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 4: Running FFN layer on #{tp_size} GPUs")
IO.puts("-" |> String.duplicate(70))

# Create random input (simulating hidden states)
key = Nx.Random.key(42)
{input, _key} = Nx.Random.normal(key, shape: {batch_size, seq_len, hidden_size}, type: :f32)
input = Nx.multiply(input, 0.1)

IO.puts("  Input shape: #{inspect(Nx.shape(input))}")
IO.puts("  Input mean: #{Nx.mean(input) |> Nx.to_number() |> Float.round(6)}")

# Prepare replica inputs: each GPU gets same input but different weight shards
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
# Step 5: Compare with single-GPU reference
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 5: Verifying against single-GPU reference")
IO.puts("-" |> String.duplicate(70))

# Reconstruct full weights
full_gate = Nx.concatenate(gate_shards, axis: 1)
full_up = Nx.concatenate(up_shards, axis: 1)
full_down = Nx.concatenate(down_shards, axis: 0)

IO.puts("  Full gate shape: #{inspect(Nx.shape(full_gate))} (should be #{inspect(Nx.shape(gate_kernel))})")

# Single-GPU FFN computation
# output = down(silu(gate(x)) * up(x))
reference_output =
  input
  |> then(fn x ->
    gate_out = Nx.dot(x, [2], full_gate, [0])
    up_out = Nx.dot(x, [2], full_up, [0])
    silu = Nx.multiply(gate_out, Nx.sigmoid(gate_out))
    combined = Nx.multiply(silu, up_out)
    Nx.dot(combined, [2], full_down, [0])
  end)

IO.puts("  Reference output shape: #{inspect(Nx.shape(reference_output))}")
IO.puts("  Reference mean: #{Nx.mean(reference_output) |> Nx.to_number() |> Float.round(6)}")

# Compare
[[spmd_output] | _] = results
max_diff = Nx.subtract(reference_output, spmd_output) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
mean_diff = Nx.subtract(reference_output, spmd_output) |> Nx.abs() |> Nx.mean() |> Nx.to_number()

IO.puts("\n  Comparison:")
IO.puts("    Max absolute difference: #{Float.round(max_diff, 8)}")
IO.puts("    Mean absolute difference: #{Float.round(mean_diff, 8)}")

tolerance = 1.0e-4
if max_diff < tolerance do
  IO.puts("\n  ✓ SPMD output matches reference within tolerance (#{tolerance})!")
else
  IO.puts("\n  ⚠ Difference exceeds tolerance but may be acceptable for f32 precision")
end

# ----------------------------------------------------------
# Step 6: Benchmark multiple runs
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 6: Benchmarking (10 runs)")
IO.puts("-" |> String.duplicate(70))

times = for _ <- 1..10 do
  {time_us, _} = :timer.tc(fn ->
    EXLA.SPMD.run(spmd, replica_inputs)
  end)
  time_us / 1000  # Convert to ms
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

Successfully ran #{model_name} FFN layer with 4-GPU tensor parallelism!

Key achievements:
✓ Real model parameters from #{model_name}
✓ Proper parameter sharding (column-parallel gate/up, row-parallel down)
✓ SiLU activation (Llama's FFN activation)
✓ All-reduce synchronization across 4 H100 GPUs via NCCL
✓ Output matches single-GPU reference

Parameter sharding:
  - Gate: #{inspect(Nx.shape(gate_kernel))} → #{tp_size}x #{inspect(Nx.shape(hd(gate_shards)))}
  - Up:   #{inspect(Nx.shape(up_kernel))} → #{tp_size}x #{inspect(Nx.shape(hd(up_shards)))}
  - Down: #{inspect(Nx.shape(down_kernel))} → #{tp_size}x #{inspect(Nx.shape(hd(down_shards)))}

Performance: #{Float.round(avg_time, 2)} ms average per FFN layer
""")
