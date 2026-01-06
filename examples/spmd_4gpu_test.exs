# 4-GPU SPMD Test
#
# Tests the EXLA.SPMD module with 4 GPUs
#
# Usage:
#   mix run examples/spmd_4gpu_test.exs

IO.puts("=" |> String.duplicate(60))
IO.puts("4-GPU SPMD Test")
IO.puts("=" |> String.duplicate(60))

# Verify CUDA client has 4 GPUs
client = EXLA.Client.fetch!(:cuda)
IO.puts("\nCUDA Client: #{client.device_count} GPUs available")

if client.device_count < 4 do
  IO.puts("ERROR: Need 4 GPUs for this test, only #{client.device_count} available")
  System.halt(1)
end

IO.puts("\n1. Testing basic SPMD addition on 4 GPUs...")

typespec = EXLA.Typespec.tensor({:f, 32}, {4})

spmd = EXLA.SPMD.build([typespec, typespec], [typespec], fn builder ->
  [a, b] = EXLA.MLIR.Function.get_arguments(builder)
  result = EXLA.MLIR.Value.add(a, b, typespec)
  [result]
end, num_replicas: 4)

IO.puts("   SPMD executable built successfully!")
IO.puts("   - num_replicas: #{spmd.num_replicas}")

# Prepare inputs for each GPU
inputs = [
  [Nx.tensor([1.0, 0.0, 0.0, 0.0]), Nx.tensor([0.0, 0.0, 0.0, 0.0])],  # GPU 0
  [Nx.tensor([0.0, 2.0, 0.0, 0.0]), Nx.tensor([0.0, 0.0, 0.0, 0.0])],  # GPU 1
  [Nx.tensor([0.0, 0.0, 3.0, 0.0]), Nx.tensor([0.0, 0.0, 0.0, 0.0])],  # GPU 2
  [Nx.tensor([0.0, 0.0, 0.0, 4.0]), Nx.tensor([0.0, 0.0, 0.0, 0.0])]   # GPU 3
]

IO.puts("\n   Running on 4 GPUs...")
results = EXLA.SPMD.run(spmd, inputs)

IO.puts("\n   Results (each GPU added its inputs):")
for {[result], gpu} <- Enum.with_index(results) do
  IO.puts("   GPU #{gpu}: #{inspect(Nx.to_list(result))}")
end

IO.puts("\n2. Testing all-reduce on 4 GPUs...")

# Build all-reduce SPMD program
spmd_reduce = EXLA.SPMD.all_reduce({:f, 32}, {4}, num_replicas: 4, op: :sum)

IO.puts("   All-reduce SPMD built successfully!")

# Each GPU contributes one element
reduce_inputs = [
  [Nx.tensor([1.0, 0.0, 0.0, 0.0])],  # GPU 0 contributes 1.0 at position 0
  [Nx.tensor([0.0, 2.0, 0.0, 0.0])],  # GPU 1 contributes 2.0 at position 1
  [Nx.tensor([0.0, 0.0, 3.0, 0.0])],  # GPU 2 contributes 3.0 at position 2
  [Nx.tensor([0.0, 0.0, 0.0, 4.0])]   # GPU 3 contributes 4.0 at position 3
]

IO.puts("\n   Running all-reduce on 4 GPUs...")
reduce_results = EXLA.SPMD.run(spmd_reduce, reduce_inputs)

IO.puts("\n   All-reduce results (all GPUs should have same sum):")
for {[result], gpu} <- Enum.with_index(reduce_results) do
  IO.puts("   GPU #{gpu}: #{inspect(Nx.to_list(result))}")
end

# Verify all results are the same
expected = [1.0, 2.0, 3.0, 4.0]
all_correct = Enum.all?(reduce_results, fn [result] ->
  Nx.to_list(result) == expected
end)

IO.puts("\n   Expected: #{inspect(expected)}")
if all_correct do
  IO.puts("   ✓ All-reduce successful! All GPUs have the same result.")
else
  IO.puts("   ✗ All-reduce FAILED - results differ across GPUs")
end

IO.puts("\n" <> ("=" |> String.duplicate(60)))
IO.puts("4-GPU SPMD Test Complete!")
IO.puts("=" |> String.duplicate(60))
