# CPU SPMD Test - Verifies SPMD logic without NCCL
#
# Usage:
#   XLA_FLAGS="--xla_force_host_platform_device_count=4" mix run examples/spmd_cpu_test.exs

IO.puts("=" |> String.duplicate(60))
IO.puts("CPU SPMD Test (4 simulated devices)")
IO.puts("=" |> String.duplicate(60))

# Use host client for CPU
client = EXLA.Client.fetch!(:host)
IO.puts("\nHost Client: #{client.device_count} devices (may be 1 if XLA_FLAGS not set)")

IO.puts("\n1. Testing basic SPMD addition...")

typespec = EXLA.Typespec.tensor({:f, 32}, {4})

spmd = EXLA.SPMD.build([typespec, typespec], [typespec], fn builder ->
  [a, b] = EXLA.MLIR.Function.get_arguments(builder)
  result = EXLA.MLIR.Value.add(a, b, typespec)
  [result]
end, num_replicas: 4, client: :host)

IO.puts("   SPMD executable built successfully!")
IO.puts("   - num_replicas: #{spmd.num_replicas}")

inputs = [
  [Nx.tensor([1.0, 0.0, 0.0, 0.0]), Nx.tensor([0.0, 0.0, 0.0, 0.0])],
  [Nx.tensor([0.0, 2.0, 0.0, 0.0]), Nx.tensor([0.0, 0.0, 0.0, 0.0])],
  [Nx.tensor([0.0, 0.0, 3.0, 0.0]), Nx.tensor([0.0, 0.0, 0.0, 0.0])],
  [Nx.tensor([0.0, 0.0, 0.0, 4.0]), Nx.tensor([0.0, 0.0, 0.0, 0.0])]
]

IO.puts("\n   Running on 4 simulated CPUs...")
results = EXLA.SPMD.run(spmd, inputs)

IO.puts("\n   Results (each device added its inputs):")
for {[result], device} <- Enum.with_index(results) do
  IO.puts("   Device #{device}: #{inspect(Nx.to_list(result))}")
end

IO.puts("\n2. Testing all-reduce on 4 devices...")

# Build all-reduce SPMD program
spmd_reduce = EXLA.SPMD.all_reduce({:f, 32}, {4}, num_replicas: 4, op: :sum, client: :host)

IO.puts("   All-reduce SPMD built successfully!")

reduce_inputs = [
  [Nx.tensor([1.0, 0.0, 0.0, 0.0])],
  [Nx.tensor([0.0, 2.0, 0.0, 0.0])],
  [Nx.tensor([0.0, 0.0, 3.0, 0.0])],
  [Nx.tensor([0.0, 0.0, 0.0, 4.0])]
]

IO.puts("\n   Running all-reduce on 4 simulated devices...")
reduce_results = EXLA.SPMD.run(spmd_reduce, reduce_inputs)

IO.puts("\n   All-reduce results (all devices should have same sum):")
for {[result], device} <- Enum.with_index(reduce_results) do
  IO.puts("   Device #{device}: #{inspect(Nx.to_list(result))}")
end

expected = [1.0, 2.0, 3.0, 4.0]
all_correct = Enum.all?(reduce_results, fn [result] ->
  Nx.to_list(result) == expected
end)

IO.puts("\n   Expected: #{inspect(expected)}")
if all_correct do
  IO.puts("   ✓ All-reduce successful! All devices have the same result.")
else
  IO.puts("   ✗ All-reduce FAILED - results differ across devices")
end

IO.puts("\n" <> ("=" |> String.duplicate(60)))
IO.puts("CPU SPMD Test Complete!")
IO.puts("=" |> String.duplicate(60))
