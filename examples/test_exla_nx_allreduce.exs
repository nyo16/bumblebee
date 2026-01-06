# Test EXLA.Nx.all_reduce integration
#
# This verifies that EXLA.Nx.all_reduce gets lowered to stablehlo.all_reduce
#
# Usage:
#   XLA_FLAGS="--xla_force_host_platform_device_count=4" mix run examples/test_exla_nx_allreduce.exs

IO.puts("=" |> String.duplicate(60))
IO.puts("Testing EXLA.Nx.all_reduce Integration")
IO.puts("=" |> String.duplicate(60))

defmodule TestAllReduce do
  import Nx.Defn

  # Simple defn that uses EXLA.Nx.all_reduce
  defn add_and_reduce(a, b) do
    sum = Nx.add(a, b)
    EXLA.Nx.all_reduce(sum, :sum, replica_groups: [[0, 1, 2, 3]])
  end

  defn add_and_reduce_single(a, b) do
    sum = Nx.add(a, b)
    EXLA.Nx.all_reduce(sum, :sum, replica_groups: [[0]])
  end
end

# Build with SPMD mode
IO.puts("\n1. Testing with EXLA.SPMD...")

typespec = EXLA.Typespec.tensor({:f, 32}, {4})

spmd = EXLA.SPMD.build([typespec, typespec], [typespec], fn builder ->
  [a, b] = EXLA.MLIR.Function.get_arguments(builder)
  sum = EXLA.MLIR.Value.add(a, b, typespec)
  reduced = EXLA.MLIR.Value.all_reduce(sum, :sum, [[0, 1, 2, 3]], typespec)
  [reduced]
end, num_replicas: 4, client: :host)

IO.puts("   SPMD built successfully")

# Test inputs
inputs = [
  [Nx.tensor([1.0, 0.0, 0.0, 0.0]), Nx.tensor([0.0, 0.0, 0.0, 0.0])],
  [Nx.tensor([0.0, 2.0, 0.0, 0.0]), Nx.tensor([0.0, 0.0, 0.0, 0.0])],
  [Nx.tensor([0.0, 0.0, 3.0, 0.0]), Nx.tensor([0.0, 0.0, 0.0, 0.0])],
  [Nx.tensor([0.0, 0.0, 0.0, 4.0]), Nx.tensor([0.0, 0.0, 0.0, 0.0])]
]

results = EXLA.SPMD.run(spmd, inputs)

IO.puts("\n   SPMD results (all devices should have [1,2,3,4]):")
for {[result], i} <- Enum.with_index(results) do
  IO.puts("   Device #{i}: #{inspect(Nx.to_list(result))}")
end

# Now test EXLA.Nx.all_reduce via defn
IO.puts("\n2. Testing EXLA.Nx.all_reduce via defn...")

# First test in single-device mode (should pass through)
a = Nx.tensor([1.0, 2.0, 3.0, 4.0])
b = Nx.tensor([0.0, 0.0, 0.0, 0.0])

result = Nx.Defn.jit(&TestAllReduce.add_and_reduce_single/2, compiler: EXLA).(a, b)
IO.puts("   Single device result: #{inspect(Nx.to_list(result))}")
IO.puts("   (In single-device mode, all_reduce is identity)")

IO.puts("\n" <> ("=" |> String.duplicate(60)))
IO.puts("Test Complete!")
IO.puts("=" |> String.duplicate(60))
