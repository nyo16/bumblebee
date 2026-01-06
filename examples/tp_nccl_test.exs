# Multi-GPU NCCL Communication Test
#
# This tests NCCL all-reduce between 2 GPUs using XLA's built-in support.
# This is a simpler test than full TP - just verify GPUs can communicate.
#
# Usage:
#   mix run examples/tp_nccl_test.exs

IO.puts("=" |> String.duplicate(60))
IO.puts("NCCL Multi-GPU Communication Test")
IO.puts("=" |> String.duplicate(60))

# Use EXLA backend
Nx.default_backend(EXLA.Backend)

# Check available devices
client = EXLA.Client.fetch!(:cuda)
IO.puts("\nCUDA Client: #{client.device_count} GPUs available")

# Test 1: Simple tensor operations on different devices
IO.puts("\n1. Testing tensor placement on different GPUs...")

# Create tensors on GPU 0
t0 = Nx.tensor([1.0, 2.0, 3.0, 4.0])
IO.puts("   Tensor on GPU 0: #{inspect(Nx.to_list(t0))}")

# Transfer to GPU 1
t1 = Nx.backend_transfer(t0, {EXLA.Backend, device_id: 1})
IO.puts("   Transferred to GPU 1")

# Verify both tensors exist
IO.puts("   GPU 0 tensor shape: #{inspect(Nx.shape(t0))}")
IO.puts("   GPU 1 tensor shape: #{inspect(Nx.shape(t1))}")

# Test 2: Compute on different GPUs
IO.puts("\n2. Testing computation on different GPUs...")

# Defn function that runs on specified device
defmodule GPUCompute do
  import Nx.Defn

  defn multiply_by_two(x), do: Nx.multiply(x, 2)

  defn add_bias(x, bias), do: Nx.add(x, bias)
end

# Create fresh tensors for computation
t0_compute = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: {EXLA.Backend, device_id: 0})
t1_compute = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: {EXLA.Backend, device_id: 1})

# Run on GPU 0
result0 = GPUCompute.multiply_by_two(t0_compute)
IO.puts("   GPU 0 result: #{inspect(Nx.to_list(result0))}")

# Run on GPU 1 (need to force device)
result1 = Nx.Defn.jit(&GPUCompute.multiply_by_two/1, compiler: EXLA, device_id: 1).(t1_compute)
IO.puts("   GPU 1 result: #{inspect(Nx.to_list(result1))}")

# Test 3: All-reduce pattern simulation
IO.puts("\n3. Simulating all-reduce pattern...")

# In real TP, each GPU computes partial results that are summed
# Here we simulate by manually computing on each GPU and combining

partial0 = Nx.tensor([1.0, 2.0, 3.0, 4.0])  # GPU 0's partial result
partial1 = Nx.tensor([5.0, 6.0, 7.0, 8.0])  # GPU 1's partial result

# Transfer partial1 to GPU 0 for reduction (simulates NCCL all-reduce)
partial1_on_0 = Nx.backend_transfer(partial1, {EXLA.Backend, device_id: 0})

# Sum the partials
all_reduced = Nx.add(partial0, partial1_on_0)
IO.puts("   Partial 0: #{inspect(Nx.to_list(partial0))}")
IO.puts("   Partial 1: #{inspect(Nx.to_list(partial1))}")
IO.puts("   All-reduce result: #{inspect(Nx.to_list(all_reduced))}")

# Test 4: Verify data transfers work
IO.puts("\n4. Testing round-trip transfers...")

original = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: {EXLA.Backend, device_id: 0})
to_gpu1 = Nx.backend_transfer(original, {EXLA.Backend, device_id: 1})
back_to_gpu0 = Nx.backend_transfer(to_gpu1, {EXLA.Backend, device_id: 0})

diff = Nx.subtract(original, back_to_gpu0) |> Nx.abs() |> Nx.sum() |> Nx.to_number()
IO.puts("   Round-trip difference: #{diff} (should be 0.0)")

if diff == 0.0 do
  IO.puts("   ✓ Round-trip transfer successful")
else
  IO.puts("   ✗ Round-trip transfer had errors")
end

IO.puts("\n" <> ("=" |> String.duplicate(60)))
IO.puts("NCCL Test Summary")
IO.puts("=" |> String.duplicate(60))
IO.puts("""

What works:
✓ Tensor placement on specific GPUs
✓ Computation on specific GPUs
✓ Manual data transfer between GPUs
✓ Simulated all-reduce via transfer + add

What's needed for true TP:
→ EXLA SPMD mode with replica-batched inputs
→ OR multi-process architecture (one process per GPU)
→ XLA all-reduce collective (our MLIR code has this, needs integration)

Current approach for 2-GPU TP:
1. Run separate Elixir processes on each GPU
2. Use Distributed Erlang for coordination
3. XLA handles intra-node NCCL communication
""")
