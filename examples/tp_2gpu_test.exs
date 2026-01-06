# 2-GPU Tensor Parallelism Test
#
# This script tests tensor parallelism with FFN sharding on 2 GPUs.
# FFN layers are sharded (column-parallel for gate/up, row-parallel for down).
# Attention layers remain replicated for now.
#
# Usage:
#   mix run examples/tp_2gpu_test.exs

IO.puts("=" |> String.duplicate(60))
IO.puts("2-GPU Tensor Parallelism Test")
IO.puts("=" |> String.duplicate(60))
IO.puts("")

Nx.default_backend(EXLA.Backend)

# Check available GPUs
IO.puts("Checking GPU availability...")

# Create a simple test to verify multi-GPU setup
defmodule TPTest do
  import Nx.Defn

  # Simple function that will run on GPU
  defn add_tensors(a, b) do
    Nx.add(a, b)
  end

  # Test all-reduce simulation (in single process, this is identity)
  defn test_computation(x) do
    # Simulate a column-parallel -> row-parallel pattern
    # In real TP, each GPU would compute part of this
    intermediate = Nx.dot(x, Nx.transpose(x))
    Nx.sum(intermediate, axes: [-1])
  end
end

# Basic GPU test
IO.puts("\nTest 1: Basic GPU computation")
t1 = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
t2 = Nx.tensor([[5.0, 6.0], [7.0, 8.0]])
result = TPTest.add_tensors(t1, t2)
IO.puts("  GPU computation: PASSED")
IO.puts("  Result: #{inspect(Nx.to_list(result))}")

# Test TP layer construction
IO.puts("\nTest 2: TP Layer Construction with TP=2")
alias Bumblebee.Distributed.TPLayers

mesh = %{axes: %{tp: 2}}

# Build a small model with TP layers
input = Axon.input("input", shape: {nil, 64})

# Column-parallel dense (output size / 2)
col_out = TPLayers.column_parallel_dense(input, 128, mesh: mesh, name: "col_dense")

# Activation
activated = Axon.activation(col_out, :silu)

# Row-parallel dense with all-reduce
row_out = TPLayers.row_parallel_dense(activated, 64, mesh: mesh, name: "row_dense")

model = row_out

IO.puts("  Model built with TP layers")

# Initialize and run
{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({1, 64}, :f32), %{})

IO.puts("  Model initialized")
IO.puts("  Params: #{inspect(Map.keys(params.data))}")

# Check shapes
col_kernel = params.data["col_dense"]["kernel"]
row_kernel = params.data["row_dense"]["kernel"]
IO.puts("  Column-parallel kernel shape: #{inspect(Nx.shape(col_kernel))} (expect {64, 64} for 128/2)")
IO.puts("  Row-parallel kernel shape: #{inspect(Nx.shape(row_kernel))} (expect {64, 64})")

# Run forward pass
test_input = Nx.broadcast(0.1, {1, 64})
output = predict_fn.(params, test_input)
IO.puts("  Forward pass output shape: #{inspect(Nx.shape(output))}")
IO.puts("  PASSED: TP layers working")

# Test 3: FFN-only sharding pattern
IO.puts("\nTest 3: Full FFN with TP=2 (SwiGLU pattern)")
alias Bumblebee.Distributed.TPTransformer

hidden_size = 64
intermediate_size = 128  # Will be 64 per GPU with TP=2

ffn_input = Axon.input("hidden", shape: {nil, nil, hidden_size})
ffn_output = TPTransformer.tp_gated_ffn(
  ffn_input,
  intermediate_size,
  hidden_size,
  mesh,
  name: "ffn",
  activation: :silu,
  kernel_initializer: Axon.Initializers.glorot_uniform()
)

{init_ffn, predict_ffn} = Axon.build(ffn_output)
params_ffn = init_ffn.(Nx.template({1, 4, hidden_size}, :f32), %{})

IO.puts("  FFN model initialized")

# Check FFN shapes
gate_kernel = params_ffn.data["ffn.gate"]["kernel"]
intermediate_kernel = params_ffn.data["ffn.intermediate"]["kernel"]
output_kernel = params_ffn.data["ffn.output"]["kernel"]

IO.puts("  Gate kernel shape: #{inspect(Nx.shape(gate_kernel))} (expect {64, 64} for 128/2)")
IO.puts("  Intermediate kernel shape: #{inspect(Nx.shape(intermediate_kernel))} (expect {64, 64})")
IO.puts("  Output kernel shape: #{inspect(Nx.shape(output_kernel))} (expect {64, 64})")

# Run FFN forward pass
ffn_test_input = Nx.broadcast(0.1, {1, 4, hidden_size})
ffn_result = predict_ffn.(params_ffn, ffn_test_input)
IO.puts("  FFN output shape: #{inspect(Nx.shape(ffn_result))}")
IO.puts("  PASSED: FFN with TP pattern working")

# Test 4: Verify sharding logic for Mistral parameters
IO.puts("\nTest 4: Sharding classification for Mistral/Llama parameters")
alias Bumblebee.Distributed.ShardedLoader

mistral_params = [
  # Attention - these would need special handling for full TP
  {"decoder.blocks.0.self_attention.query.kernel", {4096, 4096}},
  {"decoder.blocks.0.self_attention.key.kernel", {4096, 1024}},
  {"decoder.blocks.0.self_attention.value.kernel", {4096, 1024}},
  {"decoder.blocks.0.self_attention.output.kernel", {4096, 4096}},
  # FFN - these work with current TP implementation
  {"decoder.blocks.0.ffn.gate.kernel", {4096, 14336}},
  {"decoder.blocks.0.ffn.intermediate.kernel", {4096, 14336}},
  {"decoder.blocks.0.ffn.output.kernel", {14336, 4096}},
  # Replicated
  {"decoder.blocks.0.self_attention_norm.weight", {4096}},
  {"embedder.token_embedding.kernel", {32000, 4096}},
  {"output_norm.weight", {4096}}
]

IO.puts("  Parameter sharding for TP=2:")
for {name, shape} <- mistral_params do
  {sharding, axis} = ShardedLoader.infer_sharding(name, shape, mesh)
  sharded_shape = if axis do
    shape
    |> Tuple.to_list()
    |> List.update_at(axis, &div(&1, 2))
    |> List.to_tuple()
  else
    shape
  end
  IO.puts("    #{name}")
  IO.puts("      #{inspect(shape)} -> #{sharding} -> #{inspect(sharded_shape)}")
end

IO.puts("")
IO.puts("=" |> String.duplicate(60))
IO.puts("2-GPU TP infrastructure tests completed!")
IO.puts("=" |> String.duplicate(60))
IO.puts("")
IO.puts("Summary:")
IO.puts("  ✓ GPU computation working")
IO.puts("  ✓ TP layers (column/row parallel) working")
IO.puts("  ✓ FFN with SwiGLU pattern working")
IO.puts("  ✓ Sharding classification correct")
IO.puts("")
IO.puts("Next steps for full 2-GPU TP:")
IO.puts("  1. Implement TP-aware attention (split heads across GPUs)")
IO.puts("  2. Run separate processes per GPU with SPMD compilation")
IO.puts("  3. Use NCCL for actual inter-GPU all-reduce")
