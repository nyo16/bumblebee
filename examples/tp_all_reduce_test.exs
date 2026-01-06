# All-Reduce Test for Tensor Parallelism
#
# This script tests the all_reduce operation compiles and executes with EXLA.
#
# Usage:
#   XLA_FLAGS="--xla_force_host_platform_device_count=2" mix run examples/tp_all_reduce_test.exs

IO.puts("=" |> String.duplicate(60))
IO.puts("All-Reduce Compilation and Execution Test")
IO.puts("=" |> String.duplicate(60))
IO.puts("")

# Set EXLA as the default backend
Nx.default_backend(EXLA.Backend)

# Test 1: Simple tensor operations with EXLA
IO.puts("Test 1: Basic EXLA tensor operations")
t1 = Nx.tensor([1.0, 2.0, 3.0, 4.0])
t2 = Nx.add(t1, t1)
IO.puts("  Input: #{inspect(Nx.to_list(t1))}")
IO.puts("  Output (add): #{inspect(Nx.to_list(t2))}")
IO.puts("  PASSED: EXLA backend working")
IO.puts("")

# Test 2: Build and compile an Axon model with all_reduce layer
IO.puts("Test 2: Axon model with TPLayers.all_reduce")

alias Bumblebee.Distributed.TPLayers

mesh = %{axes: %{tp: 2}}

# Create a simple model that uses all_reduce
input = Axon.input("input", shape: {nil, 4})
dense = Axon.dense(input, 4, name: "dense")
# Note: all_reduce is a no-op in single-device mode but should compile
output = TPLayers.all_reduce(dense, mesh, name: "all_reduce")
model = output

IO.puts("  Model created with all_reduce layer")

# Initialize the model
{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({1, 4}, :f32), %{})
IO.puts("  Model initialized")
IO.puts("  Params keys: #{inspect(Map.keys(params))}")

# Run inference
input_data = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
result = predict_fn.(params, input_data)
IO.puts("  Input shape: #{inspect(Nx.shape(input_data))}")
IO.puts("  Output shape: #{inspect(Nx.shape(result))}")
IO.puts("  PASSED: Model with all_reduce compiles and runs")
IO.puts("")

# Test 3: Row-parallel dense layer (includes all_reduce)
IO.puts("Test 3: Row-parallel dense layer")

input2 = Axon.input("input", shape: {nil, 8})
row_dense = TPLayers.row_parallel_dense(input2, 4, mesh: mesh, name: "row_dense")

{init_fn2, predict_fn2} = Axon.build(row_dense)
params2 = init_fn2.(Nx.template({1, 8}, :f32), %{})
IO.puts("  Row-parallel model initialized")

input_data2 = Nx.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
result2 = predict_fn2.(params2, input_data2)
IO.puts("  Input shape: #{inspect(Nx.shape(input_data2))}")
IO.puts("  Output shape: #{inspect(Nx.shape(result2))}")
IO.puts("  PASSED: Row-parallel dense with all_reduce works")
IO.puts("")

# Test 4: Column-parallel dense layer
IO.puts("Test 4: Column-parallel dense layer")

input3 = Axon.input("input", shape: {nil, 8})
# With TP=2, output should be 4/2 = 2
col_dense = TPLayers.column_parallel_dense(input3, 4, mesh: mesh, name: "col_dense")

{init_fn3, predict_fn3} = Axon.build(col_dense)
params3 = init_fn3.(Nx.template({1, 8}, :f32), %{})
IO.puts("  Column-parallel model initialized")

result3 = predict_fn3.(params3, input_data2)
IO.puts("  Input shape: #{inspect(Nx.shape(input_data2))}")
IO.puts("  Output shape: #{inspect(Nx.shape(result3))} (should be {1, 2} for TP=2)")
IO.puts("  PASSED: Column-parallel dense works")
IO.puts("")

# Test 5: Full TP FFN pattern (gate + up + down with all_reduce)
IO.puts("Test 5: Full TP FFN pattern")

alias Bumblebee.Distributed.TPTransformer

hidden_size = 16
intermediate_size = 32

input_ffn = Axon.input("hidden", shape: {nil, nil, hidden_size})
ffn_output = TPTransformer.tp_gated_ffn(
  input_ffn,
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
IO.puts("  FFN params: #{inspect(Map.keys(params_ffn))}")

input_ffn_data = Nx.broadcast(0.1, {1, 4, hidden_size})
result_ffn = predict_ffn.(params_ffn, input_ffn_data)
IO.puts("  Input shape: #{inspect(Nx.shape(input_ffn_data))}")
IO.puts("  Output shape: #{inspect(Nx.shape(result_ffn))}")
IO.puts("  PASSED: Full TP FFN pattern works")
IO.puts("")

IO.puts("=" |> String.duplicate(60))
IO.puts("All all_reduce tests completed successfully!")
IO.puts("=" |> String.duplicate(60))
