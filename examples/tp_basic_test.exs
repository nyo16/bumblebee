# Basic Tensor Parallelism Test
#
# This script tests the basic components of tensor parallelism
# without loading a full model. Good for verifying the infrastructure.
#
# Usage:
#   XLA_FLAGS="--xla_force_host_platform_device_count=2" mix run examples/tp_basic_test.exs

IO.puts("=" |> String.duplicate(60))
IO.puts("Basic Tensor Parallelism Test")
IO.puts("=" |> String.duplicate(60))
IO.puts("")

# Test 1: Mesh creation
IO.puts("Test 1: Mesh creation")
mesh = Bumblebee.Distributed.mesh(2)
IO.puts("  Created mesh: #{inspect(mesh)}")
IO.puts("  Expected: %{name: \"tp\", axes: %{tp: 2}}")
IO.puts("")

# Test 2: Mesh validation
IO.puts("Test 2: Mesh validation")
case Bumblebee.Distributed.validate_mesh(mesh, :tensor_parallel) do
  :ok -> IO.puts("  Mesh validation: PASSED")
  {:error, msg} -> IO.puts("  Mesh validation: FAILED - #{msg}")
end
IO.puts("")

# Test 3: Sharding inference
IO.puts("Test 3: Sharding inference")
alias Bumblebee.Distributed.ShardedLoader

test_cases = [
  {"decoder.blocks.0.self_attention.query.kernel", {4096, 4096}},
  {"decoder.blocks.0.self_attention.key.kernel", {4096, 1024}},
  {"decoder.blocks.0.self_attention.value.kernel", {4096, 1024}},
  {"decoder.blocks.0.self_attention.output.kernel", {4096, 4096}},
  {"decoder.blocks.0.ffn.gate.kernel", {4096, 14336}},
  {"decoder.blocks.0.ffn.intermediate.kernel", {4096, 14336}},
  {"decoder.blocks.0.ffn.output.kernel", {14336, 4096}},
  {"decoder.blocks.0.self_attention_norm.scale", {4096}},
  {"embedder.token_embedding.kernel", {32000, 4096}}
]

for {name, shape} <- test_cases do
  {sharding_type, axis} = ShardedLoader.infer_sharding(name, shape, mesh)
  IO.puts("  #{name}")
  IO.puts("    Shape: #{inspect(shape)} -> Sharding: #{sharding_type}, Axis: #{inspect(axis)}")
end
IO.puts("")

# Test 4: Tensor sharding
IO.puts("Test 4: Tensor sharding")
tp_size = 2

# Create a test tensor
test_tensor = Nx.iota({4, 8}, type: :f32)
IO.puts("  Original tensor shape: #{inspect(Nx.shape(test_tensor))}")

# Column parallel (axis 1)
col_shard_0 = ShardedLoader.shard_tensor(test_tensor, :column_parallel, 1, 0, tp_size)
col_shard_1 = ShardedLoader.shard_tensor(test_tensor, :column_parallel, 1, 1, tp_size)
IO.puts("  Column-parallel shard 0 shape: #{inspect(Nx.shape(col_shard_0))}")
IO.puts("  Column-parallel shard 1 shape: #{inspect(Nx.shape(col_shard_1))}")

# Row parallel (axis 0)
row_shard_0 = ShardedLoader.shard_tensor(test_tensor, :row_parallel, 0, 0, tp_size)
row_shard_1 = ShardedLoader.shard_tensor(test_tensor, :row_parallel, 0, 1, tp_size)
IO.puts("  Row-parallel shard 0 shape: #{inspect(Nx.shape(row_shard_0))}")
IO.puts("  Row-parallel shard 1 shape: #{inspect(Nx.shape(row_shard_1))}")

# Replicated
replicated = ShardedLoader.shard_tensor(test_tensor, :replicated, nil, 0, tp_size)
IO.puts("  Replicated shape: #{inspect(Nx.shape(replicated))}")
IO.puts("")

# Test 5: TP layer shapes
IO.puts("Test 5: TP layer graph construction")
alias Bumblebee.Distributed.TPLayers

input = Axon.input("input", shape: {nil, 4096})

# Column parallel dense
col_dense = TPLayers.column_parallel_dense(input, 4096, mesh: mesh, name: "query")
IO.puts("  Created column_parallel_dense layer")

# Row parallel dense
row_dense = TPLayers.row_parallel_dense(input, 4096, mesh: mesh, name: "output")
IO.puts("  Created row_parallel_dense layer")

IO.puts("")

# Test 6: Replica groups
IO.puts("Test 6: EXLA.Collective replica groups")
groups_tp2 = EXLA.Collective.replica_groups(2)
IO.puts("  TP=2: #{inspect(groups_tp2)}")

groups_tp4 = EXLA.Collective.replica_groups(4)
IO.puts("  TP=4: #{inspect(groups_tp4)}")

groups_tp2_dp2 = EXLA.Collective.replica_groups(2, dp_size: 2)
IO.puts("  TP=2, DP=2: #{inspect(groups_tp2_dp2)}")
IO.puts("")

IO.puts("=" |> String.duplicate(60))
IO.puts("All basic tests completed!")
IO.puts("=" |> String.duplicate(60))
