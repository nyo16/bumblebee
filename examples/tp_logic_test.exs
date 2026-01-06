# Logic-only Tensor Parallelism Test
#
# This script tests the tensor parallelism logic WITHOUT requiring EXLA.
# It verifies mesh creation, sharding inference, and tensor slicing.
#
# Usage:
#   mix run examples/tp_logic_test.exs

# Disable default backend to use pure Nx
Nx.default_backend(Nx.BinaryBackend)

IO.puts("=" |> String.duplicate(60))
IO.puts("Tensor Parallelism Logic Test (No EXLA)")
IO.puts("=" |> String.duplicate(60))
IO.puts("")

# Test 1: Mesh creation
IO.puts("Test 1: Mesh creation")
mesh = Bumblebee.Distributed.mesh(2)
expected = %{name: "tp", axes: %{tp: 2}}
if mesh == expected do
  IO.puts("  PASSED: #{inspect(mesh)}")
else
  IO.puts("  FAILED: Got #{inspect(mesh)}, expected #{inspect(expected)}")
end
IO.puts("")

# Test 2: Mesh with data parallelism
IO.puts("Test 2: Mesh with DP")
mesh_tp_dp = Bumblebee.Distributed.mesh(2, dp_size: 2)
expected_dp = %{name: "tp", axes: %{tp: 2, dp: 2}}
if mesh_tp_dp == expected_dp do
  IO.puts("  PASSED: #{inspect(mesh_tp_dp)}")
else
  IO.puts("  FAILED: Got #{inspect(mesh_tp_dp)}, expected #{inspect(expected_dp)}")
end
IO.puts("")

# Test 3: Mesh validation
IO.puts("Test 3: Mesh validation")
case Bumblebee.Distributed.validate_mesh(mesh, :tensor_parallel) do
  :ok -> IO.puts("  PASSED: Mesh validation succeeded")
  {:error, msg} -> IO.puts("  FAILED: #{msg}")
end
IO.puts("")

# Test 4: Sharding inference
IO.puts("Test 4: Sharding inference")
alias Bumblebee.Distributed.ShardedLoader

test_cases = [
  # QKV projections - column parallel
  {"decoder.blocks.0.self_attention.query.kernel", {4096, 4096}, :column_parallel, 1},
  {"decoder.blocks.0.self_attention.key.kernel", {4096, 1024}, :column_parallel, 1},
  {"decoder.blocks.0.self_attention.value.kernel", {4096, 1024}, :column_parallel, 1},
  # HF naming for QKV
  {"model.layers.0.self_attn.q_proj.weight", {4096, 4096}, :column_parallel, 1},
  {"model.layers.0.self_attn.k_proj.weight", {1024, 4096}, :column_parallel, 1},
  {"model.layers.0.self_attn.v_proj.weight", {1024, 4096}, :column_parallel, 1},
  # Attention output - row parallel
  {"decoder.blocks.0.self_attention.output.kernel", {4096, 4096}, :row_parallel, 0},
  {"model.layers.0.self_attn.o_proj.weight", {4096, 4096}, :row_parallel, 0},
  # FFN gate/up - column parallel
  {"decoder.blocks.0.ffn.gate.kernel", {4096, 14336}, :column_parallel, 1},
  {"decoder.blocks.0.ffn.intermediate.kernel", {4096, 14336}, :column_parallel, 1},
  {"model.layers.0.mlp.gate_proj.weight", {14336, 4096}, :column_parallel, 1},
  {"model.layers.0.mlp.up_proj.weight", {14336, 4096}, :column_parallel, 1},
  # FFN down - row parallel
  {"decoder.blocks.0.ffn.output.kernel", {14336, 4096}, :row_parallel, 0},
  {"model.layers.0.mlp.down_proj.weight", {4096, 14336}, :row_parallel, 0},
  # Replicated params
  {"decoder.blocks.0.self_attention_norm.scale", {4096}, :replicated, nil},
  {"embedder.token_embedding.kernel", {32000, 4096}, :replicated, nil},
  {"model.embed_tokens.weight", {32000, 4096}, :replicated, nil}
]

passed = 0
failed = 0

for {name, shape, expected_type, expected_axis} <- test_cases do
  {actual_type, actual_axis} = ShardedLoader.infer_sharding(name, shape, mesh)
  if actual_type == expected_type and actual_axis == expected_axis do
    IO.puts("  PASSED: #{name}")
    passed = passed + 1
  else
    IO.puts("  FAILED: #{name}")
    IO.puts("    Expected: {#{expected_type}, #{inspect(expected_axis)}}")
    IO.puts("    Got: {#{actual_type}, #{inspect(actual_axis)}}")
    failed = failed + 1
  end
end

IO.puts("  Summary: #{passed} passed, #{failed} failed")
IO.puts("")

# Test 5: Tensor sharding
IO.puts("Test 5: Tensor sharding")
tp_size = 2

# Create a test tensor
test_tensor = Nx.iota({4, 8}, type: :f32)
IO.puts("  Original tensor shape: #{inspect(Nx.shape(test_tensor))}")

# Column parallel (axis 1) - should split 8 -> 4
col_shard_0 = ShardedLoader.shard_tensor(test_tensor, :column_parallel, 1, 0, tp_size)
col_shard_1 = ShardedLoader.shard_tensor(test_tensor, :column_parallel, 1, 1, tp_size)

if Nx.shape(col_shard_0) == {4, 4} and Nx.shape(col_shard_1) == {4, 4} do
  IO.puts("  PASSED: Column-parallel sharding")
  IO.puts("    Shard 0 shape: #{inspect(Nx.shape(col_shard_0))}")
  IO.puts("    Shard 1 shape: #{inspect(Nx.shape(col_shard_1))}")
else
  IO.puts("  FAILED: Column-parallel sharding")
  IO.puts("    Expected: {4, 4}, Got: #{inspect(Nx.shape(col_shard_0))}")
end

# Row parallel (axis 0) - should split 4 -> 2
row_shard_0 = ShardedLoader.shard_tensor(test_tensor, :row_parallel, 0, 0, tp_size)
row_shard_1 = ShardedLoader.shard_tensor(test_tensor, :row_parallel, 0, 1, tp_size)

if Nx.shape(row_shard_0) == {2, 8} and Nx.shape(row_shard_1) == {2, 8} do
  IO.puts("  PASSED: Row-parallel sharding")
  IO.puts("    Shard 0 shape: #{inspect(Nx.shape(row_shard_0))}")
  IO.puts("    Shard 1 shape: #{inspect(Nx.shape(row_shard_1))}")
else
  IO.puts("  FAILED: Row-parallel sharding")
  IO.puts("    Expected: {2, 8}, Got: #{inspect(Nx.shape(row_shard_0))}")
end

# Verify shard contents
expected_col_0 = Nx.slice(test_tensor, [0, 0], [4, 4])
expected_col_1 = Nx.slice(test_tensor, [0, 4], [4, 4])
expected_row_0 = Nx.slice(test_tensor, [0, 0], [2, 8])
expected_row_1 = Nx.slice(test_tensor, [2, 0], [2, 8])

if Nx.equal(col_shard_0, expected_col_0) |> Nx.all() |> Nx.to_number() == 1 do
  IO.puts("  PASSED: Column shard 0 contents correct")
else
  IO.puts("  FAILED: Column shard 0 contents incorrect")
end

if Nx.equal(col_shard_1, expected_col_1) |> Nx.all() |> Nx.to_number() == 1 do
  IO.puts("  PASSED: Column shard 1 contents correct")
else
  IO.puts("  FAILED: Column shard 1 contents incorrect")
end

if Nx.equal(row_shard_0, expected_row_0) |> Nx.all() |> Nx.to_number() == 1 do
  IO.puts("  PASSED: Row shard 0 contents correct")
else
  IO.puts("  FAILED: Row shard 0 contents incorrect")
end

if Nx.equal(row_shard_1, expected_row_1) |> Nx.all() |> Nx.to_number() == 1 do
  IO.puts("  PASSED: Row shard 1 contents correct")
else
  IO.puts("  FAILED: Row shard 1 contents incorrect")
end

# Replicated - should return full tensor
replicated = ShardedLoader.shard_tensor(test_tensor, :replicated, nil, 0, tp_size)
if Nx.shape(replicated) == {4, 8} do
  IO.puts("  PASSED: Replicated tensor unchanged")
else
  IO.puts("  FAILED: Replicated tensor changed")
end
IO.puts("")

# Test 6: Sharding patterns list
IO.puts("Test 6: Sharding patterns documentation")
patterns = ShardedLoader.sharding_patterns()
IO.puts("  Found #{length(patterns)} sharding patterns:")
for {regex, sharding, desc} <- patterns do
  IO.puts("    #{desc}: #{sharding}")
end
IO.puts("")

# Test 7: Replica groups generation
IO.puts("Test 7: Replica groups generation")

# Test TP=2
groups_tp2 = EXLA.Collective.replica_groups(2)
expected_tp2 = [[0, 1]]
if groups_tp2 == expected_tp2 do
  IO.puts("  PASSED: TP=2 replica groups: #{inspect(groups_tp2)}")
else
  IO.puts("  FAILED: TP=2 expected #{inspect(expected_tp2)}, got #{inspect(groups_tp2)}")
end

# Test TP=4
groups_tp4 = EXLA.Collective.replica_groups(4)
expected_tp4 = [[0, 1, 2, 3]]
if groups_tp4 == expected_tp4 do
  IO.puts("  PASSED: TP=4 replica groups: #{inspect(groups_tp4)}")
else
  IO.puts("  FAILED: TP=4 expected #{inspect(expected_tp4)}, got #{inspect(groups_tp4)}")
end

# Test TP=2, DP=2
groups_tp2_dp2 = EXLA.Collective.replica_groups(2, dp_size: 2)
expected_tp2_dp2 = [[0, 1], [2, 3]]
if groups_tp2_dp2 == expected_tp2_dp2 do
  IO.puts("  PASSED: TP=2 DP=2 replica groups: #{inspect(groups_tp2_dp2)}")
else
  IO.puts("  FAILED: TP=2 DP=2 expected #{inspect(expected_tp2_dp2)}, got #{inspect(groups_tp2_dp2)}")
end

# Test TP=2, DP=4
groups_tp2_dp4 = EXLA.Collective.replica_groups(2, dp_size: 4)
expected_tp2_dp4 = [[0, 1], [2, 3], [4, 5], [6, 7]]
if groups_tp2_dp4 == expected_tp2_dp4 do
  IO.puts("  PASSED: TP=2 DP=4 replica groups: #{inspect(groups_tp2_dp4)}")
else
  IO.puts("  FAILED: TP=2 DP=4 expected #{inspect(expected_tp2_dp4)}, got #{inspect(groups_tp2_dp4)}")
end
IO.puts("")

IO.puts("=" |> String.duplicate(60))
IO.puts("Logic tests completed!")
IO.puts("=" |> String.duplicate(60))
