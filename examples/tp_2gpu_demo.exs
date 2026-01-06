# 2-GPU Tensor Parallelism Demo with FFN-only Sharding
#
# This script demonstrates tensor parallelism with FFN sharding on 2 GPUs.
# Attention layers remain replicated (standard Bumblebee attention).
# FFN layers are sharded (column-parallel for gate/up, row-parallel for down).
#
# Usage:
#   # For real multi-GPU execution:
#   mix run examples/tp_2gpu_demo.exs
#
#   # For simulated 2-GPU testing (CPU):
#   XLA_FLAGS="--xla_force_host_platform_device_count=2" mix run examples/tp_2gpu_demo.exs

IO.puts("=" |> String.duplicate(60))
IO.puts("2-GPU Tensor Parallelism Demo (FFN-only sharding)")
IO.puts("=" |> String.duplicate(60))
IO.puts("")

Nx.default_backend(EXLA.Backend)

# Configuration
model_repo = {:hf, "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}
tp_size = 2
device_id = 0  # This device's ID (would be 1 on second GPU process)

IO.puts("Configuration:")
IO.puts("  Model: #{inspect(model_repo)}")
IO.puts("  TP Size: #{tp_size}")
IO.puts("  Device ID: #{device_id}")
IO.puts("  Sharding Mode: FFN-only (attention replicated)")
IO.puts("")

# Create device mesh
mesh = Bumblebee.Distributed.mesh(tp_size)
IO.puts("Device mesh: #{inspect(mesh)}")
IO.puts("")

# First, let's examine the standard Bumblebee model structure to understand param names
IO.puts("Step 1: Examining standard model structure...")
{:ok, standard_model_info} = Bumblebee.load_model(model_repo)
IO.puts("  Standard model loaded")
IO.puts("  Params type: #{inspect(standard_model_info.params.__struct__)}")

# Show first few param names to understand the structure
sample_params =
  standard_model_info.params.data
  |> Enum.take(5)
  |> Enum.map(fn {layer_name, layer_params} ->
    param_names = Map.keys(layer_params)
    "  #{layer_name}: #{inspect(param_names)}"
  end)
  |> Enum.join("\n")

IO.puts("  Sample layer structure:")
IO.puts(sample_params)
IO.puts("")

# Find FFN layers
ffn_layers =
  standard_model_info.params.data
  |> Enum.filter(fn {name, _} -> String.contains?(name, "ffn") end)
  |> Enum.take(3)
  |> Enum.map(fn {layer_name, layer_params} ->
    shapes = Enum.map(layer_params, fn {param_name, tensor} ->
      "#{param_name}: #{inspect(Nx.shape(tensor))}"
    end) |> Enum.join(", ")
    "  #{layer_name}: #{shapes}"
  end)
  |> Enum.join("\n")

IO.puts("  FFN layer structure:")
IO.puts(ffn_layers)
IO.puts("")

# Now let's test the sharding logic
IO.puts("Step 2: Testing sharding logic with TP=#{tp_size}...")
alias Bumblebee.Distributed.ShardedLoader

# Test with a few param names
test_params = [
  {"decoder.blocks.0.ffn.gate.kernel", {2048, 5632}},
  {"decoder.blocks.0.ffn.intermediate.kernel", {2048, 5632}},
  {"decoder.blocks.0.ffn.output.kernel", {5632, 2048}},
  {"decoder.blocks.0.self_attention.query.kernel", {2048, 2048}},
  {"decoder.blocks.0.self_attention.output.kernel", {2048, 2048}}
]

IO.puts("  Sharding with shard_attention: false (FFN only):")
for {name, shape} <- test_params do
  {sharding_type, axis} = ShardedLoader.infer_sharding(name, shape, mesh, shard_attention: false)
  sharded_shape = if axis do
    shape
    |> Tuple.to_list()
    |> List.update_at(axis, &div(&1, tp_size))
    |> List.to_tuple()
  else
    shape
  end
  IO.puts("    #{name}")
  IO.puts("      #{inspect(shape)} -> #{sharding_type} -> #{inspect(sharded_shape)}")
end
IO.puts("")

# Load with distributed mode
IO.puts("Step 3: Loading model with tensor parallelism...")
IO.puts("  (shard_attention: false to keep attention replicated)")

{:ok, model_info} = Bumblebee.Distributed.load_model(
  model_repo,
  mesh: mesh,
  device_id: device_id,
  shard_attention: false,  # FFN only
  log_sharding: false      # Set to true to see all sharding decisions
)

IO.puts("  Model loaded with distributed params")
IO.puts("")

# Show some sharded param shapes
IO.puts("Step 4: Verifying sharded parameter shapes...")
sharded_ffn_layers =
  model_info.params.data
  |> Enum.filter(fn {name, _} -> String.contains?(name, "ffn") end)
  |> Enum.take(3)
  |> Enum.map(fn {layer_name, layer_params} ->
    shapes = Enum.map(layer_params, fn {param_name, tensor} ->
      "#{param_name}: #{inspect(Nx.shape(tensor))}"
    end) |> Enum.join(", ")
    "  #{layer_name}: #{shapes}"
  end)
  |> Enum.join("\n")

IO.puts("  Sharded FFN params:")
IO.puts(sharded_ffn_layers)
IO.puts("")

# Show attention params (should be unchanged)
attention_layers =
  model_info.params.data
  |> Enum.filter(fn {name, _} -> String.contains?(name, "self_attention") end)
  |> Enum.take(1)
  |> Enum.map(fn {layer_name, layer_params} ->
    shapes = Enum.map(layer_params, fn {param_name, tensor} ->
      "#{param_name}: #{inspect(Nx.shape(tensor))}"
    end) |> Enum.join(", ")
    "  #{layer_name}: #{shapes}"
  end)
  |> Enum.join("\n")

IO.puts("  Attention params (should be unchanged):")
IO.puts(attention_layers)
IO.puts("")

IO.puts("Step 5: Loading tokenizer...")
{:ok, tokenizer} = Bumblebee.load_tokenizer(model_repo)
IO.puts("  Tokenizer loaded")
IO.puts("")

IO.puts("Step 6: Loading generation config...")
{:ok, generation_config} = Bumblebee.load_generation_config(model_repo)
generation_config = Bumblebee.configure(generation_config, max_new_tokens: 20)
IO.puts("  Generation config loaded")
IO.puts("")

IO.puts("Step 7: Creating serving...")
serving = Bumblebee.Distributed.serving(
  model_info,
  tokenizer,
  generation_config,
  compile: [batch_size: 1, sequence_length: 64],
  defn_options: [compiler: EXLA]
)
IO.puts("  Serving created")
IO.puts("")

# Run inference
prompt = "The capital of France is"
IO.puts("=" |> String.duplicate(60))
IO.puts("Running inference with TP=#{tp_size} (FFN sharding)...")
IO.puts("Prompt: \"#{prompt}\"")
IO.puts("-" |> String.duplicate(60))

result = Nx.Serving.run(serving, prompt)

IO.puts("")
IO.puts("Generated text:")
IO.puts(result.results |> List.first() |> Map.get(:text))
IO.puts("")
IO.puts("=" |> String.duplicate(60))
IO.puts("2-GPU Demo Complete!")
IO.puts("=" |> String.duplicate(60))
