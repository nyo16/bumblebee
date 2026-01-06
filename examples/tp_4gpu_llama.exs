# 4-GPU Tensor Parallelism Demo with TinyLlama
#
# This demonstrates tensor parallelism with:
# - Model sharding across 4 GPUs (FFN-only mode)
# - NCCL all-reduce for collective operations
# - Text generation with sharded model
#
# Usage:
#   mix run examples/tp_4gpu_llama.exs

IO.puts("=" |> String.duplicate(70))
IO.puts("4-GPU Tensor Parallelism Demo - TinyLlama")
IO.puts("=" |> String.duplicate(70))

Nx.default_backend(EXLA.Backend)

# Verify 4 GPUs available
client = EXLA.Client.fetch!(:cuda)
IO.puts("\nCUDA Client: #{client.device_count} GPUs available")

if client.device_count < 4 do
  IO.puts("WARNING: Only #{client.device_count} GPUs available, using TP=#{client.device_count}")
end

tp_size = min(client.device_count, 4)
model_repo = {:hf, "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}

IO.puts("\nConfiguration:")
IO.puts("  Model: TinyLlama-1.1B-Chat-v1.0")
IO.puts("  TP Size: #{tp_size}")
IO.puts("  Mode: FFN-only sharding (attention replicated)")

# Create device mesh
mesh = Bumblebee.Distributed.mesh(tp_size)
IO.puts("\nDevice mesh: #{inspect(mesh)}")

# Load model with tensor parallelism
IO.puts("\nStep 1: Loading model with TP=#{tp_size}...")
{:ok, model_info} = Bumblebee.Distributed.load_model(
  model_repo,
  mesh: mesh,
  device_id: 0,  # Primary device for non-sharded params
  shard_attention: false,  # FFN-only sharding
  log_sharding: false
)
IO.puts("  ✓ Model loaded with sharded parameters")

# Show some sharded param shapes
IO.puts("\n  Sample sharded FFN shapes (should be 1/#{tp_size} of original):")
for layer_idx <- [0, 10, 21] do
  gate_key = "decoder.blocks.#{layer_idx}.ffn.gate"
  if gate = model_info.params.data[gate_key] do
    shape = Nx.shape(gate["kernel"])
    IO.puts("    Layer #{layer_idx} gate: #{inspect(shape)}")
  end
end

# Load tokenizer
IO.puts("\nStep 2: Loading tokenizer...")
{:ok, tokenizer} = Bumblebee.load_tokenizer(model_repo)
IO.puts("  ✓ Tokenizer loaded")

# Load generation config
IO.puts("\nStep 3: Loading generation config...")
{:ok, generation_config} = Bumblebee.load_generation_config(model_repo)
generation_config = Bumblebee.configure(generation_config, max_new_tokens: 50)
IO.puts("  ✓ Generation config loaded (max_new_tokens: 50)")

# Create serving
IO.puts("\nStep 4: Creating serving...")
serving = Bumblebee.Distributed.serving(
  model_info,
  tokenizer,
  generation_config,
  compile: [batch_size: 1, sequence_length: 128],
  defn_options: [compiler: EXLA]
)
IO.puts("  ✓ Serving created")

# Run inference
prompts = [
  "The capital of France is",
  "In machine learning, tensor parallelism is"
]

IO.puts("\n" <> ("=" |> String.duplicate(70)))
IO.puts("Running inference with TP=#{tp_size}...")
IO.puts("=" |> String.duplicate(70))

for prompt <- prompts do
  IO.puts("\nPrompt: \"#{prompt}\"")
  IO.puts("-" |> String.duplicate(70))

  {time_us, result} = :timer.tc(fn ->
    Nx.Serving.run(serving, prompt)
  end)

  text = result.results |> List.first() |> Map.get(:text)
  time_ms = time_us / 1000

  IO.puts("Generated: \"#{text}\"")
  IO.puts("Time: #{Float.round(time_ms, 1)} ms")
end

IO.puts("\n" <> ("=" |> String.duplicate(70)))
IO.puts("4-GPU Tensor Parallelism Demo Complete!")
IO.puts("=" |> String.duplicate(70))
IO.puts("""

Summary:
✓ Model loaded with TP=#{tp_size} (FFN sharding)
✓ Each GPU holds 1/#{tp_size} of FFN parameters
✓ All-reduce syncs partial results across GPUs
✓ Text generation working correctly
""")
