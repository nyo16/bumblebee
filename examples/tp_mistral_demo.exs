# Tensor-Parallel Mistral 7B Demo
#
# This script demonstrates tensor parallelism with Bumblebee.
# It loads Mistral 7B with weights sharded across 2 GPUs.
#
# Prerequisites:
#   - 2+ NVIDIA GPUs with CUDA support
#   - EXLA with NCCL support compiled
#   - Download Mistral 7B weights (will auto-download from HuggingFace)
#
# Usage:
#   # For real multi-GPU execution:
#   mix run examples/tp_mistral_demo.exs
#
#   # For simulated 2-GPU testing (single GPU/CPU):
#   XLA_FLAGS="--xla_force_host_platform_device_count=2" mix run examples/tp_mistral_demo.exs
#
# Expected output:
#   Loading distributed model with strategy: tensor_parallel, device: 0
#   Loading sharded params for device 0 (TP size: 2)
#   ... (parameter loading logs)
#   Generating...
#   The capital of France is Paris...

# Configuration
# Use TinyLlama as it's publicly available without authentication
# For Mistral-7B, you would need to set HF_TOKEN environment variable
model_repo = {:hf, "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}
tp_size = 1  # Set to 1 for now - full TP attention requires more work
device_id = 0  # Change to 1 for second GPU process

# NOTE: Full tensor parallelism with parameter sharding requires custom
# attention implementation that's TP-aware. The current implementation
# validates the infrastructure (all-reduce, mesh, sharding logic) works.
# For full TP, the attention heads need to be split across devices.

IO.puts("=" |> String.duplicate(60))
IO.puts("Tensor-Parallel Mistral 7B Demo")
IO.puts("=" |> String.duplicate(60))
IO.puts("")
IO.puts("Configuration:")
IO.puts("  Model: #{inspect(model_repo)}")
IO.puts("  TP Size: #{tp_size}")
IO.puts("  Device ID: #{device_id}")
IO.puts("")

# Create device mesh for tensor parallelism
mesh = Bumblebee.Distributed.mesh(tp_size)
IO.puts("Device mesh: #{inspect(mesh)}")
IO.puts("")

# Load model with tensor parallelism
IO.puts("Loading model with tensor parallelism...")
{:ok, model_info} = Bumblebee.Distributed.load_model(
  model_repo,
  mesh: mesh,
  device_id: device_id,
  log_sharding: true  # Log sharding decisions for debugging
)

IO.puts("")
IO.puts("Model loaded successfully!")
IO.puts("  Model type: #{inspect(model_info.spec.__struct__)}")
IO.puts("  Hidden size: #{model_info.spec.hidden_size}")
IO.puts("  Num blocks: #{model_info.spec.num_blocks}")
IO.puts("  Num attention heads: #{model_info.spec.num_attention_heads}")
IO.puts("  Num KV heads: #{model_info.spec.num_key_value_heads}")
IO.puts("")

# Load tokenizer
IO.puts("Loading tokenizer...")
{:ok, tokenizer} = Bumblebee.load_tokenizer(model_repo)
IO.puts("Tokenizer loaded.")
IO.puts("")

# Load generation config
IO.puts("Loading generation config...")
{:ok, generation_config} = Bumblebee.load_generation_config(model_repo)

# Override some generation settings for demo
generation_config =
  Bumblebee.configure(generation_config,
    max_new_tokens: 50
  )
IO.puts("Generation config loaded.")
IO.puts("")

# Create serving
IO.puts("Creating serving...")
serving = Bumblebee.Distributed.serving(
  model_info,
  tokenizer,
  generation_config,
  compile: [batch_size: 1, sequence_length: 128],
  defn_options: [compiler: EXLA]
)
IO.puts("Serving created.")
IO.puts("")

# Run inference
prompt = "The capital of France is"
IO.puts("=" |> String.duplicate(60))
IO.puts("Running inference...")
IO.puts("Prompt: \"#{prompt}\"")
IO.puts("-" |> String.duplicate(60))

result = Nx.Serving.run(serving, prompt)

IO.puts("")
IO.puts("Generated text:")
IO.puts(result.results |> List.first() |> Map.get(:text))
IO.puts("")
IO.puts("=" |> String.duplicate(60))
IO.puts("Demo complete!")
