# Multi-Process Tensor Parallelism Demo
#
# True tensor parallelism requires running separate processes on each GPU.
# This script can be launched as either the coordinator or a worker.
#
# Usage for 2-GPU TP:
#   # Terminal 1 (GPU 0):
#   CUDA_VISIBLE_DEVICES=0 TP_DEVICE_ID=0 TP_SIZE=2 mix run examples/tp_multi_process.exs
#
#   # Terminal 2 (GPU 1):
#   CUDA_VISIBLE_DEVICES=1 TP_DEVICE_ID=1 TP_SIZE=2 mix run examples/tp_multi_process.exs
#
# For single-process testing with TP=1:
#   mix run examples/tp_multi_process.exs
#
# Note: For actual multi-GPU inference, you'd typically use:
# - Nx.Serving in distributed mode
# - Or a coordinator process that manages worker processes

# Get TP configuration from environment
tp_size = System.get_env("TP_SIZE", "1") |> String.to_integer()
device_id = System.get_env("TP_DEVICE_ID", "0") |> String.to_integer()

IO.puts("=" |> String.duplicate(60))
IO.puts("Tensor Parallelism Worker")
IO.puts("=" |> String.duplicate(60))
IO.puts("")
IO.puts("Configuration:")
IO.puts("  TP Size: #{tp_size}")
IO.puts("  Device ID: #{device_id}")
IO.puts("  CUDA_VISIBLE_DEVICES: #{System.get_env("CUDA_VISIBLE_DEVICES", "all")}")
IO.puts("")

Nx.default_backend(EXLA.Backend)

# Model configuration
model_repo = {:hf, "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}

# Create device mesh
mesh = Bumblebee.Distributed.mesh(tp_size)
IO.puts("Device mesh: #{inspect(mesh)}")

# Load model with appropriate sharding for this device
IO.puts("\nLoading model for device #{device_id}...")

# With shard_attention: false, attention params are replicated (same on all devices)
# FFN params are sharded:
# - gate/intermediate: split along output dim (each device gets half)
# - output: split along input dim (each device gets half)
{:ok, model_info} = Bumblebee.Distributed.load_model(
  model_repo,
  mesh: mesh,
  device_id: device_id,
  shard_attention: false,  # FFN-only sharding
  log_sharding: false
)

IO.puts("Model loaded for device #{device_id}")

# Report sharded param shapes
if tp_size > 1 do
  sample_ffn = model_info.params.data["decoder.blocks.0.ffn.gate"]
  if sample_ffn do
    IO.puts("  FFN gate kernel shape: #{inspect(Nx.shape(sample_ffn["kernel"]))}")
    IO.puts("  (full shape would be {2048, 5632}, sharded is {2048, #{div(5632, tp_size)}})")
  end
end

# Load tokenizer and generation config
{:ok, tokenizer} = Bumblebee.load_tokenizer(model_repo)
{:ok, generation_config} = Bumblebee.load_generation_config(model_repo)
generation_config = Bumblebee.configure(generation_config, max_new_tokens: 30)

# Create serving
IO.puts("\nCreating serving...")
serving = Bumblebee.Distributed.serving(
  model_info,
  tokenizer,
  generation_config,
  compile: [batch_size: 1, sequence_length: 64],
  defn_options: [compiler: EXLA]
)

# Run inference
prompt = "The capital of France is"
IO.puts("\n" <> ("=" |> String.duplicate(60)))
IO.puts("Running inference (device #{device_id})...")
IO.puts("Prompt: \"#{prompt}\"")
IO.puts("-" |> String.duplicate(60))

result = Nx.Serving.run(serving, prompt)

IO.puts("\nGenerated text:")
IO.puts(result.results |> List.first() |> Map.get(:text))
IO.puts("")
IO.puts("=" |> String.duplicate(60))

# For TP > 1, each process produces partial results that need to be combined
# In a real distributed setup, you'd use:
# 1. NCCL for all-reduce operations during inference
# 2. A coordinator to collect final outputs
if tp_size > 1 do
  IO.puts("")
  IO.puts("Note: With TP=#{tp_size}, this process computed partial results.")
  IO.puts("For correct output, all #{tp_size} processes must run and communicate.")
  IO.puts("The all-reduce operations synchronize results via NCCL.")
end
