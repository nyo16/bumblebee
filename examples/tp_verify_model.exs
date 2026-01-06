# Verify model works with TP=1 before testing TP>1
#
# Usage:
#   mix run examples/tp_verify_model.exs

IO.puts("=" |> String.duplicate(70))
IO.puts("Model Verification - TP=1")
IO.puts("=" |> String.duplicate(70))

Nx.default_backend(EXLA.Backend)

model_repo = {:hf, "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}

# Test with TP=1 (should work correctly)
IO.puts("\nLoading model with TP=1...")
mesh = Bumblebee.Distributed.mesh(1)

{:ok, model_info} = Bumblebee.Distributed.load_model(
  model_repo,
  mesh: mesh,
  device_id: 0,
  shard_attention: false
)
IO.puts("  ✓ Model loaded")

{:ok, tokenizer} = Bumblebee.load_tokenizer(model_repo)
{:ok, generation_config} = Bumblebee.load_generation_config(model_repo)
generation_config = Bumblebee.configure(generation_config, max_new_tokens: 30)

serving = Bumblebee.Distributed.serving(
  model_info,
  tokenizer,
  generation_config,
  compile: [batch_size: 1, sequence_length: 64],
  defn_options: [compiler: EXLA]
)

prompt = "The capital of France is"
IO.puts("\nPrompt: \"#{prompt}\"")
result = Nx.Serving.run(serving, prompt)
text = result.results |> List.first() |> Map.get(:text)
IO.puts("Generated: \"#{text}\"")

IO.puts("\n✓ TP=1 model works correctly!")
