# Model Building Test for Tensor Parallelism
#
# This script tests the full model building pipeline without downloading weights.
# It verifies that the TP transformer can be built and initialized.
#
# Usage:
#   mix run examples/tp_model_test.exs

IO.puts("=" |> String.duplicate(60))
IO.puts("Tensor-Parallel Model Building Test")
IO.puts("=" |> String.duplicate(60))
IO.puts("")

Nx.default_backend(EXLA.Backend)

alias Bumblebee.Distributed.TPTransformer

# Create a small test spec mimicking Mistral structure
# (using much smaller dimensions for testing)
test_spec = %Bumblebee.Text.Mistral{
  vocab_size: 1000,
  hidden_size: 64,
  num_blocks: 2,
  num_attention_heads: 4,
  num_key_value_heads: 2,
  intermediate_size: 128,
  activation: :silu,
  max_positions: 128,
  layer_norm_epsilon: 1.0e-5,
  initializer_scale: 0.02,
  attention_window_size: nil,
  rotary_embedding_base: 10000,
  architecture: :for_causal_language_modeling
}

IO.puts("Test Spec:")
IO.puts("  vocab_size: #{test_spec.vocab_size}")
IO.puts("  hidden_size: #{test_spec.hidden_size}")
IO.puts("  num_blocks: #{test_spec.num_blocks}")
IO.puts("  num_attention_heads: #{test_spec.num_attention_heads}")
IO.puts("  num_key_value_heads: #{test_spec.num_key_value_heads}")
IO.puts("  intermediate_size: #{test_spec.intermediate_size}")
IO.puts("")

# Test with TP=2
mesh = Bumblebee.Distributed.mesh(2)
IO.puts("Mesh: #{inspect(mesh)}")
IO.puts("")

# Build TP model
IO.puts("Building TP model...")
model = TPTransformer.build_model(test_spec, mesh)
IO.puts("  Model built successfully!")
IO.puts("")

# Initialize the model
IO.puts("Initializing model...")
{init_fn, predict_fn} = Axon.build(model)

# Create template inputs
batch_size = 1
seq_length = 8

templates = %{
  "input_ids" => Nx.template({batch_size, seq_length}, :s64),
  "attention_mask" => Nx.template({batch_size, seq_length}, :s64),
  "position_ids" => Nx.template({batch_size, seq_length}, :s64)
}

IO.puts("  Templates: #{inspect(Map.keys(templates))}")

params = init_fn.(templates, %{})
IO.puts("  Model initialized!")
IO.puts("  Parameter groups: #{inspect(Map.keys(params.parameters))}")
IO.puts("")

IO.puts("  (skipping parameter count)")
IO.puts("")

# Run forward pass
IO.puts("Running forward pass...")

input_ids = Nx.broadcast(1, {batch_size, seq_length}) |> Nx.as_type(:s64)
attention_mask = Nx.broadcast(1, {batch_size, seq_length}) |> Nx.as_type(:s64)
position_ids = Nx.iota({1, seq_length}, axis: 1) |> Nx.broadcast({batch_size, seq_length}) |> Nx.as_type(:s64)

inputs = %{
  "input_ids" => input_ids,
  "attention_mask" => attention_mask,
  "position_ids" => position_ids
}

outputs = predict_fn.(params, inputs)

IO.puts("  Outputs received!")
IO.puts("  Output keys: #{inspect(Map.keys(outputs))}")

if Map.has_key?(outputs, :logits) do
  IO.puts("  Logits shape: #{inspect(Nx.shape(outputs.logits))}")
end

if Map.has_key?(outputs, :hidden_state) do
  IO.puts("  Hidden state shape: #{inspect(Nx.shape(outputs.hidden_state))}")
end

IO.puts("")
IO.puts("=" |> String.duplicate(60))
IO.puts("Model building test completed successfully!")
IO.puts("=" |> String.duplicate(60))
