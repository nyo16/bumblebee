# Tensor Parallelism Demo Summary
#
# This script demonstrates the current tensor parallelism implementation in Bumblebee.
#
# WHAT WORKS:
# 1. TP=1 with the distributed module - full text generation
# 2. Parameter sharding logic - correctly identifies which params to shard
# 3. FFN-only mode (shard_attention: false) - keeps attention replicated
# 4. All infrastructure (ShardedLoader, TPLayers, TPTransformer, Distributed)
#
# CURRENT LIMITATION:
# True multi-GPU TP requires running separate processes on each GPU.
# Single-process execution only uses one GPU.
#
# Usage:
#   mix run examples/tp_demo_summary.exs

IO.puts("=" |> String.duplicate(70))
IO.puts("Bumblebee Tensor Parallelism Implementation Summary")
IO.puts("=" |> String.duplicate(70))
IO.puts("")

Nx.default_backend(EXLA.Backend)
model_repo = {:hf, "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}

# Demo 1: Verify infrastructure modules exist
IO.puts("1. INFRASTRUCTURE MODULES")
IO.puts("-" |> String.duplicate(70))
modules = [
  {Bumblebee.Distributed, "Main API for tensor parallelism"},
  {Bumblebee.Distributed.ShardedLoader, "Parameter sharding logic"},
  {Bumblebee.Distributed.TPLayers, "Column/row parallel dense layers"},
  {Bumblebee.Distributed.TPTransformer, "TP transformer blocks"},
  {EXLA.Collective, "NCCL collective operations"}
]
for {mod, desc} <- modules do
  loaded = Code.ensure_loaded?(mod)
  status = if loaded, do: "✓", else: "✗"
  IO.puts("   #{status} #{inspect(mod)}")
  IO.puts("     #{desc}")
end
IO.puts("")

# Demo 2: Show sharding logic
IO.puts("2. PARAMETER SHARDING LOGIC")
IO.puts("-" |> String.duplicate(70))
alias Bumblebee.Distributed.ShardedLoader

mesh = %{name: "tp", axes: %{tp: 2}}
test_params = [
  {"decoder.blocks.0.ffn.gate.kernel", {2048, 5632}},
  {"decoder.blocks.0.ffn.intermediate.kernel", {2048, 5632}},
  {"decoder.blocks.0.ffn.output.kernel", {5632, 2048}},
  {"decoder.blocks.0.self_attention.query.kernel", {2048, 2048}},
  {"decoder.blocks.0.self_attention.output.kernel", {2048, 2048}},
]

IO.puts("   With shard_attention: false (FFN-only mode):")
for {name, shape} <- test_params do
  {sharding, axis} = ShardedLoader.infer_sharding(name, shape, mesh, shard_attention: false)
  sharded = if axis do
    List.to_tuple(List.update_at(Tuple.to_list(shape), axis, &div(&1, 2)))
  else
    shape
  end
  icon = case sharding do
    :column_parallel -> "↕️"
    :row_parallel -> "↔️"
    :replicated -> "📋"
  end
  padding = String.duplicate(" ", max(0, 50 - String.length(name)))
  IO.puts("   #{icon} #{name}#{padding}#{inspect(shape)} -> #{inspect(sharded)}")
end
IO.puts("")

# Demo 3: Working TP=1 generation
IO.puts("3. TEXT GENERATION WITH TP=1 (WORKING)")
IO.puts("-" |> String.duplicate(70))
IO.puts("   Loading TinyLlama with Bumblebee.Distributed.load_model...")

mesh1 = Bumblebee.Distributed.mesh(1)
{:ok, model_info} = Bumblebee.Distributed.load_model(
  model_repo,
  mesh: mesh1,
  device_id: 0,
  shard_attention: false
)
IO.puts("   ✓ Model loaded successfully")

{:ok, tokenizer} = Bumblebee.load_tokenizer(model_repo)
{:ok, generation_config} = Bumblebee.load_generation_config(model_repo)
generation_config = Bumblebee.configure(generation_config, max_new_tokens: 25)

serving = Bumblebee.Distributed.serving(
  model_info, tokenizer, generation_config,
  compile: [batch_size: 1, sequence_length: 64],
  defn_options: [compiler: EXLA]
)
IO.puts("   ✓ Serving created")

prompt = "The capital of France is"
IO.puts("   Generating text...")
IO.puts("   Prompt: \"#{prompt}\"")

result = Nx.Serving.run(serving, prompt)
text = result.results |> List.first() |> Map.get(:text)
IO.puts("   Generated: \"#{text}\"")
IO.puts("")

# Summary
IO.puts("4. SUMMARY")
IO.puts("-" |> String.duplicate(70))
IO.puts("""
   COMPLETED:
   ✓ all_reduce operation in EXLA.MLIR.Value
   ✓ EXLA.Collective module for collective ops
   ✓ ShardedLoader for parameter sharding
   ✓ TPLayers (column/row parallel dense, all_reduce)
   ✓ TPTransformer for building TP models
   ✓ Bumblebee.Distributed main API
   ✓ FFN-only sharding mode (shard_attention: false)
   ✓ Text generation with TP=1

   IN PROGRESS:
   → True multi-GPU execution requires:
     - SPMD compilation mode in EXLA (exists but needs work)
     - OR multi-process architecture (one process per GPU)

   FUTURE WORK:
   → TP-aware attention (split heads across GPUs)
   → Pipeline parallelism
   → Sequence parallelism
""")

IO.puts("=" |> String.duplicate(70))
IO.puts("Demo Complete!")
IO.puts("=" |> String.duplicate(70))
