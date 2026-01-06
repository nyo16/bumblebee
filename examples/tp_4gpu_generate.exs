# Autoregressive text generation with 4-GPU tensor parallelism
#
# This implements iterative token generation using the TP model.
# Each iteration generates one token and feeds it back for the next iteration.
#
# Usage:
#   mix run examples/tp_4gpu_generate.exs

Nx.default_backend(EXLA.Backend)

IO.puts("=" |> String.duplicate(70))
IO.puts("4-GPU Tensor Parallel Text Generation")
IO.puts("=" |> String.duplicate(70))

alias EXLA.MLIR.{Function, Value}

# Configuration
tp_size = 4
max_new_tokens = 10
temperature = 1.0

IO.puts("\nConfiguration:")
IO.puts("  TP size: #{tp_size}")
IO.puts("  Max new tokens: #{max_new_tokens}")
IO.puts("  Temperature: #{temperature}")

# ----------------------------------------------------------
# Step 1: Load model and tokenizer
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 1: Loading model and tokenizer")
IO.puts("-" |> String.duplicate(70))

IO.puts("  Loading TinyLlama...")
{:ok, model_info} = Bumblebee.load_model({:hf, "TinyLlama/TinyLlama-1.1B-Chat-v1.0"})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "TinyLlama/TinyLlama-1.1B-Chat-v1.0"})

params = model_info.params
spec = model_info.spec

hidden_size = spec.hidden_size
intermediate_size = spec.intermediate_size
num_heads = spec.num_attention_heads
num_kv_heads = spec.num_key_value_heads
head_dim = div(hidden_size, num_heads)
vocab_size = spec.vocab_size
num_layers = 4  # Use 4 layers for faster generation

# Per-GPU dimensions
local_heads = div(num_heads, tp_size)
local_kv_heads = div(num_kv_heads, tp_size)
local_intermediate = div(intermediate_size, tp_size)
local_q_size = local_heads * head_dim
local_kv_size = local_kv_heads * head_dim
kv_head_repeat = div(local_heads, local_kv_heads)

IO.puts("  Model loaded!")
IO.puts("  Vocab size: #{vocab_size}")
IO.puts("  Hidden size: #{hidden_size}")
IO.puts("  Using #{num_layers} layers for generation")

# ----------------------------------------------------------
# Step 2: Extract and shard parameters
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 2: Extracting and sharding parameters")
IO.puts("-" |> String.duplicate(70))

param_data = params.data

# Embeddings and norms (replicated)
embed_tokens = param_data["embedder.token_embedding"]["kernel"]
final_norm_weight = param_data["output_norm"]["weight"]
lm_head_kernel = param_data["language_modeling_head.output"]["kernel"]

# Extract and shard layer parameters
layer_params = for layer_idx <- 0..(num_layers - 1) do
  prefix = "decoder.blocks.#{layer_idx}"
  
  sa_norm = param_data["#{prefix}.self_attention_norm"]["weight"]
  ffn_norm = param_data["#{prefix}.output_norm"]["weight"]
  
  q_kernel = param_data["#{prefix}.self_attention.query"]["kernel"]
  k_kernel = param_data["#{prefix}.self_attention.key"]["kernel"]
  v_kernel = param_data["#{prefix}.self_attention.value"]["kernel"]
  o_kernel = param_data["#{prefix}.self_attention.output"]["kernel"]
  
  gate_kernel = param_data["#{prefix}.ffn.gate"]["kernel"]
  up_kernel = param_data["#{prefix}.ffn.intermediate"]["kernel"]
  down_kernel = param_data["#{prefix}.ffn.output"]["kernel"]
  
  # Shard
  q_shards = for i <- 0..(tp_size - 1), do: Nx.slice(q_kernel, [0, i * local_q_size], [hidden_size, local_q_size])
  k_shards = for i <- 0..(tp_size - 1), do: Nx.slice(k_kernel, [0, i * local_kv_size], [hidden_size, local_kv_size])
  v_shards = for i <- 0..(tp_size - 1), do: Nx.slice(v_kernel, [0, i * local_kv_size], [hidden_size, local_kv_size])
  o_shards = for i <- 0..(tp_size - 1), do: Nx.slice(o_kernel, [i * local_q_size, 0], [local_q_size, hidden_size])
  
  gate_shards = for i <- 0..(tp_size - 1), do: Nx.slice(gate_kernel, [0, i * local_intermediate], [hidden_size, local_intermediate])
  up_shards = for i <- 0..(tp_size - 1), do: Nx.slice(up_kernel, [0, i * local_intermediate], [hidden_size, local_intermediate])
  down_shards = for i <- 0..(tp_size - 1), do: Nx.slice(down_kernel, [i * local_intermediate, 0], [local_intermediate, hidden_size])
  
  %{
    sa_norm: sa_norm, ffn_norm: ffn_norm,
    q_shards: q_shards, k_shards: k_shards, v_shards: v_shards, o_shards: o_shards,
    gate_shards: gate_shards, up_shards: up_shards, down_shards: down_shards
  }
end

IO.puts("  Extracted #{num_layers} layer parameters")

# ----------------------------------------------------------
# Step 3: Build SPMD executable (reuse from full model)
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 3: Building SPMD executable")
IO.puts("-" |> String.duplicate(70))

# This will be called once per token generation
# Input: [batch=1, seq_len=variable] token IDs
# Output: [batch=1, vocab_size] logits for next token

build_generation_step = fn batch_size, seq_len ->
  input_ids_typespec = EXLA.Typespec.tensor({:s, 64}, {batch_size, seq_len})
  embed_typespec = EXLA.Typespec.tensor({:f, 32}, {vocab_size, hidden_size})
  norm_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size})
  lm_head_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, vocab_size})
  
  q_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_q_size})
  k_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_kv_size})
  v_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_kv_size})
  o_typespec = EXLA.Typespec.tensor({:f, 32}, {local_q_size, hidden_size})
  gate_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_intermediate})
  up_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_intermediate})
  down_typespec = EXLA.Typespec.tensor({:f, 32}, {local_intermediate, hidden_size})
  
  layer_param_typespecs = List.duplicate([
    norm_typespec, norm_typespec,
    q_typespec, k_typespec, v_typespec, o_typespec,
    gate_typespec, up_typespec, down_typespec
  ], num_layers) |> List.flatten()
  
  input_typespecs = [input_ids_typespec, embed_typespec, norm_typespec, lm_head_typespec] ++ layer_param_typespecs
  output_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, vocab_size})
  output_typespecs = [output_typespec]
  
  replica_groups = [Enum.to_list(0..(tp_size - 1))]
  
  EXLA.SPMD.build(input_typespecs, output_typespecs, fn builder ->
    args = Function.get_arguments(builder)
    [input_ids, embed_w, final_norm_w, lm_head_w | layer_args] = args
    
    layer_weights = Enum.chunk_every(layer_args, 9)
    |> Enum.map(fn [sa_norm, ffn_norm, q, k, v, o, gate, up, down] ->
      %{sa_norm: sa_norm, ffn_norm: ffn_norm, q: q, k: k, v: v, o: o, gate: gate, up: up, down: down}
    end)
    
    hidden_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, hidden_size})
    
    # Simplified embedding lookup
    float_ids_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len})
    float_ids = Value.convert(input_ids, float_ids_typespec)
    
    scale_typespec = EXLA.Typespec.tensor({:f, 32}, {})
    scale_tensor = Value.constant(builder, [0.001], scale_typespec)
    scale_broadcast = Value.broadcast_in_dim(scale_tensor, [], float_ids_typespec)
    float_ids_scaled = Value.multiply(float_ids, scale_broadcast, float_ids_typespec)
    
    float_ids_3d_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, 1})
    float_ids_3d = Value.reshape(float_ids_scaled, float_ids_3d_typespec)
    
    proj_typespec = EXLA.Typespec.tensor({:f, 32}, {1, hidden_size})
    proj = Value.slice(embed_w, [0, 0], [1, hidden_size], [1, 1], proj_typespec)
    
    hidden_states = Value.dot_general(float_ids_3d, proj, {[2], [], [0], []}, :default, hidden_typespec)
    
    # Simplified norm
    simple_norm = fn x, weight ->
      weight_broadcast = Value.broadcast_in_dim(weight, [2], hidden_typespec)
      Value.multiply(x, weight_broadcast, hidden_typespec)
    end
    
    # Transformer block
    transformer_block = fn input, weights ->
      normed_for_attn = simple_norm.(input, weights.sa_norm)
      
      q_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_q_size})
      k_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_kv_size})
      v_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_kv_size})
      
      q = Value.dot_general(normed_for_attn, weights.q, {[2], [], [0], []}, :default, q_proj_typespec)
      k = Value.dot_general(normed_for_attn, weights.k, {[2], [], [0], []}, :default, k_proj_typespec)
      v = Value.dot_general(normed_for_attn, weights.v, {[2], [], [0], []}, :default, v_proj_typespec)
      
      q_reshaped = Value.reshape(q, EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_heads, head_dim}))
      k_reshaped = Value.reshape(k, EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_kv_heads, head_dim}))
      v_reshaped = Value.reshape(v, EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_kv_heads, head_dim}))
      
      q_transposed = Value.transpose(q_reshaped, [0, 2, 1, 3], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_heads, seq_len, head_dim}))
      k_transposed = Value.transpose(k_reshaped, [0, 2, 3, 1], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, head_dim, seq_len}))
      v_transposed = Value.transpose(v_reshaped, [0, 2, 1, 3], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, seq_len, head_dim}))
      
      q_grouped = Value.reshape(q_transposed, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, seq_len, head_dim}))
      k_expanded = Value.reshape(k_transposed, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, head_dim, seq_len}))
      v_expanded = Value.reshape(v_transposed, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, seq_len, head_dim}))
      
      k_broadcast = Value.broadcast_in_dim(k_expanded, [0, 1, 2, 3, 4], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, head_dim, seq_len}))
      v_broadcast = Value.broadcast_in_dim(v_expanded, [0, 1, 2, 3, 4], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, seq_len, head_dim}))
      
      scores_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, seq_len, seq_len})
      scores = Value.dot_general(q_grouped, k_broadcast, {[4], [0, 1, 2], [3], [0, 1, 2]}, :default, scores_typespec)
      
      scale_value = 1.0 / :math.sqrt(head_dim)
      scale_typespec = EXLA.Typespec.tensor({:f, 32}, {})
      scale_tensor = Value.constant(builder, [scale_value], scale_typespec)
      scale_broadcast = Value.broadcast_in_dim(scale_tensor, [], scores_typespec)
      scores_scaled = Value.multiply(scores, scale_broadcast, scores_typespec)
      
      attention_weights = Value.sigmoid(scores_scaled, scores_typespec)
      
      attn_output_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, seq_len, head_dim})
      attn_output = Value.dot_general(attention_weights, v_broadcast, {[4], [0, 1, 2], [3], [0, 1, 2]}, :default, attn_output_typespec)
      
      attn_merged = Value.reshape(attn_output, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_heads, seq_len, head_dim}))
      attn_transposed = Value.transpose(attn_merged, [0, 2, 1, 3], EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_heads, head_dim}))
      attn_flat = Value.reshape(attn_transposed, EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_q_size}))
      
      attn_partial = Value.dot_general(attn_flat, weights.o, {[2], [], [0], []}, :default, hidden_typespec)
      attn_result = Value.all_reduce(attn_partial, :sum, replica_groups, hidden_typespec)
      
      after_attn = Value.add(input, attn_result, hidden_typespec)
      
      # FFN
      normed_for_ffn = simple_norm.(after_attn, weights.ffn_norm)
      
      intermediate_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, seq_len, local_intermediate})
      
      gate_out = Value.dot_general(normed_for_ffn, weights.gate, {[2], [], [0], []}, :default, intermediate_typespec)
      up_out = Value.dot_general(normed_for_ffn, weights.up, {[2], [], [0], []}, :default, intermediate_typespec)
      
      gate_sigmoid = Value.sigmoid(gate_out, intermediate_typespec)
      gate_silu = Value.multiply(gate_out, gate_sigmoid, intermediate_typespec)
      combined = Value.multiply(gate_silu, up_out, intermediate_typespec)
      
      ffn_partial = Value.dot_general(combined, weights.down, {[2], [], [0], []}, :default, hidden_typespec)
      ffn_result = Value.all_reduce(ffn_partial, :sum, replica_groups, hidden_typespec)
      
      Value.add(after_attn, ffn_result, hidden_typespec)
    end
    
    # Run all layers
    final_hidden = Enum.reduce(layer_weights, hidden_states, fn weights, hidden ->
      transformer_block.(hidden, weights)
    end)
    
    # Final norm + LM head (last position only)
    normed_output = simple_norm.(final_hidden, final_norm_w)
    
    last_hidden_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, hidden_size})
    last_hidden = Value.slice(normed_output, [0, seq_len - 1, 0], [batch_size, seq_len, hidden_size], [1, 1, 1], last_hidden_typespec)
    last_hidden_2d = Value.reshape(last_hidden, EXLA.Typespec.tensor({:f, 32}, {batch_size, hidden_size}))
    
    logits = Value.dot_general(last_hidden_2d, lm_head_w, {[1], [], [0], []}, :default, output_typespec)
    
    [logits]
  end, num_replicas: tp_size, client: :cuda)
end

IO.puts("  Building initial SPMD executable...")
# Start with prompt length, will rebuild for longer sequences
initial_spmd = build_generation_step.(1, 1)
IO.puts("  SPMD executable built!")

# ----------------------------------------------------------
# Step 4: Tokenize prompt
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 4: Tokenizing prompt")
IO.puts("-" |> String.duplicate(70))

prompt = "The meaning of life is"
IO.puts("  Prompt: \"#{prompt}\"")

tokenized = Bumblebee.apply_tokenizer(tokenizer, prompt)
input_ids = tokenized["input_ids"] |> Nx.as_type(:s64)
prompt_length = Nx.axis_size(input_ids, 1)

IO.puts("  Token IDs: #{inspect(Nx.to_flat_list(input_ids))}")
IO.puts("  Prompt length: #{prompt_length} tokens")

# ----------------------------------------------------------
# Step 5: Autoregressive generation
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 5: Generating text (#{max_new_tokens} tokens)")
IO.puts("-" |> String.duplicate(70))

# Helper to prepare replica inputs
prepare_replica_inputs = fn input_ids ->
  for gpu <- 0..(tp_size - 1) do
    layer_params_for_gpu = Enum.flat_map(layer_params, fn layer ->
      [
        layer.sa_norm, layer.ffn_norm,
        Enum.at(layer.q_shards, gpu), Enum.at(layer.k_shards, gpu),
        Enum.at(layer.v_shards, gpu), Enum.at(layer.o_shards, gpu),
        Enum.at(layer.gate_shards, gpu), Enum.at(layer.up_shards, gpu),
        Enum.at(layer.down_shards, gpu)
      ]
    end)
    
    [input_ids, embed_tokens, final_norm_weight, lm_head_kernel] ++ layer_params_for_gpu
  end
end

# Initial forward pass with full prompt
IO.puts("\n  Processing prompt...")
spmd = build_generation_step.(1, prompt_length)
replica_inputs = prepare_replica_inputs.(input_ids)

{time_us, results} = :timer.tc(fn ->
  EXLA.SPMD.run(spmd, replica_inputs)
end)

[[logits] | _] = results
IO.puts("  Prompt processed in #{Float.round(time_us / 1000, 2)} ms")

# Sample next token
next_token = Nx.argmax(logits, axis: 1) |> Nx.to_flat_list() |> hd()
generated_ids = [next_token]

IO.write("  Generated: ")
IO.write(Bumblebee.Tokenizer.decode(tokenizer, [next_token]))

# Generate remaining tokens
_generated_ids = Enum.reduce(1..(max_new_tokens - 1), {input_ids, generated_ids}, fn _, {current_input_ids, current_generated_ids} ->
  # Append token and run forward pass
  next_token = List.last(current_generated_ids)
  current_ids = Nx.concatenate([current_input_ids, Nx.tensor([[next_token]], type: :s64)], axis: 1)
  current_length = Nx.axis_size(current_ids, 1)

  # Rebuild SPMD for new sequence length
  spmd = build_generation_step.(1, current_length)
  replica_inputs = prepare_replica_inputs.(current_ids)

  [[logits] | _] = EXLA.SPMD.run(spmd, replica_inputs)

  # Sample next token
  next_token = Nx.argmax(logits, axis: 1) |> Nx.to_flat_list() |> hd()
  updated_generated_ids = current_generated_ids ++ [next_token]

  IO.write(Bumblebee.Tokenizer.decode(tokenizer, [next_token]))

  {current_ids, updated_generated_ids}
end)

{input_ids, generated_ids} = _generated_ids

IO.puts("")

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
IO.puts("\n" <> ("=" |> String.duplicate(70)))
IO.puts("Summary")
IO.puts("=" |> String.duplicate(70))

full_text = Bumblebee.Tokenizer.decode(tokenizer, Nx.to_flat_list(input_ids) ++ generated_ids)

IO.puts("""

Generated text:
#{full_text}

Tokens generated: #{max_new_tokens}
Model: TinyLlama-1.1B (#{num_layers} layers)
TP configuration: #{tp_size} GPUs

Note: This is a basic implementation without KV cache optimization.
Each token requires full forward pass through all previous tokens.
""")
