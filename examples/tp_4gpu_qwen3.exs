# 4-GPU Tensor Parallel Text Generation - Qwen3-4B VERSION
#
# This demonstrates production-quality autoregressive generation with Qwen3:
# - Tensor parallelism across 4 GPUs (H100s)
# - KV cache for O(n) generation complexity
# - PROPER embedding lookup (gather)
# - PROPER RMSNorm normalization (reduce)
# - PROPER softmax attention (reduce)
# - PROPER Rotary Position Embeddings (RoPE)
# - **QK Norm** (Qwen3-specific: RMSNorm on Q and K after projection)
# - Qwen3-4B-Instruct model
#
# Usage:
#   LAYERS=36 TOKENS=30 mix run examples/tp_4gpu_qwen3.exs
#
# Environment Variables:
#   LAYERS    - Number of transformer layers (default: 4, full model: 36)
#   TOKENS    - Maximum new tokens to generate (default: 30)
#   TEMP      - Temperature for sampling, 0=greedy (default: 0.7)
#   TOP_K     - Top-k filtering (default: 50)
#   TOP_P     - Nucleus sampling threshold (default: 0.9)
#   MAX_SEQ   - Pre-allocated cache size (default: 128)
#   MEM_FRAC  - GPU memory fraction to allocate (default: auto-calculated)

# =============================================================================
# MEMORY CALCULATION - Must happen BEFORE EXLA client initialization
# =============================================================================
# Formula for LLM inference memory:
#   Model weights = num_params × bytes_per_param
#   KV Cache = 2 × batch × seq_len × layers × kv_heads × head_dim × bytes
#   Total ≈ weights + kv_cache + activations (small)

defmodule MemoryCalculator do
  @moduledoc """
  Calculate GPU memory requirements for LLM inference.

  Memory components:
  1. Model weights (sharded by TP)
  2. KV cache (sharded by TP)
  3. Attention score matrices: batch × heads × seq × seq
  4. FFN intermediate activations: batch × seq × intermediate_size
  5. XLA compilation buffers (~1-2GB per device)
  6. NCCL communication buffers (~512MB per device)
  7. CUDA context overhead (~500MB per device)
  """

  # Qwen3-4B-Instruct specs
  @hidden_size 2560
  @num_layers 36
  @num_heads 32
  @num_kv_heads 8
  @head_dim 128
  @intermediate_size 9728
  @vocab_size 151_936
  @bytes_per_param 4  # f32

  def calculate(opts \\ []) do
    num_layers = Keyword.get(opts, :num_layers, @num_layers)
    max_seq_len = Keyword.get(opts, :max_seq_len, 128)
    batch_size = Keyword.get(opts, :batch_size, 1)
    tp_size = Keyword.get(opts, :tp_size, 4)

    # 1. Model parameters (total, then we'll shard)
    embedding_params = @vocab_size * @hidden_size
    per_layer_params =
      (@hidden_size * @num_heads * @head_dim) +      # Q
      (@hidden_size * @num_kv_heads * @head_dim) +   # K
      (@hidden_size * @num_kv_heads * @head_dim) +   # V
      (@num_heads * @head_dim * @hidden_size) +      # O
      @head_dim + @head_dim +                        # QK norms
      (@hidden_size * @intermediate_size) * 3 +      # FFN (gate, up, down)
      @hidden_size * 2                               # layer norms

    total_params = embedding_params + (per_layer_params * num_layers) + @hidden_size
    model_bytes = total_params * @bytes_per_param
    # Embedding is replicated, attention/FFN sharded by TP
    model_bytes_per_gpu = (embedding_params * @bytes_per_param) +
                          ((total_params - embedding_params) * @bytes_per_param / tp_size)

    # 2. KV cache per GPU (kv_heads are sharded)
    local_kv_heads = @num_kv_heads / tp_size
    kv_cache_bytes_per_gpu = 2 * batch_size * max_seq_len * num_layers *
                             local_kv_heads * @head_dim * @bytes_per_param

    # 3. Attention score matrix: batch × local_heads × seq × seq
    local_heads = @num_heads / tp_size
    attention_scores_bytes = batch_size * local_heads * max_seq_len * max_seq_len * @bytes_per_param

    # 4. FFN intermediate activations per GPU
    local_intermediate = @intermediate_size / tp_size
    ffn_activation_bytes = batch_size * max_seq_len * local_intermediate * @bytes_per_param * 2  # gate + up

    # 5. Hidden state activations (replicated per layer)
    hidden_activation_bytes = batch_size * max_seq_len * @hidden_size * @bytes_per_param

    # Peak activation memory (attention OR FFN, whichever is larger, times some layers in flight)
    peak_activation_bytes = max(attention_scores_bytes, ffn_activation_bytes) +
                            hidden_activation_bytes * 2  # input + output buffers

    # 6. XLA compilation buffers - SPMD compilation uses significant temporary memory
    # Empirically observed: ~4-6GB for complex SPMD graphs with many layers
    xla_compilation_bytes = 5.0 * 1024 * 1024 * 1024

    # 7. NCCL communication buffers (~1 GB for all-reduce with large tensors)
    nccl_buffer_bytes = 1024 * 1024 * 1024

    # 8. CUDA context overhead (~500 MB)
    cuda_overhead_bytes = 500 * 1024 * 1024

    # 9. XLA temporary buffers during execution (intermediate results, fusion buffers)
    # These can be significant for transformer attention operations
    xla_temp_execution_bytes = 2.0 * 1024 * 1024 * 1024

    # Subtotals
    compute_bytes = model_bytes_per_gpu + kv_cache_bytes_per_gpu + peak_activation_bytes
    overhead_bytes = xla_compilation_bytes + nccl_buffer_bytes + cuda_overhead_bytes + xla_temp_execution_bytes

    # Total per GPU
    total_per_gpu_bytes = compute_bytes + overhead_bytes

    # Add 30% safety margin for unforeseen allocations
    total_with_margin = total_per_gpu_bytes * 1.3

    %{
      model_gb: model_bytes / (1024 * 1024 * 1024),
      model_per_gpu_gb: model_bytes_per_gpu / (1024 * 1024 * 1024),
      kv_cache_per_gpu_mb: kv_cache_bytes_per_gpu / (1024 * 1024),
      attention_mb: attention_scores_bytes / (1024 * 1024),
      ffn_activation_mb: ffn_activation_bytes / (1024 * 1024),
      xla_compile_gb: xla_compilation_bytes / (1024 * 1024 * 1024),
      xla_temp_gb: xla_temp_execution_bytes / (1024 * 1024 * 1024),
      nccl_overhead_gb: nccl_buffer_bytes / (1024 * 1024 * 1024),
      cuda_overhead_mb: cuda_overhead_bytes / (1024 * 1024),
      total_per_gpu_gb: total_with_margin / (1024 * 1024 * 1024),
      total_params: total_params
    }
  end

  def recommended_memory_fraction(_opts \\ []) do
    # vLLM-style approach: use most of GPU memory
    # The calculated estimate is informational only - XLA/SPMD needs more buffer
    # than simple calculations predict due to:
    # - SPMD compilation intermediate buffers
    # - XLA fusion and optimization passes
    # - Model loading before sharding takes effect
    #
    # Default to 85% (vLLM uses 90%, we use 85% for safety)
    0.85
  end
end

# Get config for memory calculation
num_layers_for_mem = String.to_integer(System.get_env("LAYERS", "36"))
max_seq_for_mem = String.to_integer(System.get_env("MAX_SEQ", "128"))
batch_for_mem = 1
tp_size_for_mem = 4

# Calculate memory requirements
mem_info = MemoryCalculator.calculate(
  num_layers: num_layers_for_mem,
  max_seq_len: max_seq_for_mem,
  batch_size: batch_for_mem,
  tp_size: tp_size_for_mem
)

# Allow override via environment variable
memory_fraction = case System.get_env("MEM_FRAC") do
  nil ->
    MemoryCalculator.recommended_memory_fraction(
      num_layers: num_layers_for_mem,
      max_seq_len: max_seq_for_mem,
      batch_size: batch_for_mem,
      tp_size: tp_size_for_mem
    )
  frac_str ->
    {f, _} = Float.parse(frac_str)
    f
end

IO.puts("Memory calculation (per GPU with TP=#{tp_size_for_mem}):")
IO.puts("  Model weights:")
IO.puts("    Total params: #{Float.round(mem_info.total_params / 1_000_000, 1)}M (~4B)")
IO.puts("    Total size: #{Float.round(mem_info.model_gb, 2)} GB")
IO.puts("    Per GPU (sharded): #{Float.round(mem_info.model_per_gpu_gb, 2)} GB")
IO.puts("  Runtime buffers:")
IO.puts("    KV cache: #{Float.round(mem_info.kv_cache_per_gpu_mb, 1)} MB")
IO.puts("    Attention scores: #{Float.round(mem_info.attention_mb, 1)} MB")
IO.puts("    FFN activations: #{Float.round(mem_info.ffn_activation_mb, 1)} MB")
IO.puts("  XLA/CUDA Overhead:")
IO.puts("    XLA compilation: #{Float.round(mem_info.xla_compile_gb, 1)} GB")
IO.puts("    XLA temp buffers: #{Float.round(mem_info.xla_temp_gb, 1)} GB")
IO.puts("    NCCL buffers: #{Float.round(mem_info.nccl_overhead_gb, 1)} GB")
IO.puts("    CUDA context: #{Float.round(mem_info.cuda_overhead_mb, 0)} MB")
IO.puts("  Total per GPU (with 30% margin): #{Float.round(mem_info.total_per_gpu_gb, 1)} GB")
IO.puts("  Memory fraction: #{Float.round(memory_fraction, 2)} (of ~94 GB per H100)")

# Configure EXLA client BEFORE any tensor operations
Application.put_env(:exla, :clients,
  cuda: [platform: :cuda, memory_fraction: memory_fraction, preallocate: true]
)

Nx.default_backend(EXLA.Backend)

IO.puts("=" |> String.duplicate(70))
IO.puts("4-GPU TP Generation - Qwen3-4B-Instruct")
IO.puts("=" |> String.duplicate(70))

alias EXLA.MLIR.{Function, Value, Region}

# Configuration
tp_size = 4
max_new_tokens = String.to_integer(System.get_env("TOKENS", "30"))
# Parse float that handles both "0" and "0.7" formats
parse_float = fn str ->
  case Float.parse(str) do
    {f, _} -> f
    :error -> String.to_float(str)
  end
end
temperature = parse_float.(System.get_env("TEMP", "0.7"))
top_k = String.to_integer(System.get_env("TOP_K", "50"))
top_p = parse_float.(System.get_env("TOP_P", "0.9"))
use_rope = true  # Always use RoPE for Qwen3

# Phase 3: Pre-allocated KV cache configuration
max_seq_len = String.to_integer(System.get_env("MAX_SEQ", "128"))

# Batch size (only 1 supported for simplicity)
batch_size = 1

# Phase 5: Streaming configuration
stream_mode = System.get_env("STREAM", "true") == "true"

# Model size - Qwen3-4B has 36 layers
num_layers = String.to_integer(System.get_env("LAYERS", "4"))  # Use fewer layers for demo

IO.puts("\nConfiguration:")
IO.puts("  TP size: #{tp_size}")
IO.puts("  Max new tokens: #{max_new_tokens}")
IO.puts("  Temperature: #{temperature}")
IO.puts("  Top-k: #{top_k}")
IO.puts("  Top-p: #{top_p}")
IO.puts("  Max sequence length: #{max_seq_len}")
IO.puts("  Batch size: #{batch_size}")
IO.puts("  Streaming: #{stream_mode}")

# ----------------------------------------------------------
# Step 1: Load model and tokenizer
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 1: Loading model and tokenizer")
IO.puts("-" |> String.duplicate(70))

model_id = "Qwen/Qwen3-4B-Instruct-2507"
IO.puts("  Loading #{model_id}...")
# Load in f32 (not bf16) to match SPMD typespecs
{:ok, %{params: params, spec: spec}} = Bumblebee.load_model({:hf, model_id})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, model_id})

vocab_size = spec.vocab_size
hidden_size = spec.hidden_size
intermediate_size = spec.intermediate_size
num_heads = spec.num_attention_heads
num_kv_heads = spec.num_key_value_heads
head_dim = spec.attention_head_size  # Qwen3 has explicit head_dim
rms_norm_eps = spec.layer_norm_epsilon
rope_theta = spec.rotary_embedding_base || 5_000_000.0
use_qk_norm = spec.use_qk_norm  # Qwen3-specific
tie_word_embeddings = spec.tie_word_embeddings  # Qwen3 ties embeddings

# Tensor parallelism configuration
# For Qwen3-4B: 20 heads, 4 KV heads -> 5 heads per GPU, 1 KV head per GPU
local_heads = div(num_heads, tp_size)
local_kv_heads = div(num_kv_heads, tp_size)
kv_head_repeat = div(local_heads, local_kv_heads)
local_q_size = local_heads * head_dim
local_kv_size = local_kv_heads * head_dim
local_intermediate = div(intermediate_size, tp_size)

IO.puts("  Model loaded!")
IO.puts("  Vocab size: #{vocab_size}")
IO.puts("  Hidden size: #{hidden_size}")
IO.puts("  Num heads: #{num_heads}, KV heads: #{num_kv_heads}")
IO.puts("  Head dim: #{head_dim}")
IO.puts("  Using #{num_layers} layers for generation (full model: 36)")
IO.puts("  RMS norm epsilon: #{rms_norm_eps}")
IO.puts("  RoPE theta: #{rope_theta}")
IO.puts("  QK Norm: #{use_qk_norm}")
IO.puts("  Tied embeddings: #{tie_word_embeddings}")

# Precompute RoPE sin/cos embeddings
# RoPE formula: inv_freq = 1.0 / (theta^(2i/dim)) for i in 0..dim/2-1
compute_rope_embeddings = fn max_positions ->
  # Compute inverse frequencies
  half_dim = div(head_dim, 2)
  inv_freq = for i <- 0..(half_dim - 1) do
    1.0 / :math.pow(rope_theta, 2 * i / head_dim)
  end
  inv_freq_tensor = Nx.tensor(inv_freq, type: :f32)

  # Positions: [0, 1, 2, ..., max_positions-1]
  positions = Nx.iota({max_positions}, type: :f32)

  # Outer product: positions x inv_freq -> [max_positions, half_dim]
  freqs = Nx.outer(positions, inv_freq_tensor)

  # Compute cos/sin of frequencies
  cos_freqs = Nx.cos(freqs)  # [max_positions, half_dim]
  sin_freqs = Nx.sin(freqs)  # [max_positions, half_dim]

  # Standard RoPE: concatenate [cos_freqs, cos_freqs] to get [max_positions, head_dim]
  # The first half and second half have the same values
  cos_embed = Nx.concatenate([cos_freqs, cos_freqs], axis: 1)  # [max_positions, head_dim]
  sin_embed = Nx.concatenate([sin_freqs, sin_freqs], axis: 1)  # [max_positions, head_dim]

  {cos_embed, sin_embed}
end

# Sampling function with temperature, top-k, and top-p
sample_token = fn logits, key, temp, k, p ->
  # logits shape: {batch_size, vocab_size} - we assume batch_size=1
  logits_1d = Nx.squeeze(logits, axes: [0])  # {vocab_size}

  # Apply temperature
  scaled_logits = if temp > 0 do
    Nx.divide(logits_1d, temp)
  else
    logits_1d
  end

  # Get top-k indices and values
  {top_values, top_indices} = Nx.top_k(scaled_logits, k: k)

  # Apply softmax to get probabilities
  max_val = Nx.reduce_max(top_values)
  shifted = Nx.subtract(top_values, max_val)
  exp_vals = Nx.exp(shifted)
  sum_exp = Nx.sum(exp_vals)
  probs = Nx.divide(exp_vals, sum_exp)

  # Apply top-p (nucleus) filtering
  sorted_probs = probs  # already sorted by top_k
  cumsum = Nx.cumulative_sum(sorted_probs)

  # Find cutoff index where cumsum > p
  mask = Nx.less_equal(cumsum, p)
  # Always keep at least one token
  mask = Nx.put_slice(mask, [0], Nx.tensor([1], type: :u8))

  # Zero out probabilities beyond top-p
  filtered_probs = Nx.select(mask, probs, Nx.tensor(0.0))

  # Renormalize
  filtered_sum = Nx.sum(filtered_probs)
  final_probs = Nx.divide(filtered_probs, filtered_sum)

  # Sample from categorical distribution
  {rand_val, new_key} = Nx.Random.uniform(key, 0.0, 1.0, shape: {}, type: :f32)
  rand_val = Nx.to_number(rand_val)

  # Find which bucket the random value falls into
  cumsum_final = Nx.cumulative_sum(final_probs)
  cumsum_list = Nx.to_flat_list(cumsum_final)
  indices_list = Nx.to_flat_list(top_indices)

  sampled_idx = Enum.zip(cumsum_list, indices_list)
  |> Enum.find_value(fn {cum, idx} ->
    if rand_val <= cum, do: idx, else: nil
  end)

  # Fallback to first token if nothing found (shouldn't happen)
  sampled_idx = sampled_idx || hd(indices_list)

  {trunc(sampled_idx), new_key}
end

# ----------------------------------------------------------
# Step 2: Extract and shard parameters
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 2: Extracting and sharding parameters")
IO.puts("-" |> String.duplicate(70))

param_data = params.data

# Extract global parameters (not sharded)
embed_tokens = param_data["embedder.token_embedding"]["kernel"]
final_norm_weight = param_data["output_norm"]["weight"]
# Qwen3 ties LM head to embeddings - need to transpose embed_tokens for LM head
# For tied embeddings, lm_head_kernel = embed_tokens^T (logits = hidden @ embed^T)
lm_head_kernel = if tie_word_embeddings do
  embed_tokens  # Will be transposed during computation
else
  param_data["language_modeling_head.output"]["kernel"]
end

# Extract and shard layer parameters
layer_params = for layer_idx <- 0..(num_layers - 1) do
  layer_prefix = "decoder.blocks.#{layer_idx}"

  # Norms (replicated)
  sa_norm = param_data["#{layer_prefix}.self_attention_norm"]["weight"]
  ffn_norm = param_data["#{layer_prefix}.output_norm"]["weight"]

  # QK Norm weights (Qwen3-specific) - shape {head_dim}
  q_norm = if use_qk_norm do
    param_data["#{layer_prefix}.self_attention.query_norm"]["weight"]
  else
    nil
  end
  k_norm = if use_qk_norm do
    param_data["#{layer_prefix}.self_attention.key_norm"]["weight"]
  else
    nil
  end

  # Attention weights
  q_kernel = param_data["#{layer_prefix}.self_attention.query"]["kernel"]
  k_kernel = param_data["#{layer_prefix}.self_attention.key"]["kernel"]
  v_kernel = param_data["#{layer_prefix}.self_attention.value"]["kernel"]
  o_kernel = param_data["#{layer_prefix}.self_attention.output"]["kernel"]

  # Shard attention weights (column-parallel for Q/K/V, row-parallel for O)
  q_shards = for i <- 0..(tp_size - 1), do: Nx.slice(q_kernel, [0, i * local_q_size], [hidden_size, local_q_size])
  k_shards = for i <- 0..(tp_size - 1), do: Nx.slice(k_kernel, [0, i * local_kv_size], [hidden_size, local_kv_size])
  v_shards = for i <- 0..(tp_size - 1), do: Nx.slice(v_kernel, [0, i * local_kv_size], [hidden_size, local_kv_size])
  o_shards = for i <- 0..(tp_size - 1), do: Nx.slice(o_kernel, [i * local_q_size, 0], [local_q_size, hidden_size])

  # FFN weights
  gate_kernel = param_data["#{layer_prefix}.ffn.gate"]["kernel"]
  up_kernel = param_data["#{layer_prefix}.ffn.intermediate"]["kernel"]
  down_kernel = param_data["#{layer_prefix}.ffn.output"]["kernel"]

  # Shard FFN weights (column-parallel for gate/up, row-parallel for down)
  gate_shards = for i <- 0..(tp_size - 1), do: Nx.slice(gate_kernel, [0, i * local_intermediate], [hidden_size, local_intermediate])
  up_shards = for i <- 0..(tp_size - 1), do: Nx.slice(up_kernel, [0, i * local_intermediate], [hidden_size, local_intermediate])
  down_shards = for i <- 0..(tp_size - 1), do: Nx.slice(down_kernel, [i * local_intermediate, 0], [local_intermediate, hidden_size])

  %{
    sa_norm: sa_norm, ffn_norm: ffn_norm,
    q_norm: q_norm, k_norm: k_norm,  # QK norms (Qwen3-specific)
    q_shards: q_shards, k_shards: k_shards, v_shards: v_shards, o_shards: o_shards,
    gate_shards: gate_shards, up_shards: up_shards, down_shards: down_shards
  }
end

IO.puts("  Extracted #{num_layers} layer parameters")

# ----------------------------------------------------------
# Step 3: Build SPMD executables with KV cache
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 3: Building SPMD executables with proper ops")
IO.puts("-" |> String.duplicate(70))

# Prefill phase: Process prompt and initialize KV cache
IO.puts("  Building prefill SPMD (with proper gather, RMSNorm, softmax, RoPE)...")

# Phase 3: Prefill SPMD with pre-allocated KV cache (outputs max_seq_len sized caches)
build_prefill_spmd = fn batch_size, prompt_len, max_seq_len_local ->
  input_ids_typespec = EXLA.Typespec.tensor({:s, 32}, {batch_size, prompt_len})
  embed_typespec = EXLA.Typespec.tensor({:f, 32}, {vocab_size, hidden_size})
  norm_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size})
  # For Qwen3: lm_head uses transposed embeddings (tied), shape is {vocab_size, hidden_size}
  lm_head_typespec = EXLA.Typespec.tensor({:f, 32}, {vocab_size, hidden_size})

  # RoPE embeddings: [prompt_len, head_dim]
  rope_cos_typespec = EXLA.Typespec.tensor({:f, 32}, {prompt_len, head_dim})
  rope_sin_typespec = EXLA.Typespec.tensor({:f, 32}, {prompt_len, head_dim})

  q_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_q_size})
  k_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_kv_size})
  v_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_kv_size})
  o_typespec = EXLA.Typespec.tensor({:f, 32}, {local_q_size, hidden_size})
  gate_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_intermediate})
  up_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_intermediate})
  down_typespec = EXLA.Typespec.tensor({:f, 32}, {local_intermediate, hidden_size})

  # QK norm weights (Qwen3-specific) - shape {head_dim}
  qk_norm_typespec = EXLA.Typespec.tensor({:f, 32}, {head_dim})

  layer_param_typespecs = List.duplicate([
    norm_typespec, norm_typespec,
    qk_norm_typespec, qk_norm_typespec,  # q_norm, k_norm
    q_typespec, k_typespec, v_typespec, o_typespec,
    gate_typespec, up_typespec, down_typespec
  ], num_layers) |> List.flatten()

  input_typespecs = [input_ids_typespec, embed_typespec, norm_typespec, lm_head_typespec, rope_cos_typespec, rope_sin_typespec] ++ layer_param_typespecs

  # Outputs: logits + K/V caches for each layer (pre-allocated to max_seq_len)
  output_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, vocab_size})
  # Phase 3: Caches are pre-allocated to max_seq_len
  k_cache_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, max_seq_len_local, head_dim})
  v_cache_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, max_seq_len_local, head_dim})

  # One K and V cache per layer
  cache_typespecs = List.duplicate([k_cache_typespec, v_cache_typespec], num_layers) |> List.flatten()
  output_typespecs = [output_typespec] ++ cache_typespecs

  replica_groups = [Enum.to_list(0..(tp_size - 1))]

  EXLA.SPMD.build(input_typespecs, output_typespecs, fn builder ->
    args = Function.get_arguments(builder)
    [input_ids, embed_w, final_norm_w, lm_head_w, rope_cos, rope_sin | layer_args] = args

    layer_weights = Enum.chunk_every(layer_args, 11)
    |> Enum.map(fn [sa_norm, ffn_norm, q_norm, k_norm, q, k, v, o, gate, up, down] ->
      %{sa_norm: sa_norm, ffn_norm: ffn_norm, q_norm: q_norm, k_norm: k_norm,
        q: q, k: k, v: v, o: o, gate: gate, up: up, down: down}
    end)

    hidden_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, hidden_size})

    # RoPE application function
    # Formula: rotated = x * cos + rotate_half(x) * sin
    # Where rotate_half([x0, x1, x2, x3, ...]) = [-x1, x0, -x3, x2, ...]
    apply_rope = fn x, cos_embed, sin_embed, x_typespec ->
      # x shape: [batch, num_heads, seq_len, head_dim]
      # cos/sin shape: [seq_len, head_dim]

      # Broadcast cos/sin to match x shape
      {batch_local, num_heads_local, seq_len_local, _head_dim_local} = x_typespec.shape
      cos_4d_typespec = EXLA.Typespec.tensor({:f, 32}, {1, 1, seq_len_local, head_dim})
      sin_4d_typespec = EXLA.Typespec.tensor({:f, 32}, {1, 1, seq_len_local, head_dim})

      cos_4d = Value.reshape(cos_embed, cos_4d_typespec)
      sin_4d = Value.reshape(sin_embed, sin_4d_typespec)

      cos_broadcast = Value.broadcast_in_dim(cos_4d, [0, 1, 2, 3], x_typespec)
      sin_broadcast = Value.broadcast_in_dim(sin_4d, [0, 1, 2, 3], x_typespec)

      # Create rotate_half(x)
      # Split x into first half and second half along head_dim
      half_dim = div(head_dim, 2)
      first_half_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_local, num_heads_local, seq_len_local, half_dim})
      second_half_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_local, num_heads_local, seq_len_local, half_dim})

      # Slice first half: [0:batch, 0:heads, 0:seq, 0:half_dim]
      x_first = Value.slice(x, [0, 0, 0, 0], [batch_local, num_heads_local, seq_len_local, half_dim], [1, 1, 1, 1], first_half_typespec)
      # Slice second half: [0:batch, 0:heads, 0:seq, half_dim:head_dim]
      x_second = Value.slice(x, [0, 0, 0, half_dim], [batch_local, num_heads_local, seq_len_local, head_dim], [1, 1, 1, 1], second_half_typespec)

      # Negate second half
      neg_one = Value.constant(builder, [-1.0], EXLA.Typespec.tensor({:f, 32}, {}))
      neg_one_broadcast = Value.broadcast_in_dim(neg_one, [], second_half_typespec)
      x_second_neg = Value.multiply(x_second, neg_one_broadcast, second_half_typespec)

      # rotate_half = concat(-x_second, x_first)
      x_rotated = Value.concatenate([x_second_neg, x_first], 3, x_typespec)

      # Apply rotation: x * cos + rotate_half(x) * sin
      x_cos = Value.multiply(x, cos_broadcast, x_typespec)
      x_rot_sin = Value.multiply(x_rotated, sin_broadcast, x_typespec)
      Value.add(x_cos, x_rot_sin, x_typespec)
    end

    # PROPER EMBEDDING LOOKUP using gather
    # Gather signature: gather(source, indices, index_vector_dim, slice_sizes, offset_dims, collapsed_slice_dims, start_index_map, typespec)
    # For embedding lookup: source=[vocab_size, hidden_size], indices=[batch, seq_len]
    # We want output=[batch, seq_len, hidden_size]
    hidden_states = Value.gather(
      embed_w,                    # source: [vocab_size, hidden_size]
      input_ids,                  # indices: [batch, prompt_len]
      2,                          # index_vector_dim (outside indices dims, so indices are scalars)
      [1, hidden_size],          # slice_sizes: take [1, hidden_size] slice for each index
      [2],                        # offset_dims: hidden_size goes to output dim 2 (after batch, seq)
      [0],                        # collapsed_slice_dims: collapse vocab dim (dim 0 of slice)
      [0],                        # start_index_map: index maps to vocab dim (dim 0 of source)
      hidden_typespec
    )

    # PROPER RMSNorm implementation
    rms_norm = fn x, weight, typespec ->
      # RMSNorm: x * weight / sqrt(mean(x^2) + eps)

      # Step 1: x^2
      x_squared_typespec = typespec
      x_squared = Value.multiply(x, x, x_squared_typespec)

      # Step 2: Reduce sum over hidden dimension to get sum(x^2)
      # Need to create a reduction region
      scalar_typespec = EXLA.Typespec.tensor({:f, 32}, {})
      reduce_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len})

      {region, [lhs, rhs]} = Function.push_region(builder, [scalar_typespec, scalar_typespec])
      sum_result = Value.add(lhs, rhs, scalar_typespec)
      Value.return(builder, [sum_result])
      Function.pop_region(builder)

      # Initial value for sum
      zero = Value.constant(builder, [0.0], scalar_typespec)

      # Reduce over dimension 2 (hidden_size dimension)
      [sum_squared] = Value.reduce(region, [zero], [x_squared], [2], [reduce_typespec])

      # Step 3: mean = sum / hidden_size
      hidden_size_constant = Value.constant(builder, [hidden_size * 1.0], scalar_typespec)
      hidden_size_broadcast = Value.broadcast_in_dim(hidden_size_constant, [], reduce_typespec)
      mean_squared = Value.divide(sum_squared, hidden_size_broadcast, reduce_typespec)

      # Step 4: Add epsilon
      epsilon_constant = Value.constant(builder, [rms_norm_eps], scalar_typespec)
      epsilon_broadcast = Value.broadcast_in_dim(epsilon_constant, [], reduce_typespec)
      mean_squared_eps = Value.add(mean_squared, epsilon_broadcast, reduce_typespec)

      # Step 5: rsqrt = 1 / sqrt(mean_squared + eps)
      rsqrt = Value.rsqrt(mean_squared_eps, reduce_typespec)

      # Step 6: Broadcast rsqrt back to full shape
      rsqrt_3d_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, 1})
      rsqrt_3d = Value.reshape(rsqrt, rsqrt_3d_typespec)
      rsqrt_broadcast = Value.broadcast_in_dim(rsqrt_3d, [0, 1, 2], typespec)

      # Step 7: x * rsqrt
      normalized = Value.multiply(x, rsqrt_broadcast, typespec)

      # Step 8: Multiply by weight
      weight_broadcast = Value.broadcast_in_dim(weight, [2], typespec)
      Value.multiply(normalized, weight_broadcast, typespec)
    end

    # PROPER SOFTMAX implementation
    softmax = fn scores, scores_typespec, seq_dim ->
      # Softmax: exp(x - max(x)) / sum(exp(x - max(x)))

      # Get shape info - underscore unused vars to avoid warnings
      {_batch_size, _local_kv_heads, _kv_head_repeat, _query_len, _key_len} = scores_typespec.shape

      # Step 1: Find max along seq_dim
      scalar_typespec = EXLA.Typespec.tensor({:f, 32}, {})
      reduce_shape = Tuple.delete_at(scores_typespec.shape, seq_dim)
      reduce_typespec = EXLA.Typespec.tensor({:f, 32}, reduce_shape)

      {region, [lhs, rhs]} = Function.push_region(builder, [scalar_typespec, scalar_typespec])
      max_result = Value.max(lhs, rhs, scalar_typespec)
      Value.return(builder, [max_result])
      Function.pop_region(builder)

      # Initial value: negative infinity
      neg_inf = Value.constant(builder, [-1.0e9], scalar_typespec)

      # Reduce to find max
      [max_scores] = Value.reduce(region, [neg_inf], [scores], [seq_dim], [reduce_typespec])

      # Step 2: Broadcast max back to original shape
      max_expanded_typespec = EXLA.Typespec.tensor({:f, 32}, Tuple.insert_at(reduce_shape, seq_dim, 1))
      max_expanded = Value.reshape(max_scores, max_expanded_typespec)
      max_broadcast = Value.broadcast_in_dim(max_expanded, [0, 1, 2, 3, 4], scores_typespec)

      # Step 3: Subtract max (for numerical stability)
      scores_shifted = Value.subtract(scores, max_broadcast, scores_typespec)

      # Step 4: Exp
      scores_exp = Value.exp(scores_shifted, scores_typespec)

      # Step 5: Sum exp scores
      {region, [lhs, rhs]} = Function.push_region(builder, [scalar_typespec, scalar_typespec])
      sum_result = Value.add(lhs, rhs, scalar_typespec)
      Value.return(builder, [sum_result])
      Function.pop_region(builder)

      zero = Value.constant(builder, [0.0], scalar_typespec)
      [sum_exp] = Value.reduce(region, [zero], [scores_exp], [seq_dim], [reduce_typespec])

      # Step 6: Broadcast sum back
      sum_expanded = Value.reshape(sum_exp, max_expanded_typespec)
      sum_broadcast = Value.broadcast_in_dim(sum_expanded, [0, 1, 2, 3, 4], scores_typespec)

      # Step 7: Divide to get softmax
      Value.divide(scores_exp, sum_broadcast, scores_typespec)
    end

    # Process layers and collect K/V caches
    {final_hidden, kv_caches} = Enum.reduce(layer_weights, {hidden_states, []}, fn weights, {hidden, caches} ->
      # Self-attention with cache output
      normed_for_attn = rms_norm.(hidden, weights.sa_norm, hidden_typespec)

      q_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, local_q_size})
      k_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, local_kv_size})
      v_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, local_kv_size})

      q = Value.dot_general(normed_for_attn, weights.q, {[2], [], [0], []}, :default, q_proj_typespec)
      k = Value.dot_general(normed_for_attn, weights.k, {[2], [], [0], []}, :default, k_proj_typespec)
      v = Value.dot_general(normed_for_attn, weights.v, {[2], [], [0], []}, :default, v_proj_typespec)

      # Reshape for multi-head attention
      q_reshaped = Value.reshape(q, EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, local_heads, head_dim}))
      k_reshaped = Value.reshape(k, EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, local_kv_heads, head_dim}))
      v_reshaped = Value.reshape(v, EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, local_kv_heads, head_dim}))

      # Transpose to [batch, heads, seq_len, head_dim]
      q_transposed_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_heads, prompt_len, head_dim})
      k_transposed_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, prompt_len, head_dim})
      v_transposed_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, prompt_len, head_dim})

      q_transposed_raw = Value.transpose(q_reshaped, [0, 2, 1, 3], q_transposed_typespec)
      k_transposed_raw = Value.transpose(k_reshaped, [0, 2, 1, 3], k_transposed_typespec)
      v_transposed = Value.transpose(v_reshaped, [0, 2, 1, 3], v_transposed_typespec)

      # QK Norm (Qwen3-specific): RMSNorm on Q and K per head before RoPE
      # Input shape: [batch, heads, seq_len, head_dim]
      # Weight shape: {head_dim}
      # Normalize along head_dim (axis 3)
      qk_rms_norm = fn x, weight, typespec ->
        {batch_local, num_heads_local, seq_len_local, head_dim_local} = typespec.shape

        # x^2
        x_squared = Value.multiply(x, x, typespec)

        # Reduce over head_dim to get mean(x^2)
        scalar_typespec = EXLA.Typespec.tensor({:f, 32}, {})
        reduce_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_local, num_heads_local, seq_len_local})

        {region, [lhs, rhs]} = Function.push_region(builder, [scalar_typespec, scalar_typespec])
        sum_result = Value.add(lhs, rhs, scalar_typespec)
        Value.return(builder, [sum_result])
        Function.pop_region(builder)

        zero = Value.constant(builder, [0.0], scalar_typespec)
        [sum_squared] = Value.reduce(region, [zero], [x_squared], [3], [reduce_typespec])

        # mean = sum / head_dim
        head_dim_constant = Value.constant(builder, [head_dim_local * 1.0], scalar_typespec)
        head_dim_broadcast = Value.broadcast_in_dim(head_dim_constant, [], reduce_typespec)
        mean_squared = Value.divide(sum_squared, head_dim_broadcast, reduce_typespec)

        # Add epsilon
        epsilon_constant = Value.constant(builder, [rms_norm_eps], scalar_typespec)
        epsilon_broadcast = Value.broadcast_in_dim(epsilon_constant, [], reduce_typespec)
        mean_squared_eps = Value.add(mean_squared, epsilon_broadcast, reduce_typespec)

        # rsqrt
        rsqrt = Value.rsqrt(mean_squared_eps, reduce_typespec)

        # Broadcast rsqrt back to 4D
        rsqrt_4d_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_local, num_heads_local, seq_len_local, 1})
        rsqrt_4d = Value.reshape(rsqrt, rsqrt_4d_typespec)
        rsqrt_broadcast = Value.broadcast_in_dim(rsqrt_4d, [0, 1, 2, 3], typespec)

        # Normalize
        normalized = Value.multiply(x, rsqrt_broadcast, typespec)

        # Apply weight (broadcast from {head_dim} to full shape)
        weight_broadcast = Value.broadcast_in_dim(weight, [3], typespec)
        Value.multiply(normalized, weight_broadcast, typespec)
      end

      # Apply QK norm (Qwen3-specific)
      q_normed = qk_rms_norm.(q_transposed_raw, weights.q_norm, q_transposed_typespec)
      k_normed = qk_rms_norm.(k_transposed_raw, weights.k_norm, k_transposed_typespec)

      # Apply RoPE to Q and K (conditionally)
      {q_transposed, k_transposed} = if use_rope do
        {apply_rope.(q_normed, rope_cos, rope_sin, q_transposed_typespec),
         apply_rope.(k_normed, rope_cos, rope_sin, k_transposed_typespec)}
      else
        {q_normed, k_normed}
      end

      # Phase 3: Create pre-allocated caches and use dynamic_update_slice
      # K and V have shape {batch, local_kv_heads, prompt_len, head_dim}
      # We need to write them into caches of shape {batch, local_kv_heads, max_seq_len_local, head_dim}
      k_cache_full_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, max_seq_len_local, head_dim})
      v_cache_full_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, max_seq_len_local, head_dim})

      # Initialize full caches with zeros
      zero_scalar = Value.constant(builder, [0.0], EXLA.Typespec.tensor({:f, 32}, {}))
      k_cache_zeros = Value.broadcast_in_dim(zero_scalar, [], k_cache_full_typespec)
      v_cache_zeros = Value.broadcast_in_dim(zero_scalar, [], v_cache_full_typespec)

      # Write prefill K/V at position 0 using dynamic_update_slice
      # Start indices: [0, 0, 0, 0] - write at the beginning
      zero_idx = Value.constant(builder, [0], EXLA.Typespec.tensor({:s, 32}, {}))
      k_cache = Value.dynamic_update_slice(k_cache_zeros, k_transposed, [zero_idx, zero_idx, zero_idx, zero_idx], k_cache_full_typespec)
      v_cache = Value.dynamic_update_slice(v_cache_zeros, v_transposed, [zero_idx, zero_idx, zero_idx, zero_idx], v_cache_full_typespec)

      # Grouped query attention
      k_for_attn = Value.transpose(k_transposed, [0, 1, 3, 2], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, head_dim, prompt_len}))

      q_grouped = Value.reshape(q_transposed, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, prompt_len, head_dim}))
      k_expanded = Value.reshape(k_for_attn, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, head_dim, prompt_len}))
      v_expanded = Value.reshape(v_transposed, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, prompt_len, head_dim}))

      k_broadcast = Value.broadcast_in_dim(k_expanded, [0, 1, 2, 3, 4], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, head_dim, prompt_len}))
      v_broadcast = Value.broadcast_in_dim(v_expanded, [0, 1, 2, 3, 4], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, prompt_len, head_dim}))

      scores_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, prompt_len, prompt_len})
      scores = Value.dot_general(q_grouped, k_broadcast, {[4], [0, 1, 2], [3], [0, 1, 2]}, :default, scores_typespec)

      scale_value = 1.0 / :math.sqrt(head_dim)
      scale_typespec = EXLA.Typespec.tensor({:f, 32}, {})
      scale_tensor = Value.constant(builder, [scale_value], scale_typespec)
      scale_broadcast = Value.broadcast_in_dim(scale_tensor, [], scores_typespec)
      scores_scaled = Value.multiply(scores, scale_broadcast, scores_typespec)

      # CAUSAL MASKING: Create lower triangular mask
      # row_indices >= col_indices means we can attend (lower triangular including diagonal)
      row_typespec = EXLA.Typespec.tensor({:s, 32}, {prompt_len, 1})
      col_typespec = EXLA.Typespec.tensor({:s, 32}, {1, prompt_len})
      row_indices = Value.iota(builder, 0, row_typespec)
      col_indices = Value.iota(builder, 1, col_typespec)

      # Broadcast to [prompt_len, prompt_len]
      mask_2d_typespec = EXLA.Typespec.tensor({:s, 32}, {prompt_len, prompt_len})
      row_broadcast = Value.broadcast_in_dim(row_indices, [0, 1], mask_2d_typespec)
      col_broadcast = Value.broadcast_in_dim(col_indices, [0, 1], mask_2d_typespec)

      # Create causal mask: 1 where row >= col, 0 otherwise
      causal_mask_int = Value.greater_equal(row_broadcast, col_broadcast, EXLA.Typespec.tensor({:pred, 8}, {prompt_len, prompt_len}))

      # Convert to float and create mask values: 0 for attend, -inf for don't attend
      neg_inf_scalar = Value.constant(builder, [-1.0e9], scale_typespec)
      zero_scalar = Value.constant(builder, [0.0], scale_typespec)

      mask_float_typespec = EXLA.Typespec.tensor({:f, 32}, {prompt_len, prompt_len})
      neg_inf_mask = Value.broadcast_in_dim(neg_inf_scalar, [], mask_float_typespec)
      zero_mask = Value.broadcast_in_dim(zero_scalar, [], mask_float_typespec)

      # Select: where mask is true (can attend), use 0; else use -inf
      causal_mask_2d = Value.select(causal_mask_int, zero_mask, neg_inf_mask, mask_float_typespec)

      # Broadcast mask to full attention shape [batch, heads, kv_repeat, prompt_len, prompt_len]
      mask_5d_typespec = EXLA.Typespec.tensor({:f, 32}, {1, 1, 1, prompt_len, prompt_len})
      causal_mask_5d = Value.reshape(causal_mask_2d, mask_5d_typespec)
      causal_mask_broadcast = Value.broadcast_in_dim(causal_mask_5d, [0, 1, 2, 3, 4], scores_typespec)

      # Apply mask to scores
      scores_masked = Value.add(scores_scaled, causal_mask_broadcast, scores_typespec)

      # Apply PROPER softmax
      attention_weights = softmax.(scores_masked, scores_typespec, 4)

      attn_output_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, prompt_len, head_dim})
      attn_output = Value.dot_general(attention_weights, v_broadcast, {[4], [0, 1, 2], [3], [0, 1, 2]}, :default, attn_output_typespec)

      attn_merged = Value.reshape(attn_output, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_heads, prompt_len, head_dim}))
      attn_transposed = Value.transpose(attn_merged, [0, 2, 1, 3], EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, local_heads, head_dim}))
      attn_flat = Value.reshape(attn_transposed, EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, local_q_size}))

      attn_partial = Value.dot_general(attn_flat, weights.o, {[2], [], [0], []}, :default, hidden_typespec)
      attn_result = Value.all_reduce(attn_partial, :sum, replica_groups, hidden_typespec)

      after_attn = Value.add(hidden, attn_result, hidden_typespec)

      # FFN
      normed_for_ffn = rms_norm.(after_attn, weights.ffn_norm, hidden_typespec)

      intermediate_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, prompt_len, local_intermediate})

      gate_out = Value.dot_general(normed_for_ffn, weights.gate, {[2], [], [0], []}, :default, intermediate_typespec)
      up_out = Value.dot_general(normed_for_ffn, weights.up, {[2], [], [0], []}, :default, intermediate_typespec)

      gate_sigmoid = Value.sigmoid(gate_out, intermediate_typespec)
      gate_silu = Value.multiply(gate_out, gate_sigmoid, intermediate_typespec)
      combined = Value.multiply(gate_silu, up_out, intermediate_typespec)

      ffn_partial = Value.dot_general(combined, weights.down, {[2], [], [0], []}, :default, hidden_typespec)
      ffn_result = Value.all_reduce(ffn_partial, :sum, replica_groups, hidden_typespec)

      layer_output = Value.add(after_attn, ffn_result, hidden_typespec)

      # Accumulate K/V caches for this layer
      {layer_output, caches ++ [k_cache, v_cache]}
    end)

    # Final norm + LM head (last position only)
    normed_output = rms_norm.(final_hidden, final_norm_w, hidden_typespec)

    last_hidden_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, hidden_size})
    last_hidden = Value.slice(normed_output, [0, prompt_len - 1, 0], [batch_size, prompt_len, hidden_size], [1, 1, 1], last_hidden_typespec)
    last_hidden_2d = Value.reshape(last_hidden, EXLA.Typespec.tensor({:f, 32}, {batch_size, hidden_size}))

    # lm_head_w is {vocab, hidden}, contract hidden dims: axis 1 of hidden with axis 1 of lm_head
    logits = Value.dot_general(last_hidden_2d, lm_head_w, {[1], [], [1], []}, :default, output_typespec)

    # Return logits + all K/V caches
    [logits] ++ kv_caches
  end, num_replicas: tp_size, client: :cuda)
end

IO.puts("  Prefill SPMD builder created!")

# Phase 3: Decode SPMD with pre-allocated KV cache (fixed shapes, position input)
IO.puts("  Building decode SPMD builder (with pre-allocated cache)...")

build_decode_spmd_fixed = fn batch_size, max_seq_len_local ->
  # Input: single token + position
  input_ids_typespec = EXLA.Typespec.tensor({:s, 32}, {batch_size, 1})
  position_typespec = EXLA.Typespec.tensor({:s, 32}, {})  # Scalar position
  embed_typespec = EXLA.Typespec.tensor({:f, 32}, {vocab_size, hidden_size})
  norm_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size})
  # For Qwen3: lm_head uses transposed embeddings (tied), shape is {vocab_size, hidden_size}
  lm_head_typespec = EXLA.Typespec.tensor({:f, 32}, {vocab_size, hidden_size})

  # RoPE embeddings for the single new position: [1, head_dim]
  rope_cos_typespec = EXLA.Typespec.tensor({:f, 32}, {1, head_dim})
  rope_sin_typespec = EXLA.Typespec.tensor({:f, 32}, {1, head_dim})

  q_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_q_size})
  k_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_kv_size})
  v_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_kv_size})
  o_typespec = EXLA.Typespec.tensor({:f, 32}, {local_q_size, hidden_size})
  gate_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_intermediate})
  up_typespec = EXLA.Typespec.tensor({:f, 32}, {hidden_size, local_intermediate})
  down_typespec = EXLA.Typespec.tensor({:f, 32}, {local_intermediate, hidden_size})

  # QK norm weights (Qwen3-specific) - shape {head_dim}
  qk_norm_typespec = EXLA.Typespec.tensor({:f, 32}, {head_dim})

  layer_param_typespecs = List.duplicate([
    norm_typespec, norm_typespec,
    qk_norm_typespec, qk_norm_typespec,  # q_norm, k_norm
    q_typespec, k_typespec, v_typespec, o_typespec,
    gate_typespec, up_typespec, down_typespec
  ], num_layers) |> List.flatten()

  # Phase 3: Fixed-size K/V caches (same shape for input and output)
  k_cache_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, max_seq_len_local, head_dim})
  v_cache_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, max_seq_len_local, head_dim})
  cache_typespecs = List.duplicate([k_cache_typespec, v_cache_typespec], num_layers) |> List.flatten()

  # Add position input after rope embeddings
  input_typespecs = [input_ids_typespec, position_typespec, embed_typespec, norm_typespec, lm_head_typespec, rope_cos_typespec, rope_sin_typespec] ++ layer_param_typespecs ++ cache_typespecs

  # Outputs: logits + updated K/V caches (same fixed shape)
  output_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, vocab_size})
  output_typespecs = [output_typespec] ++ cache_typespecs

  replica_groups = [Enum.to_list(0..(tp_size - 1))]

  EXLA.SPMD.build(input_typespecs, output_typespecs, fn builder ->
    args = Function.get_arguments(builder)
    [input_ids, position, embed_w, final_norm_w, lm_head_w, rope_cos, rope_sin | rest_args] = args

    # Split into layer params (11 per layer with QK norm) and caches (2 per layer)
    num_param_args = num_layers * 11
    {layer_param_args, cache_args} = Enum.split(rest_args, num_param_args)

    # Extract layer weights (including QK norm)
    layer_weights = Enum.chunk_every(layer_param_args, 11)
    |> Enum.map(fn [sa_norm, ffn_norm, q_norm, k_norm, q, k, v, o, gate, up, down] ->
      %{sa_norm: sa_norm, ffn_norm: ffn_norm, q_norm: q_norm, k_norm: k_norm,
        q: q, k: k, v: v, o: o, gate: gate, up: up, down: down}
    end)

    # Extract K/V caches (2 per layer)
    cache_pairs = Enum.chunk_every(cache_args, 2)

    # Combine weights and caches
    layer_weights_and_caches = Enum.zip(layer_weights, cache_pairs)
    |> Enum.map(fn {weights, [k_cache_in, v_cache_in]} ->
      {weights, k_cache_in, v_cache_in}
    end)

    hidden_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, hidden_size})

    # RoPE application function for decode (single position)
    apply_rope_decode = fn x, cos_embed, sin_embed, x_typespec ->
      # x shape: [batch, num_heads, 1, head_dim]
      # cos/sin shape: [1, head_dim]

      {batch_local, num_heads_local, seq_len_local, _head_dim_local} = x_typespec.shape
      cos_4d_typespec = EXLA.Typespec.tensor({:f, 32}, {1, 1, 1, head_dim})
      sin_4d_typespec = EXLA.Typespec.tensor({:f, 32}, {1, 1, 1, head_dim})

      cos_4d = Value.reshape(cos_embed, cos_4d_typespec)
      sin_4d = Value.reshape(sin_embed, sin_4d_typespec)

      cos_broadcast = Value.broadcast_in_dim(cos_4d, [0, 1, 2, 3], x_typespec)
      sin_broadcast = Value.broadcast_in_dim(sin_4d, [0, 1, 2, 3], x_typespec)

      # Create rotate_half(x)
      half_dim = div(head_dim, 2)
      first_half_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_local, num_heads_local, seq_len_local, half_dim})
      second_half_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_local, num_heads_local, seq_len_local, half_dim})

      # Slice first half: [0:batch, 0:heads, 0:seq, 0:half_dim]
      x_first = Value.slice(x, [0, 0, 0, 0], [batch_local, num_heads_local, seq_len_local, half_dim], [1, 1, 1, 1], first_half_typespec)
      # Slice second half: [0:batch, 0:heads, 0:seq, half_dim:head_dim]
      x_second = Value.slice(x, [0, 0, 0, half_dim], [batch_local, num_heads_local, seq_len_local, head_dim], [1, 1, 1, 1], second_half_typespec)

      neg_one = Value.constant(builder, [-1.0], EXLA.Typespec.tensor({:f, 32}, {}))
      neg_one_broadcast = Value.broadcast_in_dim(neg_one, [], second_half_typespec)
      x_second_neg = Value.multiply(x_second, neg_one_broadcast, second_half_typespec)

      x_rotated = Value.concatenate([x_second_neg, x_first], 3, x_typespec)

      x_cos = Value.multiply(x, cos_broadcast, x_typespec)
      x_rot_sin = Value.multiply(x_rotated, sin_broadcast, x_typespec)
      Value.add(x_cos, x_rot_sin, x_typespec)
    end

    # PROPER EMBEDDING LOOKUP using gather (for single token)
    hidden_states = Value.gather(
      embed_w,
      input_ids,
      2,
      [1, hidden_size],
      [2],
      [0],
      [0],
      hidden_typespec
    )

    # PROPER RMSNorm implementation (for decode, seq_len=1)
    rms_norm = fn x, weight, typespec ->
      scalar_typespec = EXLA.Typespec.tensor({:f, 32}, {})
      reduce_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1})

      x_squared = Value.multiply(x, x, typespec)

      {region, [lhs, rhs]} = Function.push_region(builder, [scalar_typespec, scalar_typespec])
      sum_result = Value.add(lhs, rhs, scalar_typespec)
      Value.return(builder, [sum_result])
      Function.pop_region(builder)

      zero = Value.constant(builder, [0.0], scalar_typespec)
      [sum_squared] = Value.reduce(region, [zero], [x_squared], [2], [reduce_typespec])

      hidden_size_constant = Value.constant(builder, [hidden_size * 1.0], scalar_typespec)
      hidden_size_broadcast = Value.broadcast_in_dim(hidden_size_constant, [], reduce_typespec)
      mean_squared = Value.divide(sum_squared, hidden_size_broadcast, reduce_typespec)

      epsilon_constant = Value.constant(builder, [rms_norm_eps], scalar_typespec)
      epsilon_broadcast = Value.broadcast_in_dim(epsilon_constant, [], reduce_typespec)
      mean_squared_eps = Value.add(mean_squared, epsilon_broadcast, reduce_typespec)

      rsqrt = Value.rsqrt(mean_squared_eps, reduce_typespec)

      rsqrt_3d_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, 1})
      rsqrt_3d = Value.reshape(rsqrt, rsqrt_3d_typespec)
      rsqrt_broadcast = Value.broadcast_in_dim(rsqrt_3d, [0, 1, 2], typespec)

      normalized = Value.multiply(x, rsqrt_broadcast, typespec)
      weight_broadcast = Value.broadcast_in_dim(weight, [2], typespec)
      Value.multiply(normalized, weight_broadcast, typespec)
    end

    # Process layers with pre-allocated cache
    {final_hidden, updated_caches} = Enum.reduce(layer_weights_and_caches, {hidden_states, []}, fn {weights, k_cache_in, v_cache_in}, {hidden, acc_caches} ->
      normed_for_attn = rms_norm.(hidden, weights.sa_norm, hidden_typespec)

      q_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_q_size})
      k_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_kv_size})
      v_proj_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_kv_size})

      q = Value.dot_general(normed_for_attn, weights.q, {[2], [], [0], []}, :default, q_proj_typespec)
      k_new = Value.dot_general(normed_for_attn, weights.k, {[2], [], [0], []}, :default, k_proj_typespec)
      v_new = Value.dot_general(normed_for_attn, weights.v, {[2], [], [0], []}, :default, v_proj_typespec)

      # Reshape new K/V
      k_new_reshaped = Value.reshape(k_new, EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_kv_heads, head_dim}))
      v_new_reshaped = Value.reshape(v_new, EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_kv_heads, head_dim}))

      k_new_transposed_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, head_dim})
      k_new_transposed_raw = Value.transpose(k_new_reshaped, [0, 2, 1, 3], k_new_transposed_typespec)
      v_new_transposed = Value.transpose(v_new_reshaped, [0, 2, 1, 3], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, head_dim}))

      # QK Norm (Qwen3-specific): RMSNorm on Q and K per head before RoPE
      qk_rms_norm_decode = fn x, weight, typespec ->
        {batch_local, num_heads_local, seq_len_local, head_dim_local} = typespec.shape

        x_squared = Value.multiply(x, x, typespec)

        scalar_typespec = EXLA.Typespec.tensor({:f, 32}, {})
        reduce_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_local, num_heads_local, seq_len_local})

        {region, [lhs, rhs]} = Function.push_region(builder, [scalar_typespec, scalar_typespec])
        sum_result = Value.add(lhs, rhs, scalar_typespec)
        Value.return(builder, [sum_result])
        Function.pop_region(builder)

        zero = Value.constant(builder, [0.0], scalar_typespec)
        [sum_squared] = Value.reduce(region, [zero], [x_squared], [3], [reduce_typespec])

        head_dim_constant = Value.constant(builder, [head_dim_local * 1.0], scalar_typespec)
        head_dim_broadcast = Value.broadcast_in_dim(head_dim_constant, [], reduce_typespec)
        mean_squared = Value.divide(sum_squared, head_dim_broadcast, reduce_typespec)

        epsilon_constant = Value.constant(builder, [rms_norm_eps], scalar_typespec)
        epsilon_broadcast = Value.broadcast_in_dim(epsilon_constant, [], reduce_typespec)
        mean_squared_eps = Value.add(mean_squared, epsilon_broadcast, reduce_typespec)

        rsqrt = Value.rsqrt(mean_squared_eps, reduce_typespec)

        rsqrt_4d_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_local, num_heads_local, seq_len_local, 1})
        rsqrt_4d = Value.reshape(rsqrt, rsqrt_4d_typespec)
        rsqrt_broadcast = Value.broadcast_in_dim(rsqrt_4d, [0, 1, 2, 3], typespec)

        normalized = Value.multiply(x, rsqrt_broadcast, typespec)
        weight_broadcast = Value.broadcast_in_dim(weight, [3], typespec)
        Value.multiply(normalized, weight_broadcast, typespec)
      end

      # Apply QK norm to K (Qwen3-specific)
      k_new_normed = qk_rms_norm_decode.(k_new_transposed_raw, weights.k_norm, k_new_transposed_typespec)

      # Apply RoPE to new K (the current position) - conditionally
      k_new_transposed = if use_rope do
        apply_rope_decode.(k_new_normed, rope_cos, rope_sin, k_new_transposed_typespec)
      else
        k_new_normed
      end

      # Phase 3: Use dynamic_update_slice to write new K/V at position
      # Position is the index where we write (0-indexed)
      zero_idx = Value.constant(builder, [0], EXLA.Typespec.tensor({:s, 32}, {}))
      k_cache_updated = Value.dynamic_update_slice(k_cache_in, k_new_transposed, [zero_idx, zero_idx, position, zero_idx], k_cache_typespec)
      v_cache_updated = Value.dynamic_update_slice(v_cache_in, v_new_transposed, [zero_idx, zero_idx, position, zero_idx], v_cache_typespec)

      # Reshape Q and apply RoPE (conditionally)
      q_reshaped = Value.reshape(q, EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_heads, head_dim}))
      q_transposed_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_heads, 1, head_dim})
      q_transposed_raw = Value.transpose(q_reshaped, [0, 2, 1, 3], q_transposed_typespec)

      # Apply QK norm to Q (Qwen3-specific)
      q_normed = qk_rms_norm_decode.(q_transposed_raw, weights.q_norm, q_transposed_typespec)

      q_transposed = if use_rope do
        apply_rope_decode.(q_normed, rope_cos, rope_sin, q_transposed_typespec)
      else
        q_normed
      end

      # Grouped query attention using full cache (attention mask will handle valid positions)
      k_for_attn = Value.transpose(k_cache_updated, [0, 1, 3, 2], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, head_dim, max_seq_len_local}))

      q_grouped = Value.reshape(q_transposed, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, 1, head_dim}))
      k_expanded = Value.reshape(k_for_attn, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, head_dim, max_seq_len_local}))
      v_expanded = Value.reshape(v_cache_updated, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, 1, max_seq_len_local, head_dim}))

      k_broadcast = Value.broadcast_in_dim(k_expanded, [0, 1, 2, 3, 4], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, head_dim, max_seq_len_local}))
      v_broadcast = Value.broadcast_in_dim(v_expanded, [0, 1, 2, 3, 4], EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, max_seq_len_local, head_dim}))

      scores_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, 1, max_seq_len_local})
      scores = Value.dot_general(q_grouped, k_broadcast, {[4], [0, 1, 2], [3], [0, 1, 2]}, :default, scores_typespec)

      scale_value = 1.0 / :math.sqrt(head_dim)
      scale_typespec = EXLA.Typespec.tensor({:f, 32}, {})
      scale_tensor = Value.constant(builder, [scale_value], scale_typespec)
      scale_broadcast = Value.broadcast_in_dim(scale_tensor, [], scores_typespec)
      scores_scaled = Value.multiply(scores, scale_broadcast, scores_typespec)

      # Phase 3: Create attention mask based on position
      # Mask positions > position (we're at position, so attend to 0..position)
      col_indices_typespec = EXLA.Typespec.tensor({:s, 32}, {max_seq_len_local})
      col_indices = Value.iota(builder, 0, col_indices_typespec)

      # Broadcast position to compare
      position_broadcast_typespec = EXLA.Typespec.tensor({:s, 32}, {max_seq_len_local})
      position_broadcast = Value.broadcast_in_dim(position, [], position_broadcast_typespec)

      # Mask: attend where col_idx <= position
      valid_mask = Value.less_equal(col_indices, position_broadcast, EXLA.Typespec.tensor({:pred, 8}, {max_seq_len_local}))

      # Convert to float mask
      neg_inf_scalar = Value.constant(builder, [-1.0e9], scale_typespec)
      zero_scalar = Value.constant(builder, [0.0], scale_typespec)
      mask_1d_typespec = EXLA.Typespec.tensor({:f, 32}, {max_seq_len_local})
      neg_inf_1d = Value.broadcast_in_dim(neg_inf_scalar, [], mask_1d_typespec)
      zero_1d = Value.broadcast_in_dim(zero_scalar, [], mask_1d_typespec)
      mask_1d = Value.select(valid_mask, zero_1d, neg_inf_1d, mask_1d_typespec)

      # Broadcast mask to full attention shape
      mask_5d_typespec = EXLA.Typespec.tensor({:f, 32}, {1, 1, 1, 1, max_seq_len_local})
      mask_5d = Value.reshape(mask_1d, mask_5d_typespec)
      mask_broadcast = Value.broadcast_in_dim(mask_5d, [0, 1, 2, 3, 4], scores_typespec)

      # Apply mask
      scores_masked = Value.add(scores_scaled, mask_broadcast, scores_typespec)

      # PROPER SOFTMAX for decode phase
      scalar_typespec = EXLA.Typespec.tensor({:f, 32}, {})
      reduce_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, 1})

      {region, [lhs, rhs]} = Function.push_region(builder, [scalar_typespec, scalar_typespec])
      max_result = Value.max(lhs, rhs, scalar_typespec)
      Value.return(builder, [max_result])
      Function.pop_region(builder)

      neg_inf = Value.constant(builder, [-1.0e9], scalar_typespec)
      [max_scores] = Value.reduce(region, [neg_inf], [scores_masked], [4], [reduce_typespec])

      max_expanded = Value.reshape(max_scores, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, 1, 1}))
      max_broadcast = Value.broadcast_in_dim(max_expanded, [0, 1, 2, 3, 4], scores_typespec)
      scores_shifted = Value.subtract(scores_masked, max_broadcast, scores_typespec)
      scores_exp = Value.exp(scores_shifted, scores_typespec)

      {region, [lhs, rhs]} = Function.push_region(builder, [scalar_typespec, scalar_typespec])
      sum_result = Value.add(lhs, rhs, scalar_typespec)
      Value.return(builder, [sum_result])
      Function.pop_region(builder)

      zero = Value.constant(builder, [0.0], scalar_typespec)
      [sum_exp] = Value.reduce(region, [zero], [scores_exp], [4], [reduce_typespec])

      sum_expanded = Value.reshape(sum_exp, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, 1, 1}))
      sum_broadcast = Value.broadcast_in_dim(sum_expanded, [0, 1, 2, 3, 4], scores_typespec)
      attention_weights = Value.divide(scores_exp, sum_broadcast, scores_typespec)

      attn_output_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, local_kv_heads, kv_head_repeat, 1, head_dim})
      attn_output = Value.dot_general(attention_weights, v_broadcast, {[4], [0, 1, 2], [3], [0, 1, 2]}, :default, attn_output_typespec)

      attn_merged = Value.reshape(attn_output, EXLA.Typespec.tensor({:f, 32}, {batch_size, local_heads, 1, head_dim}))
      attn_transposed = Value.transpose(attn_merged, [0, 2, 1, 3], EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_heads, head_dim}))
      attn_flat = Value.reshape(attn_transposed, EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_q_size}))

      attn_partial = Value.dot_general(attn_flat, weights.o, {[2], [], [0], []}, :default, hidden_typespec)
      attn_result = Value.all_reduce(attn_partial, :sum, replica_groups, hidden_typespec)

      after_attn = Value.add(hidden, attn_result, hidden_typespec)

      # FFN
      normed_for_ffn = rms_norm.(after_attn, weights.ffn_norm, hidden_typespec)

      intermediate_typespec = EXLA.Typespec.tensor({:f, 32}, {batch_size, 1, local_intermediate})

      gate_out = Value.dot_general(normed_for_ffn, weights.gate, {[2], [], [0], []}, :default, intermediate_typespec)
      up_out = Value.dot_general(normed_for_ffn, weights.up, {[2], [], [0], []}, :default, intermediate_typespec)

      gate_sigmoid = Value.sigmoid(gate_out, intermediate_typespec)
      gate_silu = Value.multiply(gate_out, gate_sigmoid, intermediate_typespec)
      combined = Value.multiply(gate_silu, up_out, intermediate_typespec)

      ffn_partial = Value.dot_general(combined, weights.down, {[2], [], [0], []}, :default, hidden_typespec)
      ffn_result = Value.all_reduce(ffn_partial, :sum, replica_groups, hidden_typespec)

      layer_output = Value.add(after_attn, ffn_result, hidden_typespec)

      {layer_output, acc_caches ++ [k_cache_updated, v_cache_updated]}
    end)

    # Final norm + LM head
    normed_output = rms_norm.(final_hidden, final_norm_w, hidden_typespec)
    last_hidden_2d = Value.reshape(normed_output, EXLA.Typespec.tensor({:f, 32}, {batch_size, hidden_size}))
    # lm_head_w is {vocab, hidden}, contract hidden dims: axis 1 of hidden with axis 1 of lm_head
    logits = Value.dot_general(last_hidden_2d, lm_head_w, {[1], [], [1], []}, :default, output_typespec)

    [logits] ++ updated_caches
  end, num_replicas: tp_size, client: :cuda)
end

IO.puts("  Decode SPMD builder created!")
IO.puts("  Note: This will take longer to compile due to proper operations")

# ----------------------------------------------------------
# Step 4: Tokenize prompt(s) - Phase 4: Batch processing support
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 4: Tokenizing prompt(s)")
IO.puts("-" |> String.duplicate(70))

# Qwen3 chat template default prompt
default_prompt = """
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
"""

# Phase 4: Support multiple prompts via PROMPTS env var (separated by |||)
single_prompt = System.get_env("PROMPT", String.trim(default_prompt))
prompts_str = System.get_env("PROMPTS", single_prompt)
prompts = String.split(prompts_str, "|||") |> Enum.take(batch_size)

# Qwen3 special tokens
pad_token_id = 151643  # <|endoftext|>

# Pad prompts list to batch_size if needed
prompts = if length(prompts) < batch_size do
  prompts ++ List.duplicate(hd(prompts), batch_size - length(prompts))
else
  prompts
end

IO.puts("  Batch size: #{batch_size}")
for {p, idx} <- Enum.with_index(prompts) do
  IO.puts("  Prompt #{idx}: \"#{String.slice(p, 0..50)}#{if String.length(p) > 50, do: "...", else: ""}\"")
end

# Tokenize all prompts
tokenized_list = Enum.map(prompts, fn p -> Bumblebee.apply_tokenizer(tokenizer, p) end)
prompt_lengths = Enum.map(tokenized_list, fn t -> Nx.axis_size(t["input_ids"], 1) end)
max_prompt_len = Enum.max(prompt_lengths)

IO.puts("  Prompt lengths: #{inspect(prompt_lengths)}")
IO.puts("  Max prompt length: #{max_prompt_len}")

# Phase 4: Left-pad inputs to max length (for proper causal attention)
padded_input_ids = for {tok, len} <- Enum.zip(tokenized_list, prompt_lengths) do
  ids = tok["input_ids"] |> Nx.as_type(:s32)
  if len < max_prompt_len do
    # Left-pad with pad_token_id
    padding = Nx.broadcast(pad_token_id, {1, max_prompt_len - len}) |> Nx.as_type(:s32)
    Nx.concatenate([padding, ids], axis: 1)
  else
    ids
  end
end

# Stack into batch tensor
input_ids = Nx.concatenate(padded_input_ids, axis: 0)  # {batch_size, max_prompt_len}
prompt_length = max_prompt_len

IO.puts("  Batched input shape: #{inspect(Nx.shape(input_ids))}")
if batch_size == 1 do
  IO.puts("  Token IDs: #{inspect(Nx.to_flat_list(input_ids))}")
end

# Track sequence lengths per batch for proper attention masking
seq_lengths = Nx.tensor(prompt_lengths, type: :s32)

# ----------------------------------------------------------
# Step 5: Test prefill phase
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 5: Testing prefill phase")
IO.puts("-" |> String.duplicate(70))

# Phase 3: Validate prompt fits in max_seq_len
if prompt_length + max_new_tokens > max_seq_len do
  IO.puts("  WARNING: prompt_length (#{prompt_length}) + max_new_tokens (#{max_new_tokens}) > max_seq_len (#{max_seq_len})")
  IO.puts("  Adjusting max_new_tokens to fit.")
end
actual_max_new_tokens = min(max_new_tokens, max_seq_len - prompt_length)

IO.puts("  Building prefill SPMD for batch_size=#{batch_size}, prompt_len=#{prompt_length}, max_seq_len=#{max_seq_len}...")
IO.puts("  (This may take a while due to proper gather/reduce/softmax...)")
prefill_spmd = build_prefill_spmd.(batch_size, prompt_length, max_seq_len)

# Prepare replica inputs (including QK norm weights)
prepare_prefill_inputs = fn input_ids, rope_cos, rope_sin ->
  for gpu <- 0..(tp_size - 1) do
    layer_params_for_gpu = Enum.flat_map(layer_params, fn layer ->
      [
        layer.sa_norm, layer.ffn_norm,
        layer.q_norm, layer.k_norm,  # QK norm weights (Qwen3-specific)
        Enum.at(layer.q_shards, gpu), Enum.at(layer.k_shards, gpu),
        Enum.at(layer.v_shards, gpu), Enum.at(layer.o_shards, gpu),
        Enum.at(layer.gate_shards, gpu), Enum.at(layer.up_shards, gpu),
        Enum.at(layer.down_shards, gpu)
      ]
    end)

    [input_ids, embed_tokens, final_norm_weight, lm_head_kernel, rope_cos, rope_sin] ++ layer_params_for_gpu
  end
end

IO.puts("  Running prefill on #{tp_size} GPUs...")
{rope_cos_prefill, rope_sin_prefill} = compute_rope_embeddings.(prompt_length)
replica_inputs = prepare_prefill_inputs.(input_ids, rope_cos_prefill, rope_sin_prefill)

{time_us, results} = :timer.tc(fn ->
  EXLA.SPMD.run(prefill_spmd, replica_inputs)
end)

IO.puts("  Prefill completed in #{Float.round(time_us / 1000, 2)} ms")

# Extract results - need ALL replicas' KV caches (each GPU has local heads)
# results is [[logits | kv_caches], [logits | kv_caches], [logits | kv_caches], [logits | kv_caches]]
# We take logits from first replica (they're all-reduced so identical)
# But we need KV caches from EACH replica
[[logits | _] | _] = results
kv_caches_per_replica = for replica_result <- results do
  [_ | kv_caches] = replica_result
  kv_caches
end

IO.puts("\n  Output shapes:")
IO.puts("    Logits: #{inspect(Nx.shape(logits))}")
IO.puts("    Number of K/V cache pairs per GPU: #{div(length(hd(kv_caches_per_replica)), 2)}")
IO.puts("    K cache shape (per layer): #{inspect(Nx.shape(hd(hd(kv_caches_per_replica))))}")

# Initialize random key for sampling
random_key = Nx.Random.key(System.system_time(:nanosecond))

# Phase 4: Sample first token for each batch item
# For batch_size > 1, we need to sample each item independently
{next_tokens, random_key} = if batch_size == 1 do
  # Single batch - use original sampling
  {token, key} = if temperature > 0 do
    sample_token.(logits, random_key, temperature, top_k, top_p)
  else
    {Nx.argmax(logits, axis: 1) |> Nx.to_flat_list() |> hd(), random_key}
  end
  {[token], key}
else
  # Batch mode - sample each item
  Enum.reduce(0..(batch_size - 1), {[], random_key}, fn b, {tokens, key} ->
    batch_logits = Nx.slice(logits, [b, 0], [1, vocab_size])
    {token, new_key} = if temperature > 0 do
      sample_token.(batch_logits, key, temperature, top_k, top_p)
    else
      {Nx.argmax(batch_logits, axis: 1) |> Nx.to_flat_list() |> hd(), key}
    end
    {tokens ++ [token], new_key}
  end)
end

IO.puts("\n  First generated tokens: #{inspect(next_tokens)}")
for {tok, idx} <- Enum.with_index(next_tokens) do
  IO.puts("    Batch #{idx}: \"#{Bumblebee.Tokenizer.decode(tokenizer, [tok])}\"")
end

# ----------------------------------------------------------
# Step 6: Autoregressive generation with KV cache
# ----------------------------------------------------------
IO.puts("\n" <> ("-" |> String.duplicate(70)))
IO.puts("Step 6: Generating text with KV cache (#{max_new_tokens - 1} more tokens)")
IO.puts("-" |> String.duplicate(70))

# Phase 3: Precompute RoPE embeddings for max sequence length
{rope_cos_full, rope_sin_full} = compute_rope_embeddings.(max_seq_len)

# Phase 3: Prepare decode inputs function (includes position tensor)
prepare_decode_inputs_fixed = fn token_id, position, caches ->
  token_tensor = Nx.tensor([[token_id]], type: :s32)
  position_tensor = Nx.tensor(position, type: :s32)  # Scalar position

  # Get RoPE embeddings for this specific position (from precomputed)
  rope_cos_pos = Nx.slice(rope_cos_full, [position, 0], [1, head_dim])
  rope_sin_pos = Nx.slice(rope_sin_full, [position, 0], [1, head_dim])

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

    # Phase 3: Position tensor comes after token_tensor
    [token_tensor, position_tensor, embed_tokens, final_norm_weight, lm_head_kernel, rope_cos_pos, rope_sin_pos] ++ layer_params_for_gpu ++ caches
  end
end

# Phase 3: Build decode SPMD once (fixed shape)
IO.puts("  Building decode SPMD for batch_size=#{batch_size}, max_seq_len=#{max_seq_len} (one-time compilation)...")
decode_spmd_fixed = build_decode_spmd_fixed.(batch_size, max_seq_len)
IO.puts("  Decode SPMD compiled!")

# EOS token for stopping generation (Qwen3: <|im_end|> or <|endoftext|>)
eos_token_id = 151645  # <|im_end|> token for Qwen3

# Phase 4: Generate tokens with decode phase (batch-aware)
# Track generated tokens per batch item
generated_tokens_per_batch = for tok <- next_tokens, do: [tok]
current_caches_per_replica = kv_caches_per_replica  # Per-GPU caches
current_position = prompt_length  # Position for next token

# Phase 4: Track finished status per batch
finished_mask = for tok <- next_tokens, do: tok == eos_token_id

# Phase 5: Streaming callback (default: print to stdout)
stream_callback = fn event ->
  case event do
    {:token, batch_idx, _token_id, decoded_text, _position} ->
      if stream_mode and (batch_size == 1 or batch_idx == 0) do
        IO.write(decoded_text)
      end
      :continue
    {:progress, i, total} ->
      if stream_mode, do: IO.write(" [#{i}/#{total}]")
      :continue
    {:eos, batch_idx} ->
      if stream_mode and (batch_size == 1 or batch_idx == 0) do
        IO.write(" [EOS]")
      end
      :continue
    _ -> :continue
  end
end

# Print initial generated text for batch 0
IO.write("  Generated text: #{Bumblebee.Tokenizer.decode(tokenizer, hd(next_tokens))}")

# Check if all batches finished
all_finished = Enum.all?(finished_mask)

# Phase 4: Prepare batch decode inputs (all batch items with same position)
# NOTE: caches_per_replica is a list of 4 cache lists (one per GPU)
prepare_batch_decode_inputs = fn tokens_per_batch, position, caches_per_replica ->
  # Create batch token tensor from last token of each batch
  last_tokens = Enum.map(tokens_per_batch, &List.last/1)
  token_tensor = Nx.tensor([last_tokens], type: :s32) |> Nx.transpose()  # {batch_size, 1}
  position_tensor = Nx.tensor(position, type: :s32)  # Scalar position

  # Get RoPE embeddings for this position
  rope_cos_pos = Nx.slice(rope_cos_full, [position, 0], [1, head_dim])
  rope_sin_pos = Nx.slice(rope_sin_full, [position, 0], [1, head_dim])

  for gpu <- 0..(tp_size - 1) do
    layer_params_for_gpu = Enum.flat_map(layer_params, fn layer ->
      [
        layer.sa_norm, layer.ffn_norm,
        layer.q_norm, layer.k_norm,  # QK norm weights (Qwen3-specific)
        Enum.at(layer.q_shards, gpu), Enum.at(layer.k_shards, gpu),
        Enum.at(layer.v_shards, gpu), Enum.at(layer.o_shards, gpu),
        Enum.at(layer.gate_shards, gpu), Enum.at(layer.up_shards, gpu),
        Enum.at(layer.down_shards, gpu)
      ]
    end)

    # Use each GPU's own cache!
    gpu_caches = Enum.at(caches_per_replica, gpu)
    [token_tensor, position_tensor, embed_tokens, final_norm_weight, lm_head_kernel, rope_cos_pos, rope_sin_pos] ++ layer_params_for_gpu ++ gpu_caches
  end
end

{final_generated_tokens_per_batch, _, _, _, _} = if all_finished do
  {generated_tokens_per_batch, current_caches_per_replica, current_position, random_key, finished_mask}
else
  Enum.reduce_while(1..(actual_max_new_tokens - 1), {generated_tokens_per_batch, current_caches_per_replica, current_position, random_key, finished_mask}, fn i, {tokens_per_batch, caches_per_replica, position, key, finished} ->
    # Phase 3: Use pre-compiled decode SPMD (no recompilation!)
    replica_inputs = prepare_batch_decode_inputs.(tokens_per_batch, position, caches_per_replica)

    # Run decode - need ALL replicas' updated caches
    decode_results = EXLA.SPMD.run(decode_spmd_fixed, replica_inputs)
    # Take logits from first replica (all-reduced so identical)
    [[new_logits | _] | _] = decode_results
    # Extract updated caches from each replica
    new_caches_per_replica = for replica_result <- decode_results do
      [_ | new_caches] = replica_result
      new_caches
    end

    # Phase 4: Sample next token for each batch item
    {new_tokens, new_key} = Enum.reduce(0..(batch_size - 1), {[], key}, fn b, {tokens, k} ->
      # Skip finished sequences (but still add a placeholder)
      if Enum.at(finished, b) do
        {tokens ++ [eos_token_id], k}
      else
        batch_logits = Nx.slice(new_logits, [b, 0], [1, vocab_size])
        {token, new_k} = if temperature > 0 do
          sample_token.(batch_logits, k, temperature, top_k, top_p)
        else
          {Nx.argmax(batch_logits, axis: 1) |> Nx.to_flat_list() |> hd(), k}
        end
        {tokens ++ [token], new_k}
      end
    end)

    # Update tokens per batch
    new_tokens_per_batch = for {tok_list, new_tok} <- Enum.zip(tokens_per_batch, new_tokens) do
      tok_list ++ [new_tok]
    end

    # Phase 5: Stream token for batch 0
    batch0_token = hd(new_tokens)
    unless hd(finished) do
      decoded_token = Bumblebee.Tokenizer.decode(tokenizer, [batch0_token])
      stream_callback.({:token, 0, batch0_token, decoded_token, position})
    end

    # Progress indicator every 5 tokens
    if rem(i, 5) == 0 do
      stream_callback.({:progress, i, actual_max_new_tokens - 1})
    end

    # Update finished mask
    new_finished = for {old_fin, new_tok} <- Enum.zip(finished, new_tokens) do
      old_fin or new_tok == eos_token_id
    end

    # Report EOS for any newly finished
    for {old_fin, new_fin, b} <- Enum.zip([finished, new_finished, 0..(batch_size - 1)]) |> Enum.map(&Tuple.to_list/1) |> Enum.map(&List.to_tuple/1) do
      if new_fin and not old_fin do
        stream_callback.({:eos, b})
      end
    end

    # Check if all finished
    if Enum.all?(new_finished) do
      {:halt, {new_tokens_per_batch, new_caches_per_replica, position + 1, new_key, new_finished}}
    else
      {:cont, {new_tokens_per_batch, new_caches_per_replica, position + 1, new_key, new_finished}}
    end
  end)
end

# Flatten for single batch case
final_generated_tokens = if batch_size == 1 do
  hd(final_generated_tokens_per_batch)
else
  final_generated_tokens_per_batch
end

IO.puts("")

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
IO.puts("\n" <> ("=" |> String.duplicate(70)))
IO.puts("Summary")
IO.puts("=" |> String.duplicate(70))

# Phase 4: Generate output for all batches
if batch_size == 1 do
  full_tokens = Nx.to_flat_list(input_ids) ++ final_generated_tokens
  full_text = Bumblebee.Tokenizer.decode(tokenizer, full_tokens)

  IO.puts("""

  ✓ COMPLETE - Production-Quality TP Generation!

  Implemented:
    - PROPER embedding lookup via Value.gather
    - PROPER RMSNorm via Value.reduce
    - PROPER softmax attention via Value.reduce
    - PROPER causal masking via Value.iota
    - PROPER Rotary Position Embeddings (RoPE)
    - QK Norm (Qwen3-specific)
    - Full autoregressive generation with KV cache
    - Sampling (temperature/top-k/top-p)
    - EOS token detection
    - Pre-allocated KV cache (no recompilation)
    - Streaming output with callback
    - Batch processing support

  Prompt: "#{String.slice(hd(prompts), 0..50)}..."

  Full Generated Text:
  #{full_text}

  Performance:
    - Prefill: #{Float.round(time_us / 1000, 2)} ms for #{prompt_length} tokens
    - Decode: #{length(final_generated_tokens)} tokens generated with O(1) computation per token
    - Pre-allocated cache: no recompilation per token!

  Configuration:
    - Model: Qwen3-4B-Instruct (#{num_layers} layers)
    - TP: #{tp_size} GPUs
    - Max sequence length: #{max_seq_len}
    - Batch size: #{batch_size}
    - Temperature: #{temperature}, Top-k: #{top_k}, Top-p: #{top_p}
    - Cache shape: [#{batch_size}, #{local_kv_heads}, #{max_seq_len}, #{head_dim}]

  Environment Variables:
    LAYERS=#{num_layers} TOKENS=#{max_new_tokens} TEMP=#{temperature} TOP_K=#{top_k} TOP_P=#{top_p}
    MAX_SEQ=#{max_seq_len} BATCH=#{batch_size} STREAM=#{stream_mode}
  """)
else
  # Batch output
  IO.puts("\n\n✓ COMPLETE - Batch Generation (#{batch_size} sequences)!")
  IO.puts("\nGenerated sequences:")
  for {prompt_text, gen_tokens, idx} <- Enum.zip([prompts, final_generated_tokens_per_batch, 0..(batch_size - 1)]) |> Enum.map(&Tuple.to_list/1) |> Enum.map(&List.to_tuple/1) do
    # Get original prompt tokens (accounting for left padding)
    orig_len = Enum.at(prompt_lengths, idx)
    prompt_tokens = Nx.slice(input_ids, [idx, prompt_length - orig_len], [1, orig_len]) |> Nx.to_flat_list()
    full_tokens = prompt_tokens ++ gen_tokens
    full_text = Bumblebee.Tokenizer.decode(tokenizer, full_tokens)
    IO.puts("\n  [Batch #{idx}] #{String.slice(full_text, 0..200)}#{if String.length(full_text) > 200, do: "...", else: ""}")
    IO.puts("    (#{length(gen_tokens)} tokens generated)")
  end

  IO.puts("""

  Performance:
    - Prefill: #{Float.round(time_us / 1000, 2)} ms for #{batch_size}x#{prompt_length} tokens
    - Pre-allocated cache: no recompilation per token!

  Configuration:
    - Model: Qwen3-4B-Instruct (#{num_layers} layers)
    - TP: #{tp_size} GPUs, Batch: #{batch_size}
    - Cache shape: [#{batch_size}, #{local_kv_heads}, #{max_seq_len}, #{head_dim}]
  """)
end
