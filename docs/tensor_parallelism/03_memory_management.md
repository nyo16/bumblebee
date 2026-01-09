# Memory Management

This document explains GPU memory allocation, KV cache sizing, and troubleshooting OOM errors.

## Memory Formula

Total GPU memory requirement follows this formula (similar to vLLM):

```
Total Memory = Model Weights + KV Cache + Activations + XLA Overhead

Where:
  Model Weights = total_params × bytes_per_param
  KV Cache = 2 × batch × seq_len × layers × kv_heads × head_dim × bytes
  Activations = batch × seq × hidden × multiplier × bytes
  XLA Overhead = compilation buffers + NCCL buffers + CUDA context
```

### Example: Qwen3-4B (f32)

```
Model Parameters:
  Embedding:     389M params (vocab × hidden = 151,936 × 2,560)
  Per layer:     107M params
  All 36 layers: 3,850M params
  Total:         ~4,022M params

Memory breakdown:
  Model weights: 4,022M × 4 bytes = 15.0 GB
  KV cache (seq=128): 36 MB total
  Activations: ~10 MB
  XLA overhead: ~2-5 GB (compilation, NCCL, CUDA)

  Total: ~17-20 GB
  Per GPU (TP=4): ~4.5-5 GB actual usage
```

## Memory Calculator Module

The implementation includes a memory calculator for estimating requirements:

```elixir
defmodule MemoryCalculator do
  @moduledoc """
  Calculate GPU memory requirements for tensor-parallel inference.
  Uses vLLM-style estimation approach.
  """

  def calculate(opts) do
    # Model dimensions
    hidden_size = Keyword.get(opts, :hidden_size, 2560)
    num_layers = Keyword.get(opts, :num_layers, 36)
    num_heads = Keyword.get(opts, :num_heads, 32)
    num_kv_heads = Keyword.get(opts, :num_kv_heads, 8)
    head_dim = Keyword.get(opts, :head_dim, 128)
    intermediate_size = Keyword.get(opts, :intermediate_size, 9728)
    vocab_size = Keyword.get(opts, :vocab_size, 151_936)

    # Runtime config
    batch_size = Keyword.get(opts, :batch_size, 1)
    max_seq_len = Keyword.get(opts, :max_seq_len, 128)
    tp_size = Keyword.get(opts, :tp_size, 4)
    bytes_per_param = Keyword.get(opts, :bytes_per_param, 4)

    # Calculate model parameters
    embedding_params = vocab_size * hidden_size

    per_layer_params =
      (hidden_size * num_heads * head_dim) +      # Q proj
      (hidden_size * num_kv_heads * head_dim) +   # K proj
      (hidden_size * num_kv_heads * head_dim) +   # V proj
      (num_heads * head_dim * hidden_size) +      # O proj
      (head_dim * 2) +                            # QK norm (Qwen3)
      (hidden_size * intermediate_size * 3) +     # FFN (gate, up, down)
      (hidden_size * 2)                           # Layer norms

    total_params = embedding_params + (per_layer_params * num_layers) + hidden_size

    # Memory calculations
    model_memory_gb = (total_params * bytes_per_param) / (1024 * 1024 * 1024)

    kv_cache_bytes = 2 * batch_size * max_seq_len * num_layers *
                     num_kv_heads * head_dim * bytes_per_param
    kv_cache_mb = kv_cache_bytes / (1024 * 1024)

    %{
      total_params_m: total_params / 1_000_000,
      model_memory_gb: model_memory_gb,
      kv_cache_mb: kv_cache_mb,
      per_gpu_gb: model_memory_gb / tp_size
    }
  end

  def recommended_memory_fraction(_opts \\ []) do
    # vLLM-style approach: use most of GPU memory
    # vLLM default is 90%, we use 85% for safety margin
    0.85
  end
end
```

## Why 85% Memory Fraction?

The default `memory_fraction: 0.85` is chosen based on:

### vLLM Approach (Reference)
- vLLM uses 90% as default (`gpu_memory_utilization=0.9`)
- Reserves 10% for CUDA context, PyTorch overhead, safety margin

### Our Approach
- Use 85% (slightly more conservative)
- XLA/EXLA has different memory patterns than PyTorch
- BFC allocator needs headroom for fragmentation

### Why Not Lower?
- Lower fractions waste GPU memory
- XLA pre-allocates to avoid runtime allocation overhead
- Pre-allocation improves performance consistency

```
Memory Fraction Comparison:

┌────────────┬───────────┬─────────────┬───────────────┐
│ Fraction   │ Per GPU   │ 4× H100     │ Notes         │
├────────────┼───────────┼─────────────┼───────────────┤
│ 0.50       │ 47 GB     │ 188 GB      │ Too wasteful  │
│ 0.85       │ 80 GB     │ 320 GB      │ Recommended   │
│ 0.90       │ 85 GB     │ 340 GB      │ Aggressive    │
│ 0.95       │ 89 GB     │ 356 GB      │ Risk of OOM   │
└────────────┴───────────┴─────────────┴───────────────┘
```

## KV Cache Architecture

### Pre-allocated vs Dynamic

**Choice:** Pre-allocated fixed-size cache

```elixir
# Pre-allocate for max_seq_len at startup
k_cache_shape = {batch, local_kv_heads, max_seq_len, head_dim}
v_cache_shape = {batch, local_kv_heads, max_seq_len, head_dim}

# Update at position (O(1) operation)
k_cache = Value.dynamic_update_slice(
  k_cache_in,
  k_new,
  [0, 0, position, 0],
  k_cache_typespec
)
```

**Why pre-allocated:**
- Single SPMD compilation (no recompile per sequence length)
- O(1) cache updates via `dynamic_update_slice`
- Predictable memory usage

**Trade-off:**
- Memory allocated for max_seq_len even if unused
- For variable workloads, PagedAttention (future work) is more efficient

### KV Cache Size Scaling

```
KV Cache Formula:
  2 × batch × seq_len × layers × kv_heads × head_dim × bytes

Qwen3-4B (8 KV heads, 36 layers, head_dim=128, f32):
┌───────────┬────────────┬──────────────┐
│ seq_len   │ Total      │ Per GPU (TP4)│
├───────────┼────────────┼──────────────┤
│ 128       │ 36 MB      │ 9 MB         │
│ 512       │ 144 MB     │ 36 MB        │
│ 2,048     │ 576 MB     │ 144 MB       │
│ 8,192     │ 2.3 GB     │ 576 MB       │
│ 32,768    │ 9.2 GB     │ 2.3 GB       │
└───────────┴────────────┴──────────────┘
```

### Position-Based Attention Masking

With pre-allocated cache, we use position-based masking:

```elixir
# Build causal mask for current position
# Only attend to positions [0, current_position]
build_causal_mask = fn seq_len, current_pos ->
  # Valid positions: 0 to current_pos
  # Mask with -inf for positions > current_pos
  row_indices = Nx.iota({seq_len}) |> Nx.reshape({1, seq_len})
  valid_mask = Nx.less_equal(row_indices, current_pos)
  Nx.select(valid_mask, 0.0, -1.0e9)
end
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEM_FRAC` | 0.85 | GPU memory fraction to allocate |
| `MAX_SEQ` | 128 | Pre-allocated KV cache length |
| `BATCH` | 1 | Batch size |

```bash
# Conservative allocation for debugging
MEM_FRAC=0.5 LAYERS=4 mix run examples/tp_4gpu_qwen3.exs

# Full model with longer context
MEM_FRAC=0.9 MAX_SEQ=512 LAYERS=36 mix run examples/tp_4gpu_qwen3.exs
```

## Troubleshooting OOM Errors

### Common Errors

**1. XLA Allocation Failure**
```
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 49807360 bytes
```

**Causes:**
- Memory fraction too low
- Max sequence length too high
- Model too large for available VRAM

**Fixes:**
```bash
# Increase memory fraction
MEM_FRAC=0.9 mix run examples/tp_4gpu_qwen3.exs

# Reduce sequence length
MAX_SEQ=64 mix run examples/tp_4gpu_qwen3.exs

# Use fewer layers for testing
LAYERS=8 mix run examples/tp_4gpu_qwen3.exs
```

**2. NCCL Initialization Failure**
```
NCCL WARN Bootstrap : no socket interface found
```

**Cause:** Multi-GPU communication setup issue

**Fix:** Ensure all GPUs are visible and NCCL is properly installed

**3. Compilation OOM**
```
XLA compilation failed: out of memory
```

**Cause:** XLA needs memory for compilation graphs

**Fix:**
- Reduce model complexity
- Increase memory fraction temporarily during compilation
- Use `EXLA_TARGET=host` for CPU-only compilation testing

### Memory Debugging

```elixir
# Check GPU memory before/after operations
defp log_gpu_memory(label) do
  # nvidia-smi approach
  {output, _} = System.cmd("nvidia-smi", [
    "--query-gpu=memory.used,memory.total",
    "--format=csv,noheader,nounits"
  ])
  IO.puts("#{label}: #{String.trim(output)}")
end
```

### XLA Memory Allocator

XLA uses a BFC (Best-Fit with Coalescing) allocator:

```
BFC Allocator Behavior:
┌─────────────────────────────────────────────────────────────┐
│  Total GPU Memory: 94 GB (H100)                             │
├─────────────────────────────────────────────────────────────┤
│  memory_fraction: 0.85 → Allocates 80 GB upfront            │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ BFC Pool (80 GB)                                     │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────────────┐ │   │
│  │  │ Model  │ │ KV     │ │ Acts   │ │ Free (coalesce)│ │   │
│  │  │ 4 GB   │ │ 9 MB   │ │ 10 MB  │ │ ~76 GB         │ │   │
│  │  └────────┘ └────────┘ └────────┘ └────────────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Reserved (14 GB): CUDA context, NCCL, system               │
└─────────────────────────────────────────────────────────────┘
```

**Key behaviors:**
- Pre-allocation avoids runtime cudaMalloc calls
- Coalescing reduces fragmentation
- Unused memory stays in pool (not returned to system)

## Memory Optimization Tips

### 1. Use BF16/FP16 for Weights
```elixir
# Half precision: 2x memory reduction
{:ok, model} = Bumblebee.load_model({:hf, model_id}, type: :bf16)
```

### 2. Reduce Max Sequence Length
```bash
# Only allocate what you need
MAX_SEQ=64 mix run examples/tp_4gpu_qwen3.exs
```

### 3. Right-size Batch
```bash
# Single batch for interactive use
BATCH=1 mix run examples/tp_4gpu_qwen3.exs

# Larger batch for throughput
BATCH=4 MAX_SEQ=256 mix run examples/tp_4gpu_qwen3.exs
```

### 4. Monitor During Development
```bash
# Watch GPU memory in real-time
watch -n 1 nvidia-smi
```

## Summary

| Aspect | Recommendation |
|--------|----------------|
| Memory fraction | 0.85 (vLLM-style) |
| KV cache | Pre-allocated, position-masked |
| Sequence length | Start small (128), scale up as needed |
| Debugging | Use MEM_FRAC=0.5 + few layers first |
| Production | MEM_FRAC=0.9 for maximum utilization |
