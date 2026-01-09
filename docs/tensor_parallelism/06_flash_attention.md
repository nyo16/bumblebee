# Flash Attention Implementation Plan

This document provides a detailed plan for implementing Flash Attention in EXLA.

## What is Flash Attention?

Flash Attention is an IO-aware exact attention algorithm that:
- Reduces memory from O(seq²) to O(seq)
- Improves speed by 2-4x through better memory access patterns
- Produces mathematically identical results to standard attention

### The Problem with Standard Attention

```
Standard attention computation:

1. S = Q @ K^T           # Shape: [batch, heads, seq, seq]
2. P = softmax(S / √d)   # Shape: [batch, heads, seq, seq]  ← Memory hog!
3. O = P @ V             # Shape: [batch, heads, seq, head_dim]

Memory for scores matrix (batch=1, heads=32, seq=4096):
  32 × 4096 × 4096 × 4 bytes = 2 GB per layer!

For 36 layers: 72 GB just for attention scores!
```

### The Flash Attention Solution

```
Instead of materializing full seq×seq matrix:
  - Process in tiles (e.g., 256×256)
  - Compute softmax incrementally (online softmax)
  - Never store full attention matrix

Memory: O(seq) instead of O(seq²)
  256 × 256 × 4 bytes = 256 KB per tile
```

## Algorithm Deep Dive

### Online Softmax

The key insight is computing softmax incrementally:

```
Standard softmax:
  softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

  Problem: Need full row to compute max and sum

Online softmax (incremental):
  Process tiles, maintain running max and sum

  For each tile j:
    1. Compute local max: m_j = max(scores_j)
    2. Update global max: m_new = max(m_old, m_j)
    3. Rescale previous accumulator:
       scale = exp(m_old - m_new)
       out = out × scale
       sum = sum × scale
    4. Add new contribution:
       weights_j = exp(scores_j - m_new)
       out += weights_j @ V_j
       sum += sum(weights_j)
    5. Update: m_old = m_new

  Final: out = out / sum
```

### Tiling Strategy

```
Q matrix [seq_q, head_dim]     K^T matrix [head_dim, seq_k]
┌─────────────────────────┐    ┌─────────────────────────┐
│ Q₀  │ Q₁  │ Q₂  │ Q₃   │    │ K₀  │ K₁  │ K₂  │ K₃   │
│(256)│(256)│(256)│(256) │    │(256)│(256)│(256)│(256) │
└─────────────────────────┘    └─────────────────────────┘

Processing order (for causal attention):
┌─────────────────────────────────────────┐
│     K₀    K₁    K₂    K₃               │
│   ┌─────┬─────┬─────┬─────┐            │
│Q₀ │ ██  │     │     │     │ ← Only K₀  │
│   ├─────┼─────┼─────┼─────┤            │
│Q₁ │ ██  │ ██  │     │     │ ← K₀, K₁   │
│   ├─────┼─────┼─────┼─────┤            │
│Q₂ │ ██  │ ██  │ ██  │     │ ← K₀-K₂    │
│   ├─────┼─────┼─────┼─────┤            │
│Q₃ │ ██  │ ██  │ ██  │ ██  │ ← All K    │
│   └─────┴─────┴─────┴─────┘            │
│                                         │
│ Causal mask: only process lower triangle│
└─────────────────────────────────────────┘
```

### Pseudocode

```python
def flash_attention(Q, K, V, block_size=256, causal=True):
    """
    Q, K, V: [batch, heads, seq, head_dim]
    Returns: O [batch, heads, seq, head_dim]
    """
    batch, heads, seq_len, head_dim = Q.shape
    num_blocks = ceil(seq_len / block_size)

    # Output accumulators
    O = zeros(batch, heads, seq_len, head_dim)
    L = zeros(batch, heads, seq_len)  # log-sum-exp
    M = full(batch, heads, seq_len, -inf)  # running max

    for q_block_idx in range(num_blocks):
        q_start = q_block_idx * block_size
        q_end = min(q_start + block_size, seq_len)
        Q_block = Q[:, :, q_start:q_end, :]

        # For causal: only attend to keys up to current position
        k_end_block = q_block_idx + 1 if causal else num_blocks

        for k_block_idx in range(k_end_block):
            k_start = k_block_idx * block_size
            k_end = min(k_start + block_size, seq_len)
            K_block = K[:, :, k_start:k_end, :]
            V_block = V[:, :, k_start:k_end, :]

            # Compute attention scores for this tile
            S_block = Q_block @ K_block.T / sqrt(head_dim)

            # Apply causal mask within block if needed
            if causal and q_block_idx == k_block_idx:
                # Mask upper triangle within this block
                mask = triu(ones(q_end-q_start, k_end-k_start), k=1)
                S_block = S_block.masked_fill(mask, -inf)

            # Online softmax update
            M_block_old = M[:, :, q_start:q_end]
            M_block_new = maximum(M_block_old, S_block.max(dim=-1))

            # Rescale previous output
            scale = exp(M_block_old - M_block_new)
            O[:, :, q_start:q_end, :] *= scale.unsqueeze(-1)
            L[:, :, q_start:q_end] *= scale

            # Add new contribution
            P_block = exp(S_block - M_block_new.unsqueeze(-1))
            O[:, :, q_start:q_end, :] += P_block @ V_block
            L[:, :, q_start:q_end] += P_block.sum(dim=-1)

            # Update running max
            M[:, :, q_start:q_end] = M_block_new

    # Final normalization
    O = O / L.unsqueeze(-1)
    return O
```

## Implementation Options

### Option A: cuDNN Flash Attention (Recommended)

**Approach:** Use NVIDIA's cuDNN 8.9+ Flash Attention via NIF bindings

```
Pros:
  ✓ Production-ready, heavily optimized
  ✓ Supports H100 tensor cores
  ✓ Handles edge cases (variable seq len, masking)
  ✓ Forward + backward (for training)

Cons:
  ✗ NVIDIA-only (no AMD/TPU)
  ✗ Requires cuDNN 8.9+
  ✗ Black box (harder to debug)
```

**Implementation steps:**

1. Add cuDNN NIF to EXLA
```c
// c_src/exla/exla_cudnn.cc
#include <cudnn.h>

ERL_NIF_TERM flash_attention_forward(ErlNifEnv* env, ...) {
    cudnnAttnDescriptor_t attn_desc;
    cudnnCreateAttnDescriptor(&attn_desc);

    cudnnSetAttnDescriptor(attn_desc,
        CUDNN_ATTN_QUERYMAP_ALL_TO_ONE,  // Standard attention
        num_heads,
        1.0f / sqrt(head_dim),           // Scale
        CUDNN_DATA_FLOAT,                // Data type
        ...
    );

    cudnnMultiHeadAttnForward(handle, attn_desc, ...);
}
```

2. Create Elixir wrapper
```elixir
# lib/exla/cudnn.ex
defmodule EXLA.cuDNN do
  @on_load :load_nif

  def flash_attention(q, k, v, opts \\ []) do
    causal = Keyword.get(opts, :causal, true)
    dropout = Keyword.get(opts, :dropout, 0.0)

    nif_flash_attention(q, k, v, causal, dropout)
  end

  defp nif_flash_attention(_q, _k, _v, _causal, _dropout) do
    :erlang.nif_error(:not_loaded)
  end
end
```

3. Wire into SPMD builder
```elixir
# In attention function:
output = Value.custom_call(
  "flash_attention",
  [q, k, v],
  output_typespec,
  backend_config: %{
    causal: true,
    block_size: 256
  }
)
```

### Option B: Custom MLIR Implementation

**Approach:** Implement tiled attention in StableHLO

```
Pros:
  ✓ Works on any XLA backend (GPU, TPU)
  ✓ No external dependencies
  ✓ Full control over algorithm

Cons:
  ✗ Complex to implement correctly
  ✗ Won't match cuDNN performance
  ✗ Harder to optimize
```

**Implementation steps:**

1. Build tile iteration with while_loop
```elixir
def flash_attention_mlir(builder, q, k, v, opts) do
  block_size = opts[:block_size] || 256
  seq_len = opts[:seq_len]
  num_blocks = div(seq_len + block_size - 1, block_size)

  # Initialize outputs
  output = Value.constant(builder, zeros(output_shape), output_typespec)
  running_max = Value.constant(builder, neg_inf(max_shape), max_typespec)
  running_sum = Value.constant(builder, zeros(sum_shape), sum_typespec)

  # Outer loop over Q blocks
  {output, running_max, running_sum} = Value.while_loop(
    builder,
    {output, running_max, running_sum, q_block_idx = 0},
    fn {out, m, s, q_idx} ->
      Value.less(q_idx, num_blocks)
    end,
    fn {out, m, s, q_idx} ->
      # Extract Q block
      q_block = Value.dynamic_slice(q, [0, 0, q_idx * block_size, 0],
                                    [batch, heads, block_size, head_dim])

      # Inner loop over K blocks (up to current block for causal)
      {out_new, m_new, s_new} = inner_loop(builder, q_block, k, v, q_idx, out, m, s)

      {out_new, m_new, s_new, Value.add(q_idx, 1)}
    end
  )

  # Final normalization
  Value.divide(output, Value.broadcast(running_sum))
end
```

2. Implement online softmax in inner loop
```elixir
defp inner_loop(builder, q_block, k, v, q_block_idx, out, m_old, s_old) do
  Value.while_loop(
    builder,
    {out, m_old, s_old, k_block_idx = 0},
    fn {_, _, _, k_idx} ->
      # Causal: k_idx <= q_idx
      Value.less_equal(k_idx, q_block_idx)
    end,
    fn {out, m, s, k_idx} ->
      # Extract K, V blocks
      k_block = Value.dynamic_slice(k, ...)
      v_block = Value.dynamic_slice(v, ...)

      # Compute scores: Q @ K^T / sqrt(d)
      scores = Value.dot_general(q_block, k_block, ...)
      scores = Value.multiply(scores, scale)

      # Apply causal mask if on diagonal block
      scores = apply_causal_mask(builder, scores, q_block_idx, k_idx)

      # Online softmax update
      m_new = Value.maximum(m, Value.reduce_max(scores, axis: -1))
      scale_factor = Value.exp(Value.subtract(m, m_new))

      out_scaled = Value.multiply(out, scale_factor)
      s_scaled = Value.multiply(s, scale_factor)

      weights = Value.exp(Value.subtract(scores, m_new))
      out_new = Value.add(out_scaled, Value.dot_general(weights, v_block, ...))
      s_new = Value.add(s_scaled, Value.reduce_sum(weights, axis: -1))

      {out_new, m_new, s_new, Value.add(k_idx, 1)}
    end
  )
end
```

### Option C: Triton Kernel via Custom Op

**Approach:** Write Triton kernel, compile to PTX, call from EXLA

```
Pros:
  ✓ Good performance (close to cuDNN)
  ✓ Readable kernel code
  ✓ Easy to modify/debug

Cons:
  ✗ Requires Triton setup
  ✗ NVIDIA-only
  ✗ Additional build dependency
```

**Triton kernel example:**

```python
# flash_attention_kernel.py
import triton
import triton.language as tl

@triton.jit
def flash_attention_kernel(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    seq_len, head_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ID
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulators
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    # Load Q block (stays in registers)
    q = tl.load(Q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)

    # Iterate over K, V blocks
    for start_n in range(0, (pid_m + 1) * BLOCK_M, BLOCK_N):
        # Load K block
        k = tl.load(K + (start_n + offs_n[:, None]) * stride_kn + offs_k[None, :] * stride_kk)

        # Compute QK^T
        qk = tl.dot(q, tl.trans(k))
        qk *= 1.0 / tl.sqrt(float(head_dim))

        # Causal mask
        mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = tl.where(mask, qk, -float('inf'))

        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)

        l_i = l_i * alpha + tl.sum(tl.exp(qk - m_new[:, None]), axis=1)

        # Load V and update output
        v = tl.load(V + (start_n + offs_n[:, None]) * stride_vn + offs_k[None, :] * stride_vk)
        p = tl.exp(qk - m_new[:, None])
        o_i = o_i * alpha[:, None] + tl.dot(p, v)

        m_i = m_new

    # Final normalization and store
    o_i = o_i / l_i[:, None]
    tl.store(O + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok, o_i)
```

**Integration with EXLA:**

```elixir
# Compile Triton to PTX
def compile_flash_kernel do
  System.cmd("python", ["compile_triton.py", "--kernel", "flash_attention"])
end

# Call via custom_call
def flash_attention_triton(builder, q, k, v) do
  Value.custom_call(
    "flash_attention_triton",
    [q, k, v],
    output_typespec,
    target: "ptx",
    kernel_path: "flash_attention.ptx"
  )
end
```

## Integration with Current SPMD

Current attention code:
```elixir
# Standard attention (O(seq²) memory)
scores = Value.dot_general(q, k_transposed, ...)  # [batch, heads, seq, seq]
scores_scaled = Value.multiply(scores, scale, ...)
scores_masked = Value.add(scores_scaled, mask, ...)
attn_weights = softmax(scores_masked)              # Still [batch, heads, seq, seq]
output = Value.dot_general(attn_weights, v, ...)
```

With Flash Attention:
```elixir
# Flash Attention (O(seq) memory)
output = flash_attention(builder, q, k, v,
  causal: true,
  block_size: 256
)
# No intermediate seq×seq tensor!
```

### Handling KV Cache

Flash Attention with incremental decoding:

```elixir
def flash_attention_with_cache(builder, q, k_cache, v_cache, new_k, new_v, position) do
  # Update cache first
  k_cache_updated = Value.dynamic_update_slice(k_cache, new_k, [0, 0, position, 0])
  v_cache_updated = Value.dynamic_update_slice(v_cache, new_v, [0, 0, position, 0])

  # Flash attention on full cache up to position
  # For decode: Q is single token, K/V is full cache
  output = flash_attention_decode(builder, q, k_cache_updated, v_cache_updated,
    seq_len: position + 1,
    causal: true
  )

  {output, k_cache_updated, v_cache_updated}
end
```

## Performance Expectations

### Memory Reduction

```
Sequence length: 4096, heads: 32, batch: 1

Standard Attention:
  Scores matrix: 32 × 4096 × 4096 × 4 = 2.0 GB per layer
  36 layers: 72 GB just for attention!

Flash Attention (block_size=256):
  Per tile: 32 × 256 × 256 × 4 = 8 MB
  Total: ~8 MB (reused across tiles)
  Savings: 250x memory reduction
```

### Speed Improvement

```
Expected speedup on H100:

┌────────────┬───────────────┬───────────────┬─────────┐
│ Seq Length │ Standard      │ Flash         │ Speedup │
├────────────┼───────────────┼───────────────┼─────────┤
│ 512        │ 0.5ms         │ 0.4ms         │ 1.25x   │
│ 2048       │ 4ms           │ 2ms           │ 2x      │
│ 4096       │ 16ms          │ 5ms           │ 3.2x    │
│ 8192       │ 64ms          │ 12ms          │ 5.3x    │
│ 16384      │ OOM           │ 30ms          │ ∞       │
└────────────┴───────────────┴───────────────┴─────────┘

Note: Standard attention OOMs at longer sequences!
```

## Recommended Implementation Path

```
Phase 1: cuDNN Integration (2-3 weeks)
├── Add cuDNN NIF bindings to EXLA
├── Implement EXLA.cuDNN.flash_attention/4
├── Wire into SPMD builder as custom_call
└── Test correctness vs standard attention

Phase 2: MLIR Fallback (2-3 weeks)
├── Implement basic tiled attention in StableHLO
├── Add online softmax with while_loop
├── Used when cuDNN unavailable
└── Works on TPU/CPU

Phase 3: Optimization (1-2 weeks)
├── Tune block sizes for different seq lengths
├── Add GQA-specific optimizations
├── Profile and optimize memory access
└── Benchmark against vLLM/TensorRT-LLM
```

## Testing Strategy

### Correctness Tests

```elixir
defmodule FlashAttentionTest do
  use ExUnit.Case

  test "matches standard attention output" do
    q = Nx.random_uniform({1, 8, 128, 64})
    k = Nx.random_uniform({1, 8, 128, 64})
    v = Nx.random_uniform({1, 8, 128, 64})

    standard_output = standard_attention(q, k, v, causal: true)
    flash_output = flash_attention(q, k, v, causal: true)

    assert Nx.all_close?(standard_output, flash_output, atol: 1.0e-5)
  end

  test "handles causal masking correctly" do
    # Verify causal mask is applied correctly
  end

  test "handles variable sequence lengths" do
    # Test with different seq lengths
  end
end
```

### Performance Tests

```elixir
defmodule FlashAttentionBench do
  def run do
    for seq_len <- [512, 1024, 2048, 4096, 8192] do
      q = Nx.random_uniform({1, 32, seq_len, 128})
      k = Nx.random_uniform({1, 32, seq_len, 128})
      v = Nx.random_uniform({1, 32, seq_len, 128})

      # Warmup
      flash_attention(q, k, v, causal: true)

      # Benchmark
      {time_us, _} = :timer.tc(fn ->
        for _ <- 1..100, do: flash_attention(q, k, v, causal: true)
      end)

      IO.puts("seq_len=#{seq_len}: #{time_us / 100}µs per call")
    end
  end
end
```

## References

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135) - Original algorithm
- [Flash Attention 2](https://arxiv.org/abs/2307.08691) - Improved version
- [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/api/index.html) - NVIDIA API
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) - Triton implementation
