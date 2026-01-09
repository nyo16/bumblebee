# Future Roadmap

This document outlines planned optimizations and features for the tensor parallelism implementation.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION ROADMAP                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1: Flash Attention ─────────────────── HIGH PRIORITY │
│  ├── cuDNN integration                                       │
│  ├── Memory: O(seq²) → O(seq)                               │
│  └── Speed: 1.5-2x improvement                               │
│                                                              │
│  Phase 2: Quantization ────────────────────── HIGH PRIORITY │
│  ├── INT8 weights (2x memory)                                │
│  ├── FP8 on H100 (native support)                           │
│  └── Mixed precision inference                               │
│                                                              │
│  Phase 3: Continuous Batching ─────────────── MED PRIORITY  │
│  ├── Dynamic batch management                                │
│  ├── Iteration-level scheduling                              │
│  └── PagedAttention for KV cache                             │
│                                                              │
│  Phase 4: Multi-Node ──────────────────────── MED PRIORITY  │
│  ├── Pipeline parallelism (PP)                               │
│  ├── Hybrid TP + PP                                          │
│  └── InfiniBand/RoCE support                                 │
│                                                              │
│  Phase 5: Speculative Decoding ────────────── LOW PRIORITY  │
│  ├── Draft model approach                                    │
│  ├── Self-speculative                                        │
│  └── 2-3x throughput improvement                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Phase 1: Flash Attention

**Status:** Planned
**Priority:** High
**Expected gain:** 2-4x memory reduction, 1.5x speed

Flash Attention eliminates the O(seq²) memory bottleneck by computing attention tile-by-tile without materializing the full attention matrix.

### Why It Matters

```
Memory comparison (batch=1, heads=32, seq=4096):

Standard Attention:
  Attention scores: batch × heads × seq × seq × 4 bytes
                  = 1 × 32 × 4096 × 4096 × 4 = 2 GB per layer!

Flash Attention:
  Tiled computation: tile_size² at a time
                   = 256 × 256 × 4 = 256 KB per tile
```

### Implementation Options

1. **cuDNN Flash Attention** (Recommended)
   - NVIDIA-optimized, production-ready
   - Requires cuDNN 8.9+
   - Add NIF bindings to EXLA

2. **Custom MLIR Implementation**
   - StableHLO while_loop for tile iteration
   - Online softmax with log-sum-exp

3. **Triton Kernel**
   - Write Triton kernel, compile to PTX
   - Call via EXLA custom_call

See [06_flash_attention.md](./06_flash_attention.md) for detailed implementation plan.

## Phase 2: Quantization

**Status:** Research
**Priority:** High
**Expected gain:** 2-4x memory reduction, 1.3x speed

### INT8 Quantization

```elixir
# Load model with INT8 weights
{:ok, model} = Bumblebee.load_model({:hf, model_id}, type: :int8)

# Memory comparison (Qwen3-4B):
# FP32: 15 GB
# BF16: 7.5 GB
# INT8: 3.75 GB
```

**Implementation approach:**
- Weight-only quantization (simple, good quality)
- Per-channel scales for better accuracy
- Keep activations in BF16/FP16

### FP8 on H100

H100 has native FP8 support with Transformer Engine:

```
FP8 formats:
  E4M3: Higher precision, ±448 range
  E5M2: Lower precision, ±57344 range

Performance:
  - 2x compute vs FP16 on H100 tensor cores
  - Native support in cuDNN 8.9+
```

### GPTQ/AWQ Integration

```elixir
# Load pre-quantized model
{:ok, model} = Bumblebee.load_model(
  {:hf, "Qwen/Qwen3-4B-Instruct-GPTQ-Int4"},
  quantization: :gptq
)
```

## Phase 3: Continuous Batching

**Status:** Planned
**Priority:** Medium
**Expected gain:** 2-3x throughput

### Current Limitation

```
Static batching:
  - All sequences in batch must have same length
  - If one sequence finishes early, GPU waits
  - Inefficient for variable-length requests

┌─────────────────────────────────────────────────────────────┐
│ Static Batch (current)                                       │
├─────────────────────────────────────────────────────────────┤
│ Request 1: ████████████████████░░░░░░░░░░░░ (finished)      │
│ Request 2: ██████████████████████████████████ (still going) │
│ Request 3: ████████████████████████░░░░░░░░░ (finished)     │
│                                                              │
│ GPU cycles wasted on finished requests ░░░░░                 │
└─────────────────────────────────────────────────────────────┘
```

### Continuous Batching Solution

```
┌─────────────────────────────────────────────────────────────┐
│ Continuous Batching                                          │
├─────────────────────────────────────────────────────────────┤
│ Time →                                                       │
│ Iteration 1: [Req1, Req2, Req3, Req4]                       │
│ Iteration 2: [Req1, Req2, Req3, Req5] ← Req4 done, Req5 in  │
│ Iteration 3: [Req2, Req5, Req6, Req7] ← Req1,3 done         │
│                                                              │
│ GPU always full! No wasted cycles.                          │
└─────────────────────────────────────────────────────────────┘
```

### PagedAttention for KV Cache

```elixir
# Current: Contiguous pre-allocated cache
k_cache = Nx.tensor({batch, heads, max_seq, head_dim})

# Future: Paged cache (like vLLM)
defmodule PagedKVCache do
  defstruct [:block_tables, :blocks, :block_size]

  def allocate_block(cache, request_id) do
    # Find free block, assign to request
  end

  def free_blocks(cache, request_id) do
    # Return blocks to pool when request completes
  end
end
```

## Phase 4: Multi-Node Scaling

**Status:** Research
**Priority:** Medium
**Expected gain:** Scale to larger models (70B+)

### Pipeline Parallelism (PP)

```
PP=2 across nodes:
┌─────────────────────────────────────────────────────────────┐
│ Node 0 (Layers 0-17)     │ Node 1 (Layers 18-35)           │
│ ┌─────────────────────┐  │ ┌─────────────────────┐         │
│ │ GPU0 │ GPU1 │ GPU2 │ GPU3│ │ GPU0 │ GPU1 │ GPU2 │ GPU3│  │
│ │ TP=4 within node    │  │ │ TP=4 within node    │         │
│ └─────────────────────┘  │ └─────────────────────┘         │
│          │               │          ▲                       │
│          └───────────────┼──────────┘                       │
│              InfiniBand (send activations)                  │
└─────────────────────────────────────────────────────────────┘
```

### Hybrid TP + PP

```
Configuration for 70B model on 8 nodes (32 GPUs):
  - TP=4 within each node (fast NVLink)
  - PP=8 across nodes (InfiniBand)

Layer distribution:
  Node 0: Layers 0-9   (TP=4)
  Node 1: Layers 10-19 (TP=4)
  ...
  Node 7: Layers 70-79 (TP=4)
```

### Implementation Considerations

```elixir
# Multi-node SPMD configuration
config = [
  num_nodes: 8,
  gpus_per_node: 4,
  tp_size: 4,  # Within node
  pp_size: 8,  # Across nodes
  interconnect: :infiniband
]

# Pipeline schedule (1F1B)
defmodule PipelineScheduler do
  def schedule_microbatches(num_microbatches, num_stages) do
    # 1F1B: 1 Forward, 1 Backward interleaved
    # Minimizes pipeline bubbles
  end
end
```

## Phase 5: Speculative Decoding

**Status:** Research
**Priority:** Low (after other optimizations)
**Expected gain:** 2-3x throughput

### Concept

```
Standard decoding: Generate one token at a time
  Iteration 1: "The"
  Iteration 2: "capital"
  Iteration 3: "of"
  ...

Speculative decoding: Draft multiple tokens, verify in parallel
  Draft model: "The capital of France is Paris"
  Main model: Verify all 6 tokens in ONE forward pass
  If 5/6 accepted: 5x speedup for that batch!
```

### Implementation Approach

```elixir
defmodule SpeculativeDecoder do
  def decode(main_model, draft_model, prompt, opts) do
    k = opts[:speculation_length] || 4

    loop(prompt, []) do
      # 1. Draft k tokens with small model
      draft_tokens = draft_k_tokens(draft_model, current_tokens, k)

      # 2. Verify all k+1 tokens with main model (single forward)
      logits = forward_all(main_model, current_tokens ++ draft_tokens)

      # 3. Accept tokens that match draft distribution
      accepted = verify_and_accept(logits, draft_tokens)

      # 4. Continue from last accepted position
      current_tokens ++ accepted
    end
  end
end
```

### Self-Speculative Decoding

Use the same model with early exit for drafting:

```
Full model: 36 layers (slow, accurate)
Draft mode: First 8 layers only (fast, approximate)

No separate draft model needed!
```

## Summary

| Phase | Feature | Memory | Speed | Complexity |
|-------|---------|--------|-------|------------|
| 1 | Flash Attention | 2-4x ↓ | 1.5x ↑ | High |
| 2 | INT8 Quantization | 2x ↓ | 1.3x ↑ | Medium |
| 2 | FP8 (H100) | 2x ↓ | 2x ↑ | Medium |
| 3 | Continuous Batching | - | 2x ↑ | High |
| 3 | PagedAttention | Variable | - | High |
| 4 | Multi-Node PP | Scale | Scale | Very High |
| 5 | Speculative Decoding | - | 2-3x ↑ | High |

## Timeline (Tentative)

```
Q1 2025:
  ├── Flash Attention (cuDNN integration)
  └── INT8 weight quantization

Q2 2025:
  ├── FP8 support on H100
  └── Basic continuous batching

Q3 2025:
  ├── PagedAttention
  └── Multi-node prototype

Q4 2025:
  ├── Production multi-node
  └── Speculative decoding
```

## Contributing

Interested in helping? Key areas:

1. **Flash Attention**: cuDNN NIF bindings
2. **Quantization**: INT8 kernel optimization
3. **Batching**: Iteration-level scheduler
4. **Testing**: Benchmarks and correctness tests

See the main Bumblebee contribution guide for details.
