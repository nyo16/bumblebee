# KV Cache Implementation Progress

## Overview

KV cache optimization is critical for efficient autoregressive generation. Without it, each new token requires recomputing attention for all previous tokens (O(n²) complexity).

## Current Status

### ✓ Completed

1. **Basic Generation** (`examples/tp_4gpu_generate.exs`)
   - Autoregressive generation working with 4-GPU tensor parallelism
   - Generates text token-by-token
   - **Problem**: Rebuilds SPMD for each sequence length
   - **Problem**: Recomputes Q, K, V for all tokens each step

### ⧗ In Progress

2. **KV Cache Prefill Phase** (`examples/tp_4gpu_generate_kvcache.exs`)
   - SPMD builder for prefill phase implemented
   - Takes prompt tokens, outputs logits + K/V cache for all layers
   - Cache shape: `[batch, local_kv_heads, seq_len, head_dim]` per layer per GPU
   - **Blocked**: Model loading issues with TinyLlama cache

### ☐ Remaining

3. **KV Cache Decode Phase**
   - Accept single token + previous K/V cache as inputs
   - Concatenate new K/V to cache
   - Compute attention only for new token position
   - Return logits + updated cache

4. **Full Generation Loop with Cache**
   - Prefill: Process prompt, initialize cache
   - Decode: Generate tokens one at a time using cache
   - Performance: O(n) per token instead of O(n²)

## Technical Design

### Prefill Phase
```
Input:  [batch, prompt_len] token IDs
Process: Full forward pass through all prompt tokens
Output:
  - Logits [batch, vocab_size] for next token
  - K cache [batch, local_kv_heads, prompt_len, head_dim] per layer
  - V cache [batch, local_kv_heads, prompt_len, head_dim] per layer
```

### Decode Phase
```
Input:
  - [batch, 1] - single new token
  - K cache [batch, local_kv_heads, cache_len, head_dim] per layer
  - V cache [batch, local_kv_heads, cache_len, head_dim] per layer
Process:
  - Compute Q, K, V only for new token
  - Concatenate new K/V to cache
  - Attention: Q_new @ K_full^T (1 x cache_len)
Output:
  - Logits [batch, vocab_size]
  - Updated K cache [batch, local_kv_heads, cache_len+1, head_dim]
  - Updated V cache [batch, local_kv_heads, cache_len+1, head_dim]
```

### Tensor Parallelism Considerations

- Each GPU stores its local K/V heads (`local_kv_heads = num_kv_heads / tp_size`)
- No all-reduce needed for cache (local to each GPU)
- All-reduce only needed for output projection
- Cache memory: `2 * num_layers * batch * local_kv_heads * max_seq_len * head_dim * 4 bytes`
  - Example: 4 layers, 1 batch, 1 local head (4 GPUs), 512 max len, 64 head dim = 1MB per GPU

## Performance Impact

### Without KV Cache (Current)
- Token 0: Process 1 token
- Token 1: Process 2 tokens (recompute token 0)
- Token 2: Process 3 tokens (recompute tokens 0, 1)
- Token n: Process n+1 tokens
- **Total**: O(n²) computation

### With KV Cache (Target)
- Prefill: Process prompt_len tokens once
- Token 0: Process 1 token with cached context
- Token 1: Process 1 token with cached context
- Token n: Process 1 token with cached context
- **Total**: O(n) computation per token

### Expected Speedup
For generating 100 tokens with 50-token prompt:
- Without cache: ~5,050 token computations
- With cache: ~150 token computations
- **Speedup**: ~33x faster

## Next Steps

1. Resolve TinyLlama model loading issue (checkpoint vs model format)
2. Complete decode phase SPMD implementation
3. Implement concatenation operations in MLIR for cache update
4. Add generation loop that uses prefill/decode phases
5. Benchmark: Compare with/without cache performance
6. Test on longer sequences (50-200 tokens)

## Files

- `examples/tp_4gpu_generate.exs` - Basic generation (no cache)
- `examples/tp_4gpu_generate_kvcache.exs` - KV cache implementation (in progress)
- `docs/KV_CACHE_PROGRESS.md` - This file
