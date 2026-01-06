# Tensor Parallelism Continuation Plan

**Goal**: Complete TP=4 text generation with Mistral 7B on 4 H100 GPUs

**Status**: FFN layer working, need full model + autoregressive generation

---

## What's Done ✅

### EXLA Infrastructure
- `EXLA.SPMD` module - build/run multi-replica executables
- `EXLA.MLIR.Value.all_reduce/5` - StableHLO all-reduce op
- `EXLA.Nx.all_reduce/3` - defn-compatible via `Expr.optional`
- `EXLA.Collective` - replica_groups helper

### Bumblebee Infrastructure
- `Bumblebee.Distributed` - load_model, mesh, serving (TP=1)
- `ShardedLoader` - infer_sharding, shard_tensor
- `TPLayers` - column_parallel_dense, row_parallel_dense, all_reduce
- `TPTransformer` - build_model (placeholder)

### Demos Working
- 4-GPU all-reduce via NCCL ✅
- Mistral 7B FFN layer with TP=4 ✅ (~744ms/layer)
- TinyLlama FFN layer with TP=4 ✅ (~228ms/layer)
- TP=1 text generation ✅

---

## Phase 1: Full Transformer Block (SPMD)

### Step 1.1: Add Attention Layer to SPMD Demo

**File**: `examples/tp_4gpu_attention.exs`

Build SPMD executable for attention:
- Column-parallel Q, K, V projections
- Local attention computation (each GPU has subset of heads)
- Row-parallel output projection + all-reduce

```
Input [batch, seq, hidden]
  │
  ├─ Q projection (column-parallel) → [batch, seq, local_heads * head_dim]
  ├─ K projection (column-parallel) → [batch, seq, local_heads * head_dim]
  ├─ V projection (column-parallel) → [batch, seq, local_heads * head_dim]
  │
  ▼
Attention (local per GPU, each has hidden/tp_size heads)
  │
  ▼
Output projection (row-parallel) → partial [batch, seq, hidden]
  │
  ▼
All-reduce → [batch, seq, hidden] (replicated)
```

### Step 1.2: Combine Attention + FFN

**File**: `examples/tp_4gpu_transformer_block.exs`

Single transformer block:
```
Input (replicated)
  │
  ├─ RMSNorm (replicated)
  ▼
Attention (TP) + all-reduce
  │
  ├─ Residual add
  ├─ RMSNorm (replicated)
  ▼
FFN (TP) + all-reduce
  │
  ├─ Residual add
  ▼
Output (replicated)
```

### Step 1.3: Stack All Layers

**File**: `examples/tp_4gpu_full_model.exs`

- Loop through all 32 transformer blocks
- Add embedding lookup (replicated)
- Add LM head (replicated or vocab-parallel)

---

## Phase 2: Parameter Loading for TP=4

### Step 2.1: Multi-Device Parameter Distribution

Currently params are loaded for device_id=0. Need:
- Load full params once
- Shard and distribute to each GPU
- Store as 4 separate param sets

**Update**: `Bumblebee.Distributed.load_model/2`

```elixir
def load_model(repository, opts) do
  # Load full params
  {:ok, %{params: full_params, spec: spec}} = Bumblebee.load_model(repository)

  # Create sharded params for each device
  replica_params = for device_id <- 0..(tp_size - 1) do
    apply_sharding(full_params, mesh, device_id, opts)
  end

  {:ok, %{
    params: replica_params,  # List of 4 param sets
    spec: spec,
    mesh: mesh
  }}
end
```

### Step 2.2: Flatten Params for SPMD Input

SPMD needs params as flat list of tensors. Create helper:

```elixir
def flatten_params_for_spmd(replica_params, layer_order) do
  # Returns list of lists: [[gpu0_param1, gpu0_param2, ...], [gpu1_param1, ...], ...]
end
```

---

## Phase 3: Autoregressive Generation

### Step 3.1: Single Token Forward Pass

**File**: `lib/bumblebee/distributed/spmd_generation.ex`

```elixir
defmodule Bumblebee.Distributed.SPMDGeneration do
  def build_forward(model_info, opts) do
    # Build SPMD executable for single token prediction
    # Input: token_ids, attention_mask, position_ids, kv_cache
    # Output: logits, updated_kv_cache
  end

  def generate(forward, tokenizer, prompt, opts) do
    # Autoregressive loop:
    # 1. Tokenize prompt
    # 2. Run prefill (process full prompt)
    # 3. Loop: predict next token, append, repeat until done
  end
end
```

### Step 3.2: KV Cache Management

For TP, KV cache is partitioned:
- Each GPU stores cache for its heads only
- No cross-GPU communication needed for cache

```elixir
def init_kv_cache(spec, batch_size, max_length, tp_size) do
  local_heads = div(spec.num_attention_heads, tp_size)

  for _layer <- 1..spec.num_blocks do
    %{
      key: Nx.broadcast(0.0, {batch_size, local_heads, max_length, head_dim}),
      value: Nx.broadcast(0.0, {batch_size, local_heads, max_length, head_dim})
    }
  end
end
```

### Step 3.3: Token Sampling

After forward pass, sample from logits:
- All GPUs have identical logits (after all-reduce in LM head if TP'd)
- Sample on CPU, broadcast back

---

## Phase 4: Integration & API

### Step 4.1: High-Level API

```elixir
# Load model distributed across 4 GPUs
mesh = Bumblebee.Distributed.mesh(4)
{:ok, model_info} = Bumblebee.Distributed.load_model(
  {:hf, "mistralai/Mistral-7B-v0.1"},
  mesh: mesh
)

# Build SPMD generator
generator = Bumblebee.Distributed.SPMDGeneration.build(
  model_info,
  tokenizer,
  generation_config,
  batch_size: 1,
  max_length: 512
)

# Generate text
result = Bumblebee.Distributed.SPMDGeneration.run(
  generator,
  "The capital of France is"
)
IO.puts(result.text)
```

### Step 4.2: Streaming Support

Hook into generation loop for streaming:

```elixir
Bumblebee.Distributed.SPMDGeneration.stream(generator, prompt)
|> Enum.each(fn token -> IO.write(token) end)
```

---

## Implementation Order

1. **Phase 1.1**: Attention layer SPMD demo
2. **Phase 1.2**: Full transformer block demo
3. **Phase 2.1**: Multi-device param distribution
4. **Phase 1.3**: Full model forward pass
5. **Phase 3.1**: Single token generation
6. **Phase 3.2**: KV cache integration
7. **Phase 4.1**: High-level API
8. **Phase 3.3**: Token sampling & full generation
9. **Phase 4.2**: Streaming

---

## Testing Milestones

| Milestone | Test |
|-----------|------|
| Attention layer | Output matches single-GPU |
| Transformer block | Output matches single-GPU |
| Full forward | Logits match single-GPU |
| Single token | Next token prediction works |
| Generation | Produces coherent text |
| Performance | Faster than single-GPU for batch>1 |

---

## Files to Create/Modify

### New Files
- `examples/tp_4gpu_attention.exs`
- `examples/tp_4gpu_transformer_block.exs`
- `examples/tp_4gpu_full_model.exs`
- `lib/bumblebee/distributed/spmd_generation.ex`

### Modify
- `lib/bumblebee/distributed.ex` - multi-device params
- `lib/bumblebee/distributed/sharded_loader.ex` - attention sharding
- `lib/bumblebee/distributed/tp_transformer.ex` - full SPMD model

---

## Estimated Complexity

| Phase | Complexity | Notes |
|-------|------------|-------|
| 1.1 Attention | Medium | Similar to FFN, more ops |
| 1.2 Block | Low | Combine existing |
| 1.3 Full model | Medium | Loop + embeddings |
| 2.1 Multi-device | Low | Iterate existing |
| 3.1 Forward | Medium | SPMD builder |
| 3.2 KV cache | High | State management |
| 4.1 API | Low | Wrapper |
| 3.3 Sampling | Medium | Decode loop |
