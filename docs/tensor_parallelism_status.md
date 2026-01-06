# Tensor Parallelism Implementation Status

## What Works ✅

### 1. EXLA SPMD Module (`EXLA.SPMD`)
- Multi-device SPMD execution with `num_replicas`
- Basic operations execute correctly on all 4 GPUs
- Tested: Each GPU runs same computation on different data

### 2. NCCL All-Reduce (4 GPUs)
```elixir
# Tested successfully:
GPU 0 input: [1.0, 0.0, 0.0, 0.0]
GPU 1 input: [0.0, 2.0, 0.0, 0.0]
GPU 2 input: [0.0, 0.0, 3.0, 0.0]
GPU 3 input: [0.0, 0.0, 0.0, 4.0]

# After all-reduce, all GPUs have:
[1.0, 2.0, 3.0, 4.0]
```

### 3. Parameter Sharding (`ShardedLoader`)
- Correctly identifies column-parallel vs row-parallel parameters
- FFN-only mode: Shards FFN, keeps attention replicated
- TP=4 example: FFN gate kernel {2048, 5632} → {2048, 1408}

### 4. TP Infrastructure
- `Bumblebee.Distributed` - Main API
- `ShardedLoader` - Parameter sharding logic
- `TPLayers` - Column/row parallel layer builders
- `TPTransformer` - TP model architecture

### 5. Single-GPU Execution (TP=1)
- Full text generation works correctly
- Model loads with proper parameter mapping
- Inference produces coherent output

## Current Gap 🔧

### All-Reduce Integration in Defn

The `TPLayers.all_reduce` is currently a pass-through:

```elixir
# Current (placeholder):
defnp all_reduce_impl(tensor, _opts, _op, _replica_groups) do
  tensor  # Just passes through
end
```

**Why**: EXLA's defn compiler doesn't have a built-in hook for SPMD collective operations.

**Impact**: With TP > 1, partial results from each GPU don't get summed, producing garbage output.

## Path Forward

### Option A: SPMD-Aware Serving (Recommended)

Create a new serving mode that:
1. Builds the model graph with Axon (current approach)
2. Wraps the forward pass in `EXLA.SPMD.build/4`
3. Runs with replica-batched inputs

```elixir
# Conceptual API:
serving = Bumblebee.Distributed.spmd_serving(
  model_info,
  tokenizer,
  generation_config,
  num_replicas: 4  # Uses SPMD mode
)
```

### Option B: Custom EXLA Op for All-Reduce

Add all-reduce as a recognized operation in `EXLA.Defn`:

1. Define `:all_reduce` as an Nx expression type
2. Add lowering rule in `EXLA.Defn.cached_recur_operator`
3. Emit `stablehlo.all_reduce` during MLIR building

### Option C: Nx.Defn Extension

Propose adding collective operations to Nx:
- `Nx.Defn.Collective.all_reduce/3`
- Backend-agnostic interface
- EXLA provides NCCL implementation

## Test Commands

```bash
# SPMD all-reduce (works):
mix run examples/spmd_4gpu_test.exs

# CPU SPMD test:
XLA_FLAGS="--xla_force_host_platform_device_count=4" mix run examples/spmd_cpu_test.exs

# TP=1 model (works):
mix run examples/tp_verify_model.exs

# TP=4 model (needs all-reduce fix):
mix run examples/tp_4gpu_llama.exs
```

## Files Modified/Created

### EXLA (nx/exla)
- `lib/exla/mlir/value.ex` - Added `all_reduce/5`
- `lib/exla/collective.ex` - High-level collective API
- `lib/exla/spmd.ex` - SPMD execution wrapper

### Bumblebee
- `lib/bumblebee/distributed.ex` - Main TP API
- `lib/bumblebee/distributed/sharded_loader.ex` - Parameter sharding
- `lib/bumblebee/distributed/tp_layers.ex` - TP layer builders
- `lib/bumblebee/distributed/tp_transformer.ex` - TP model architecture
- `examples/spmd_*.exs` - SPMD test scripts
- `examples/tp_*.exs` - TP demo scripts

## Summary

| Component | Status |
|-----------|--------|
| EXLA SPMD module | ✅ Working |
| NCCL all-reduce | ✅ Working (4 GPUs) |
| Parameter sharding | ✅ Working |
| TP layer definitions | ✅ Working |
| TP=1 inference | ✅ Working |
| TP>1 inference | ⏳ Needs all-reduce integration |

The infrastructure is complete. The remaining work is integrating SPMD execution into Bumblebee's serving pipeline.
