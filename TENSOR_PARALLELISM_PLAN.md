# Tensor Parallelism for Bumblebee - Complete Implementation Plan

> **Context**: This plan was created through extensive research into Nx, EXLA, Bumblebee, Axon, vLLM, SGLang, and llama.cpp. A future Claude instance should be able to continue implementation from this document.

---

## Table of Contents
1. [Project Goal](#project-goal)
2. [User Configuration](#user-configuration)
3. [Research Findings](#research-findings)
4. [Existing Work: PR #1646](#existing-work-pr-1646)
5. [Architecture Overview](#architecture-overview)
6. [Implementation Phases](#implementation-phases)
7. [Detailed Implementation Steps](#detailed-implementation-steps)
8. [Code Examples](#code-examples)
9. [Testing Strategy](#testing-strategy)
10. [File Reference](#file-reference)

---

## Project Goal

Implement **tensor parallelism** (TP) for LLM inference in Bumblebee, similar to vLLM/SGLang. This allows running large models like Mistral 7B across multiple GPUs by sharding model weights and using collective operations (all-reduce) for synchronization.

**Target**: Mistral 7B with TP=2 (2 GPUs)

---

## User Configuration

- **Use Case**: Inference only (no training)
- **First Model**: Mistral 7B
- **GPU Setup**: 2x GPUs available for testing
- **Starting Point**: Fork Paulo Valente's PR #1646 (`pv-feat/exla-sharding-poc`)
- **Integration**: EXLA-specific collective ops initially

---

## Research Findings

### 1. Current Nx/EXLA Capabilities

**What exists:**
- Device placement via `{EXLA.Backend, client: :cuda, device_id: N}`
- `Nx.Serving` with `partitions: true` for data parallelism (not tensor parallelism)
- `EXLA.Client.fetch!(:cuda).device_count` returns available GPUs
- SPMD compilation support via `num_partitions` option

**What's missing:**
- Collective operations (all-reduce, all-gather, reduce-scatter)
- Tensor sharding primitives for model parallelism
- Sharded parameter loading

**Key files in EXLA:**
- `deps/exla/lib/exla/client.ex` - Client with device_count
- `deps/exla/lib/exla/backend.ex` - Device placement logic
- `deps/exla/lib/exla/defn.ex` - `__partitions_options__/1` for multi-device
- `deps/exla/lib/exla/mlir/module.ex` - SPMD compilation options

### 2. Current Bumblebee Architecture

**Model Loading Pipeline:**
```
Bumblebee.load_model/2 (lib/bumblebee.ex:580-602)
    ↓
load_params/5 (lib/bumblebee.ex:621-645)
    ↓
PyTorchParams.load_params!/4 (lib/bumblebee/conversion/pytorch_params.ex:32-80)
    ↓
with_default_backend() wraps Nx operations
```

**Key Options:**
- `:backend` - specifies device for tensor allocation
- `:preallocate_params` - moves params to device at serving init

**Parameter Structure:**
```elixir
%Axon.ModelState{
  data: %{
    "encoder.blocks.0.self_attention.query.kernel" => tensor,
    "encoder.blocks.0.self_attention.key.kernel" => tensor,
    ...
  }
}
```

**Serving Pattern (lib/bumblebee/shared.ex:450-461):**
```elixir
def maybe_preallocate(params, preallocate?, defn_options) do
  if preallocate? do
    backend = Nx.Defn.to_backend(defn_options)
    Nx.backend_copy(params, backend)
  else
    params
  end
end
```

### 3. Transformer Layer Structure (lib/bumblebee/layers/transformer.ex)

**Multi-Head Attention (lines 713-892):**
```
Input: {batch, seq, hidden_size}
    ↓
Query Dense: {hidden_size, num_heads * head_size} → Column-parallel
Key Dense:   {hidden_size, num_heads * head_size} → Column-parallel
Value Dense: {hidden_size, num_heads * head_size} → Column-parallel
    ↓
split_heads: {batch, seq, num_heads, head_size}
    ↓
Attention computation (local, no communication)
    ↓
flatten_trailing: {batch, seq, num_heads * head_size}
    ↓
Output Dense: {num_heads * head_size, hidden_size} → Row-parallel + ALL-REDUCE
```

**Feed-Forward Network (Gated FFN for Llama/Mistral):**
```
Input: {batch, seq, hidden_size}
    ↓
Gate Dense:         {hidden_size, intermediate_size} → Column-parallel
Intermediate Dense: {hidden_size, intermediate_size} → Column-parallel
    ↓
SiLU activation + element-wise multiply
    ↓
Output Dense: {intermediate_size, hidden_size} → Row-parallel + ALL-REDUCE
```

**Total per transformer block: 2 all-reduce operations**

### 4. vLLM Tensor Parallelism Pattern

**Column-Parallel Linear:**
- Weight split along output dimension (axis 1)
- Input is replicated across all GPUs
- Output is partitioned
- NO communication needed

**Row-Parallel Linear:**
- Weight split along input dimension (axis 0)
- Input is partitioned (from prior column-parallel)
- Output needs all-reduce to sum partial results
- ALL-REDUCE after this layer

**Sharding Map for Llama/Mistral:**

| Layer | Weight Shape | Sharding | All-Reduce? |
|-------|-------------|----------|-------------|
| Q projection | {H, H} | Column (axis 1) | No |
| K projection | {H, H} | Column (axis 1) | No |
| V projection | {H, H} | Column (axis 1) | No |
| Attn output | {H, H} | Row (axis 0) | **Yes** |
| FFN gate | {H, I} | Column (axis 1) | No |
| FFN up | {H, I} | Column (axis 1) | No |
| FFN down | {I, H} | Row (axis 0) | **Yes** |

Where H = hidden_size, I = intermediate_size

### 5. Mistral 7B Specifications

```
hidden_size: 4096
num_attention_heads: 32
num_key_value_heads: 8 (GQA - grouped query attention)
head_size: 128 (4096 / 32)
intermediate_size: 14336
num_hidden_layers: 32
vocab_size: 32000
sliding_window: 4096
```

**With TP=2:**
- hidden_size per GPU: 2048
- attention heads per GPU: 16
- KV heads per GPU: 4
- intermediate per GPU: 7168

---

## Existing Work: PR #1646

**Branch**: `pv-feat/exla-sharding-poc`
**URL**: https://github.com/elixir-nx/nx/pull/1646
**Author**: Paulo Valente (@polvalente)
**Status**: Draft PR

### What's Implemented

**1. EXLA.Sharding module** (new file):
```elixir
defmodule EXLA.Sharding do
  # DeviceMesh: name + axes with sizes
  defstruct DeviceMesh: [:name, :axes]

  # TensorSharding: mesh_name + dimension-to-axis mapping
  defstruct TensorSharding: [:mesh_name, :axes]

  def mesh(name, axes)      # Create device mesh
  def sharding(mesh, axes)  # Create tensor sharding spec
end
```

**2. C++ Changes** (exla/c_src/exla/exla.cc):
- `mlir_add_mesh()` - Adds mesh to MLIR module
- `mlir_create_tensor_sharding_attr()` - Creates sharding attributes
- `mlir_set_arg_sharding()` - Sets sharding on function arguments
- Shardy dialect registration

**3. MLIR Module Changes** (exla/lib/exla/mlir/module.ex):
- `add_mesh/2` function
- SPMD compilation with `use_spmd` option
- `input_shardings` passed to executable

**4. Client Changes** (exla/c_src/exla/exla_client.cc):
- Updated `UnpackRunArguments()` for SPMD partitions
- Modified device assignment for automatic partitioning

### What's Missing from PR #1646

1. **Collective operations** - No all-reduce, all-gather, reduce-scatter
2. **Higher-level API** - No easy way to specify TP patterns
3. **Bumblebee integration** - No sharded model loading

### Open Design Question from PR

> "The goal is for us to discuss whether we want these as Nx callbacks, or if we want to add a way for EXLA to declare its own defn symbols"

**Our approach**: Start with EXLA-specific (`EXLA.Collective.all_reduce`)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    USER API                                  │
│  Bumblebee.Distributed.load_model/2                         │
│  Bumblebee.Distributed.serving/4                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              BUMBLEBEE DISTRIBUTED LAYER                     │
│  lib/bumblebee/distributed.ex           - Main API           │
│  lib/bumblebee/distributed/sharded_loader.ex - Param loading │
│  lib/bumblebee/distributed/tp_layers.ex - TP layer wrappers  │
│  lib/bumblebee/distributed/tp_transformer.ex - TP blocks     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 EXLA EXTENSIONS                              │
│  exla/lib/exla/sharding.ex    - Mesh + Sharding (PR #1646)  │
│  exla/lib/exla/collective.ex  - All-reduce, all-gather (NEW) │
│  exla/c_src/exla/exla.cc      - C++ collective NIFs (NEW)    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    XLA / NCCL                                │
│  StableHLO collective ops → NCCL on NVIDIA GPUs             │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: EXLA Collective Operations
**Goal**: Add all-reduce, all-gather to EXLA building on PR #1646

**Files to create/modify:**
- `exla/lib/exla/collective.ex` (NEW)
- `exla/c_src/exla/exla.cc` (ADD collective NIFs)
- `exla/lib/exla/nif.ex` (ADD NIF bindings)

### Phase 2: Sharded Parameter Loading
**Goal**: Load Mistral weights sharded across GPUs

**Files to create:**
- `lib/bumblebee/distributed.ex` (NEW)
- `lib/bumblebee/distributed/sharded_loader.ex` (NEW)

### Phase 3: TP-Aware Transformer Layers
**Goal**: Implement column-parallel and row-parallel dense with all-reduce

**Files to create:**
- `lib/bumblebee/distributed/tp_layers.ex` (NEW)
- `lib/bumblebee/distributed/tp_transformer.ex` (NEW)

### Phase 4: Distributed Serving
**Goal**: End-to-end inference with TP

**Files to modify:**
- `lib/bumblebee/text/generation.ex`
- `lib/bumblebee/layers/decoder.ex` (sharded KV cache)

---

## Detailed Implementation Steps

### Step 1: Clone and Setup PR #1646

```bash
# Clone Nx repository
git clone https://github.com/elixir-nx/nx.git
cd nx
git fetch origin pull/1646/head:pv-feat/exla-sharding-poc
git checkout pv-feat/exla-sharding-poc

# Build EXLA
cd exla
mix deps.get
mix compile

# Test existing sharding
XLA_FLAGS="--xla_force_host_platform_device_count=2" mix run sharding.exs
```

### Step 2-4: See Implementation Code Files

The full implementation code is in separate files:
- `docs/tp_implementation/exla_collective.ex`
- `docs/tp_implementation/bumblebee_distributed.ex`
- `docs/tp_implementation/sharded_loader.ex`
- `docs/tp_implementation/tp_layers.ex`
- `docs/tp_implementation/tp_transformer.ex`

---

## Testing Strategy

### Test 1: Verify Multi-GPU Setup
```elixir
# Should return 2 or more
EXLA.Client.fetch!(:cuda).device_count
```

### Test 2: Test PR #1646 Sharding Example
```bash
cd nx/exla
XLA_FLAGS="--xla_force_host_platform_device_count=2" mix run sharding.exs
```

### Test 3: All-Reduce Correctness
```elixir
mesh = EXLA.Sharding.mesh(:tp, tp: 2)
partial = Nx.tensor([1.0, 2.0, 3.0, 4.0])
result = EXLA.Collective.all_reduce(partial, :sum, mesh: mesh)
# Verify result is sum of partials from all devices
```

### Test 4: End-to-End Inference
```elixir
# Compare outputs: single GPU vs TP=2
# Should produce identical results (within floating point tolerance)

# Single GPU
{:ok, single_model} = Bumblebee.load_model({:hf, "mistralai/Mistral-7B-v0.1"})
single_output = run_inference(single_model, "Hello")

# TP=2
mesh = EXLA.Sharding.mesh(:tp, tp: 2)
{:ok, tp_model} = Bumblebee.Distributed.load_model({:hf, "mistralai/Mistral-7B-v0.1"}, mesh: mesh)
tp_output = run_inference(tp_model, "Hello")

# Compare
assert_close(single_output, tp_output, atol: 1.0e-5)
```

---

## File Reference

### EXLA Files (to modify/create)

| File | Action | Purpose |
|------|--------|---------|
| `exla/lib/exla/collective.ex` | CREATE | All-reduce, all-gather APIs |
| `exla/lib/exla/sharding.ex` | EXISTS (PR #1646) | Mesh + TensorSharding |
| `exla/lib/exla/nif.ex` | MODIFY | Add collective NIF bindings |
| `exla/c_src/exla/exla.cc` | MODIFY | Add collective C++ NIFs |
| `exla/lib/exla/mlir/module.ex` | EXISTS (PR #1646) | SPMD compilation |

### Bumblebee Files (to create)

| File | Purpose |
|------|---------|
| `lib/bumblebee/distributed.ex` | Main API module |
| `lib/bumblebee/distributed/sharded_loader.ex` | Sharded parameter loading |
| `lib/bumblebee/distributed/tp_layers.ex` | Column/row parallel layers |
| `lib/bumblebee/distributed/tp_transformer.ex` | TP transformer blocks |

### Bumblebee Files (reference, may need modification)

| File | Purpose |
|------|---------|
| `lib/bumblebee.ex` | Main entry point |
| `lib/bumblebee/layers/transformer.ex` | Existing transformer impl |
| `lib/bumblebee/layers/decoder.ex` | KV cache handling |
| `lib/bumblebee/text/mistral.ex` | Mistral model spec |
| `lib/bumblebee/shared.ex` | Serving utilities |

---

## Summary

This plan enables tensor parallelism in Bumblebee by:

1. **Building on PR #1646** - Uses Paulo Valente's mesh/sharding foundation
2. **Adding collective ops** - All-reduce for synchronization after row-parallel layers
3. **Sharded loading** - Each GPU loads only its portion of weights
4. **TP-aware layers** - Column-parallel (no sync) and row-parallel (with sync) dense layers
5. **2 all-reduces per block** - After attention output and FFN output

**Target**: Mistral 7B on 2 GPUs with tensor parallelism
