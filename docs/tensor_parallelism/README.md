# Tensor Parallelism for LLM Inference

This documentation covers the tensor parallelism (TP) implementation in Bumblebee/EXLA for running large language models across multiple GPUs.

## What is Tensor Parallelism?

Tensor Parallelism splits individual layers across multiple GPUs, allowing models that don't fit on a single GPU to be run efficiently. Unlike pipeline parallelism (which assigns different layers to different GPUs), TP keeps all GPUs busy processing every token.

```
┌─────────────────────────────────────────────────────────────────┐
│                    TENSOR PARALLELISM (TP=4)                    │
├─────────────────────────────────────────────────────────────────┤
│  Input: [batch, seq, hidden]  (replicated on all GPUs)          │
│                              │                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  GPU 0   │ │  GPU 1   │ │  GPU 2   │ │  GPU 3   │           │
│  │ 8 heads  │ │ 8 heads  │ │ 8 heads  │ │ 8 heads  │           │
│  │ 2 KV     │ │ 2 KV     │ │ 2 KV     │ │ 2 KV     │           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
│       │            │            │            │                   │
│       └────────────┴─────┬──────┴────────────┘                   │
│                          │                                       │
│                    ALL-REDUCE (NCCL)                             │
│                          │                                       │
│                   [batch, seq, hidden]                           │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Running Qwen3-4B on 4 GPUs

```bash
cd /path/to/bumblebee

# Full model (36 layers), 20 tokens, greedy decoding
LAYERS=36 TOKENS=20 TEMP=0 mix run examples/tp_4gpu_qwen3.exs
```

### Running Mistral 7B on 4 GPUs

```bash
LAYERS=32 TOKENS=20 mix run examples/tp_4gpu_generate_final.exs
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LAYERS` | 4 | Number of transformer layers to use |
| `TOKENS` | 30 | Maximum tokens to generate |
| `TEMP` | 0.7 | Sampling temperature (0 = greedy) |
| `TOP_K` | 50 | Top-k filtering |
| `TOP_P` | 0.9 | Nucleus sampling threshold |
| `MAX_SEQ` | 128 | Pre-allocated KV cache length |
| `MEM_FRAC` | 0.85 | GPU memory fraction to allocate |
| `PROMPT` | (default) | Input prompt text |

## Supported Models

| Model | Parameters | TP Support | Notes |
|-------|------------|------------|-------|
| Qwen3-4B-Instruct | 4B | ✅ TP=4 | QK Norm, chat template |
| Mistral-7B-v0.1 | 7B | ✅ TP=4 | Base model |
| Mistral-7B-Instruct | 7B | ✅ TP=4 | Instruction-tuned |

## Hardware Requirements

### Minimum
- 4× NVIDIA GPUs with NVLink (for fast all-reduce)
- 16GB+ VRAM per GPU (for 4B models)
- CUDA 12.0+, cuDNN 8.9+

### Recommended
- 4× H100 NVL (94GB each)
- NVLink 4.0 (900 GB/s bandwidth)
- For 7B+ models: 24GB+ VRAM per GPU

### Performance on 4× H100 NVL

| Model | Prefill (26 tokens) | Decode (per token) |
|-------|---------------------|-------------------|
| Qwen3-4B | ~35s (incl. compile) | ~5-8ms |
| Mistral-7B | ~45s (incl. compile) | ~10-15ms |

*Note: First run includes XLA compilation. Subsequent runs with same shapes are faster.*

## Documentation

| Document | Description |
|----------|-------------|
| [01_architecture.md](./01_architecture.md) | Design decisions and architecture |
| [02_how_it_works.md](./02_how_it_works.md) | Implementation walkthrough |
| [03_memory_management.md](./03_memory_management.md) | GPU memory and KV cache |
| [04_performance.md](./04_performance.md) | Benchmarks and optimization |
| [05_future_roadmap.md](./05_future_roadmap.md) | Future plans |
| [06_flash_attention.md](./06_flash_attention.md) | Flash Attention implementation |

## Key Features

- **EXLA SPMD** - Multi-device execution via StableHLO
- **NCCL All-Reduce** - Efficient cross-GPU communication
- **Pre-allocated KV Cache** - O(1) per-token decode, no recompilation
- **Proper Operations** - Real RMSNorm, softmax, RoPE (not approximations)
- **Production Features** - Sampling, EOS detection, streaming output
- **Memory Calculator** - Automatic GPU memory allocation

## Example Output

```
$ LAYERS=36 TOKENS=20 TEMP=0 PROMPT="What is the capital of France?" \
    mix run examples/tp_4gpu_qwen3.exs

Memory calculation (per GPU with TP=4):
  Model weights:
    Total params: 4022.5M (~4B)
    Per GPU (sharded): 4.83 GB
  Runtime buffers:
    KV cache: 9.0 MB
  Total per GPU: 17.3 GB
  Memory fraction: 0.85

Generated text: The capital of France is Paris. [EOS]
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         BUMBLEBEE                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │  Model Loading  │  │   Tokenization   │  │   Generation   │  │
│  │  (HuggingFace)  │  │   (Bumblebee)    │  │   Loop         │  │
│  └────────┬────────┘  └────────┬─────────┘  └───────┬────────┘  │
└───────────┼────────────────────┼────────────────────┼───────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                           EXLA                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │   SPMD.build    │  │   MLIR Builder   │  │   SPMD.run     │  │
│  │   (replicas)    │  │   (StableHLO)    │  │   (execution)  │  │
│  └────────┬────────┘  └────────┬─────────┘  └───────┬────────┘  │
└───────────┼────────────────────┼────────────────────┼───────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                         XLA / NCCL                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │  GPU 0 (H100)   │  │  GPU 1 (H100)   │  │  GPU 2/3...    │  │
│  │  Local compute  │◄─►│  Local compute  │◄─►│  Local compute │  │
│  └─────────────────┘  └─────────────────┘  └────────────────┘  │
│                    ▲           ▲           ▲                     │
│                    └───────────┴───────────┘                     │
│                         NVLink (900 GB/s)                        │
└─────────────────────────────────────────────────────────────────┘
```

## License

Same as Bumblebee - Apache 2.0
