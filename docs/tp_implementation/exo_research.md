# Insights from exo (exo-explore/exo)

**Repository**: https://github.com/exo-explore/exo

**exo** is an open-source distributed AI cluster that runs LLMs across multiple devices with automatic peer discovery and topology-aware parallelization.

---

## Key Features

- **Automatic device discovery** - No manual configuration needed
- **RDMA over Thunderbolt 5** - 99% latency reduction between devices
- **Topology-aware placement** - Optimal model distribution based on network
- **Tensor parallelism** - 1.8x speedup on 2 devices, 3.2x on 4 devices

---

## Architecture

### 1. Topology-Aware Placement

Uses a **directed graph** (via rustworkx) to represent network topology:

```python
# Graph structure
rx.PyDiGraph[NodeInfo, Connection]

# Key operations:
- Node operations (add, remove, query)
- Connection management
- Cycle detection (general and thunderbolt-specific)
- Neighbor discovery
- Leaf node identification
```

**Placement Algorithm** (greedy):
1. Find network cycles (interconnected node groups)
2. Filter by memory requirements
3. Validate sharding constraints (`hidden_size % tp_size == 0`)
4. Prefer smallest cycles, then Thunderbolt connections
5. Final tiebreaker: maximum total available RAM

### 2. Dual Parallelism Strategies

```python
# Pipeline Parallelism - Layer-based
class PipelineShardMetadata:
    start_layer: int    # Inclusive
    end_layer: int      # Exclusive
    device_rank: int
    world_size: int
    is_first_layer: bool
    is_last_layer: bool

# Tensor Parallelism - Weight-based
class TensorShardMetadata:
    device_rank: int
    world_size: int
    # Constraint: hidden_size % world_size == 0
```

### 3. Model-Specific Sharding Strategies

```python
class LlamaShardingStrategy:
    # Attention projections:
    # - Q, K, V: "all-to-sharded" (broadcast input, shard output)
    # - Output: "sharded-to-all" (gather output)
    # Divides attention heads by number of devices

class DeepSeekShardingStrategy:
    # Same as Llama plus:
    # - MoE handling with in-place sharding
    # - Uses all_sum for gradient aggregation

class QwenShardingStrategy:
    # Attention + MoE-specific sharding
```

### 4. Communication Patterns

**Core patterns**:
- **all-to-sharded**: Input replicated → each device produces output shard
- **sharded-to-all**: Each device has shard → gather produces full output

**Routing**: Gossip-based pub/sub for message passing

### 5. Distributed Coordination

**Leader Election** (clock-based consensus):
```python
# Message comparison priority:
clock > seniority > command_count > node_id

# Non-candidates have seniority = -1
# 3-second election timeout
# Messages rebroadcast after timeout
```

### 6. Auto-Parallelism Implementation

**Pipeline Auto-Parallel**:
```python
def pipeline_auto_parallel():
    # 1. Slice layers based on PipelineShardMetadata
    # 2. Wrap first layer with PipelineFirstLayer
    #    - Receives activations from previous device
    # 3. Wrap last layer with PipelineLastLayer
    #    - Sends outputs to next device
    #    - Performs all_gather for final reconstruction
```

**Tensor Auto-Parallel**:
```python
def tensor_auto_parallel():
    # 1. Shard weights using model-specific strategy
    # 2. Use shard_linear with patterns:
    #    - "all-to-sharded" for Q, K, V projections
    #    - "sharded-to-all" for output projections
    # 3. Insert collective ops at boundaries
```

---

## Ideas Applicable to Bumblebee

### 1. Topology Discovery
```elixir
# Could use Erlang's :digraph module
defmodule Bumblebee.Distributed.Topology do
  def discover_devices do
    # Query EXLA for available GPUs
    # Build graph with connection info
    # Detect high-speed links (NVLink)
  end

  def optimal_placement(model_size, devices) do
    # Find smallest device group with sufficient memory
    # Prefer high-bandwidth connections
  end
end
```

### 2. Sharding Constraints Validation
```elixir
def validate_tp_config(spec, tp_size) do
  cond do
    rem(spec.hidden_size, tp_size) != 0 ->
      {:error, "hidden_size must be divisible by tp_size"}
    rem(spec.num_attention_heads, tp_size) != 0 ->
      {:error, "num_attention_heads must be divisible by tp_size"}
    rem(spec.num_key_value_heads, tp_size) != 0 ->
      {:error, "num_key_value_heads must be divisible by tp_size"}
    true ->
      :ok
  end
end
```

### 3. Dual Strategy Support
```elixir
defmodule Bumblebee.Distributed do
  @type strategy :: :tensor_parallel | :pipeline_parallel

  def load_model(repo, opts) do
    case Keyword.get(opts, :strategy, :tensor_parallel) do
      :tensor_parallel -> load_tp_model(repo, opts)
      :pipeline_parallel -> load_pp_model(repo, opts)
    end
  end
end
```

### 4. Model-Specific Sharding
```elixir
defmodule Bumblebee.Distributed.ShardingStrategy do
  @callback shard_attention(spec, mesh) :: sharding_spec
  @callback shard_ffn(spec, mesh) :: sharding_spec
  @callback shard_moe(spec, mesh) :: sharding_spec  # For MoE models
end

defmodule Bumblebee.Distributed.LlamaStrategy do
  @behaviour Bumblebee.Distributed.ShardingStrategy
  # Standard column/row parallel for attention and FFN
end

defmodule Bumblebee.Distributed.DeepSeekStrategy do
  @behaviour Bumblebee.Distributed.ShardingStrategy
  # Additional MoE handling
end
```

### 5. Connection Priority
```elixir
def prioritize_connections(devices) do
  devices
  |> Enum.sort_by(fn device ->
    case device.connection_type do
      :nvlink -> 0      # Highest priority
      :nvswitch -> 1
      :pcie_x16 -> 2
      :pcie_x8 -> 3
      :ethernet -> 4    # Lowest priority
    end
  end)
end
```

---

## Key Takeaways

1. **Start simple**: Pipeline parallel is easier than tensor parallel
2. **Validate first**: Check sharding constraints before attempting TP
3. **Model-specific**: Different models may need different strategies
4. **Topology matters**: Connection speed affects optimal placement
5. **Erlang advantage**: Natural fit for distributed coordination (leader election, gossip)
