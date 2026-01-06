defmodule Bumblebee.Distributed.TPLayers do
  @moduledoc """
  Tensor-parallel layer implementations.

  This module provides building blocks for tensor-parallel transformer models:

  - **Column-parallel dense**: Weight split by output dimension, no sync needed
  - **Row-parallel dense**: Weight split by input dimension, all-reduce after
  - **All-reduce layer**: Sums partial results across all devices

  ## Usage Pattern

  In a tensor-parallel transformer:

  ```
  Input (replicated)
      │
      ▼
  Column-Parallel Dense (no sync)
      │
      ▼
  Local Computation
      │
      ▼
  Row-Parallel Dense + All-Reduce
      │
      ▼
  Output (replicated)
  ```

  ## Example

      # QKV projection (column-parallel)
      query = TPLayers.column_parallel_dense(hidden, 4096, mesh: mesh, name: "query")

      # Output projection (row-parallel with all-reduce)
      output = TPLayers.row_parallel_dense(attention_out, 4096, mesh: mesh, name: "output")
  """

  import Nx.Defn

  @doc """
  Column-parallel dense layer.

  The weight matrix is sharded along the output dimension (axis 1).
  Each GPU holds a slice of size `(input_size, output_size / tp_size)`.

  - **Input**: Replicated across all GPUs
  - **Output**: Partitioned across GPUs
  - **Communication**: None required

  ## Arguments

    * `input` - Input Axon node
    * `units` - Total output units (will be divided by TP size)
    * `opts` - Options:
      * `:mesh` - Device mesh (required)
      * `:name` - Layer name
      * `:kernel_initializer` - Weight initializer
      * `:use_bias` - Whether to use bias (default: false for TP)

  ## Example

      # Full model has 4096 hidden size, TP=2
      # Each GPU computes 2048 outputs
      query = TPLayers.column_parallel_dense(hidden, 4096, mesh: mesh)
  """
  @spec column_parallel_dense(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def column_parallel_dense(input, units, opts \\ []) do
    mesh = Keyword.fetch!(opts, :mesh)
    tp_size = mesh_tp_size(mesh)
    local_units = div(units, tp_size)

    name = Keyword.get(opts, :name)
    kernel_initializer = Keyword.get(opts, :kernel_initializer)
    use_bias = Keyword.get(opts, :use_bias, false)

    dense_opts = [name: name, use_bias: use_bias]

    dense_opts =
      if kernel_initializer,
        do: [{:kernel_initializer, kernel_initializer} | dense_opts],
        else: dense_opts

    # Standard dense with reduced output size
    # Params will be loaded sharded by ShardedLoader
    Axon.dense(input, local_units, dense_opts)
  end

  @doc """
  Row-parallel dense layer.

  The weight matrix is sharded along the input dimension (axis 0).
  Each GPU holds a slice of size `(input_size / tp_size, output_size)`.

  - **Input**: Partitioned across GPUs (from prior column-parallel layer)
  - **Output**: Partial result, needs all-reduce
  - **Communication**: All-reduce after this layer

  The all-reduce is automatically inserted after the dense computation.

  ## Arguments

    * `input` - Input Axon node (partitioned)
    * `units` - Output units (full size, not divided)
    * `opts` - Options:
      * `:mesh` - Device mesh (required)
      * `:name` - Layer name
      * `:kernel_initializer` - Weight initializer
      * `:use_bias` - Whether to use bias (default: false for TP)
      * `:all_reduce` - Whether to insert all-reduce (default: true)

  ## Example

      # Attention output projection with all-reduce
      output = TPLayers.row_parallel_dense(attention_out, 4096, mesh: mesh)
  """
  @spec row_parallel_dense(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def row_parallel_dense(input, units, opts \\ []) do
    mesh = Keyword.fetch!(opts, :mesh)
    name = Keyword.get(opts, :name)
    kernel_initializer = Keyword.get(opts, :kernel_initializer)
    use_bias = Keyword.get(opts, :use_bias, false)
    do_all_reduce = Keyword.get(opts, :all_reduce, true)

    dense_opts = [name: name, use_bias: use_bias]

    dense_opts =
      if kernel_initializer,
        do: [{:kernel_initializer, kernel_initializer} | dense_opts],
        else: dense_opts

    # Dense with full output size
    # Input is already partitioned, so weight input dim is local
    # Params loaded sharded by ShardedLoader
    output = Axon.dense(input, units, dense_opts)

    # Insert all-reduce to sum partial results
    if do_all_reduce do
      all_reduce(output, mesh, name: "#{name}_all_reduce")
    else
      output
    end
  end

  @doc """
  All-reduce layer - sums partial results across all devices.

  This is a custom Axon layer that inserts a collective all-reduce
  operation into the computation graph.

  ## Arguments

    * `input` - Input Axon node
    * `mesh` - Device mesh
    * `opts` - Options:
      * `:name` - Layer name
      * `:op` - Reduction operation (default: :sum)

  ## Example

      # Manual all-reduce insertion
      output = TPLayers.all_reduce(partial_output, mesh)
  """
  @spec all_reduce(Axon.t(), map(), keyword()) :: Axon.t()
  def all_reduce(input, mesh, opts \\ []) do
    name = Keyword.get(opts, :name, "all_reduce")
    op = Keyword.get(opts, :op, :sum)
    tp_size = mesh_tp_size(mesh)

    # Create replica groups for TP
    replica_groups = EXLA.Collective.replica_groups(tp_size)

    Axon.layer(
      &all_reduce_impl(&1, &2, op, replica_groups),
      [input],
      name: name,
      op_name: :all_reduce
    )
  end

  # The actual all-reduce implementation using EXLA.Nx.all_reduce
  # This gets lowered to stablehlo.all_reduce when compiled with EXLA
  defnp all_reduce_impl(tensor, _opts, op, replica_groups) do
    EXLA.Nx.all_reduce(tensor, op, replica_groups: replica_groups)
  end

  @doc """
  All-gather layer - gathers shards from all devices.

  Assembles a partitioned tensor into a full tensor replicated on all devices.

  ## Arguments

    * `input` - Input Axon node (partitioned)
    * `mesh` - Device mesh
    * `opts` - Options:
      * `:name` - Layer name
      * `:axis` - Axis along which to gather (default: last axis)
  """
  @spec all_gather(Axon.t(), map(), keyword()) :: Axon.t()
  def all_gather(input, mesh, opts \\ []) do
    name = Keyword.get(opts, :name, "all_gather")
    axis = Keyword.get(opts, :axis, -1)
    tp_size = mesh_tp_size(mesh)

    Axon.layer(
      &all_gather_impl(&1, &2, axis, tp_size),
      [input],
      name: name,
      op_name: :all_gather
    )
  end

  defnp all_gather_impl(tensor, _opts, _axis, _tp_size) do
    # Placeholder implementation
    # In single-device mode, just return the tensor
    # In SPMD mode, this would be replaced with actual all-gather
    tensor
  end

  @doc """
  Parallel embedding layer.

  The embedding table is sharded along the hidden dimension (column-parallel).
  Each GPU holds a slice of the embedding vectors.

  ## Arguments

    * `input` - Input token IDs
    * `vocab_size` - Vocabulary size
    * `hidden_size` - Full hidden dimension (will be divided by TP size)
    * `opts` - Options including `:mesh`

  ## Returns

    Partitioned embeddings (each GPU has hidden_size/tp_size dimensions)
  """
  @spec parallel_embedding(Axon.t(), pos_integer(), pos_integer(), keyword()) :: Axon.t()
  def parallel_embedding(input, vocab_size, hidden_size, opts \\ []) do
    mesh = Keyword.fetch!(opts, :mesh)
    tp_size = mesh_tp_size(mesh)
    local_hidden = div(hidden_size, tp_size)
    name = Keyword.get(opts, :name, "embedding")

    # Embedding lookup with reduced hidden size per GPU
    Axon.embedding(input, vocab_size, local_hidden, name: name)
  end

  @doc """
  Parallel output projection (LM head).

  For vocabulary-parallel output:
  - Each GPU holds a slice of the vocab dimension
  - Logits are partitioned across GPUs
  - Use all-gather before softmax if needed

  ## Arguments

    * `input` - Hidden states
    * `vocab_size` - Full vocabulary size
    * `opts` - Options including `:mesh`
  """
  @spec parallel_lm_head(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def parallel_lm_head(input, vocab_size, opts \\ []) do
    mesh = Keyword.fetch!(opts, :mesh)
    tp_size = mesh_tp_size(mesh)
    local_vocab = div(vocab_size, tp_size)
    name = Keyword.get(opts, :name, "lm_head")

    # Project to local vocab slice
    # Note: For TP, input is already partitioned from row-parallel layers
    # We might need to all-gather first depending on the model
    Axon.dense(input, local_vocab, name: name, use_bias: false)
  end

  # Helper functions

  defp mesh_tp_size(mesh) do
    case mesh do
      %{axes: axes} -> Map.get(axes, :tp, 1)
      _ -> 1
    end
  end
end
