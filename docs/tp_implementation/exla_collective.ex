# EXLA Collective Operations
# File: exla/lib/exla/collective.ex
#
# This module provides collective communication operations for distributed
# tensor computation. It should be added to the Nx repository after forking PR #1646.

defmodule EXLA.Collective do
  @moduledoc """
  Collective communication operations for distributed tensor computation.

  These operations synchronize data across multiple devices in a mesh.
  They are essential for tensor parallelism where partial results from
  different GPUs need to be combined.

  ## Overview

  In tensor parallelism, model weights are sharded across GPUs. After certain
  operations (row-parallel linear layers), partial results need to be summed
  across all GPUs. This is done via `all_reduce`.

  ## Example

      # Create a device mesh with 2 GPUs
      mesh = EXLA.Sharding.mesh(:tp, tp: 2)

      # Each GPU computes a partial result
      partial_output = row_parallel_matmul(input, sharded_weight)

      # Sum partial results across all GPUs
      full_output = EXLA.Collective.all_reduce(partial_output, :sum, mesh: mesh)
  """

  @doc """
  Reduces tensors across all devices using the specified operation.

  The input tensor on each device contains a partial result. After all_reduce,
  each device will have the combined result (e.g., sum of all partial results).

  ## Arguments
    * `tensor` - The input tensor (partial result on each device)
    * `op` - Reduction operation: `:sum`, `:mean`, `:max`, `:min`, `:product`
    * `opts` - Options:
      * `:mesh` - Device mesh (required)
      * `:replica_groups` - Custom replica groups (optional, derived from mesh)

  ## Returns
    The reduced tensor, identical on all devices in the mesh.

  ## Example

      # Each GPU has partial output from row-parallel matmul
      # GPU 0: [[1, 2], [3, 4]]
      # GPU 1: [[5, 6], [7, 8]]

      result = EXLA.Collective.all_reduce(partial, :sum, mesh: mesh)
      # Both GPUs now have: [[6, 8], [10, 12]]
  """
  @spec all_reduce(Nx.Tensor.t(), atom(), keyword()) :: Nx.Tensor.t()
  def all_reduce(tensor, op \\ :sum, opts \\ []) do
    mesh = Keyword.fetch!(opts, :mesh)
    replica_groups = Keyword.get_lazy(opts, :replica_groups, fn ->
      mesh_to_replica_groups(mesh)
    end)

    # Call into NIF - this creates a StableHLO AllReduceOp
    EXLA.NIF.mlir_all_reduce(tensor, op, replica_groups)
  end

  @doc """
  Gathers tensor shards from all devices and concatenates along the specified axis.

  Each device has a shard of the full tensor. After all_gather, each device
  will have the complete tensor assembled from all shards.

  ## Arguments
    * `tensor` - The local tensor shard
    * `opts` - Options:
      * `:mesh` - Device mesh (required)
      * `:axis` - Axis along which to concatenate (default: 0)
      * `:replica_groups` - Custom replica groups (optional)

  ## Returns
    The gathered tensor with shape expanded along the gather axis.

  ## Example

      # GPU 0 has: [1, 2]
      # GPU 1 has: [3, 4]

      result = EXLA.Collective.all_gather(shard, axis: 0, mesh: mesh)
      # Both GPUs now have: [1, 2, 3, 4]
  """
  @spec all_gather(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def all_gather(tensor, opts \\ []) do
    mesh = Keyword.fetch!(opts, :mesh)
    axis = Keyword.get(opts, :axis, 0)
    replica_groups = Keyword.get_lazy(opts, :replica_groups, fn ->
      mesh_to_replica_groups(mesh)
    end)

    EXLA.NIF.mlir_all_gather(tensor, axis, replica_groups)
  end

  @doc """
  Reduces tensors across devices and scatters the result.

  This is equivalent to all_reduce followed by chunking/scattering the result.
  More efficient than doing both operations separately.

  ## Arguments
    * `tensor` - The input tensor
    * `op` - Reduction operation: `:sum`, `:mean`, `:max`, `:min`
    * `opts` - Options:
      * `:mesh` - Device mesh (required)
      * `:axis` - Axis along which to scatter (default: 0)

  ## Returns
    The reduced and scattered tensor shard for this device.

  ## Example

      # GPU 0 has: [1, 2, 3, 4]
      # GPU 1 has: [5, 6, 7, 8]

      result = EXLA.Collective.reduce_scatter(tensor, :sum, axis: 0, mesh: mesh)
      # GPU 0 gets: [6, 8]  (sum of first halves)
      # GPU 1 gets: [10, 12] (sum of second halves)
  """
  @spec reduce_scatter(Nx.Tensor.t(), atom(), keyword()) :: Nx.Tensor.t()
  def reduce_scatter(tensor, op \\ :sum, opts \\ []) do
    mesh = Keyword.fetch!(opts, :mesh)
    axis = Keyword.get(opts, :axis, 0)
    replica_groups = Keyword.get_lazy(opts, :replica_groups, fn ->
      mesh_to_replica_groups(mesh)
    end)

    EXLA.NIF.mlir_reduce_scatter(tensor, op, axis, replica_groups)
  end

  @doc """
  Performs an all-to-all operation across devices.

  Each device sends a different chunk to each other device and receives
  chunks from all other devices.

  ## Arguments
    * `tensor` - The input tensor
    * `opts` - Options:
      * `:mesh` - Device mesh (required)
      * `:split_axis` - Axis along which to split for sending
      * `:concat_axis` - Axis along which to concatenate received chunks

  ## Returns
    The tensor after all-to-all exchange.
  """
  @spec all_to_all(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def all_to_all(tensor, opts \\ []) do
    mesh = Keyword.fetch!(opts, :mesh)
    split_axis = Keyword.get(opts, :split_axis, 0)
    concat_axis = Keyword.get(opts, :concat_axis, 0)
    replica_groups = Keyword.get_lazy(opts, :replica_groups, fn ->
      mesh_to_replica_groups(mesh)
    end)

    EXLA.NIF.mlir_all_to_all(tensor, split_axis, concat_axis, replica_groups)
  end

  # Convert a device mesh to replica groups for XLA collective operations
  # For tensor parallelism, all devices in the TP dimension form one group
  defp mesh_to_replica_groups(%EXLA.Sharding.DeviceMesh{} = mesh) do
    # For simple TP, all devices are in one group
    size = mesh_size(mesh)
    [Enum.to_list(0..(size - 1))]
  end

  defp mesh_size(%EXLA.Sharding.DeviceMesh{axes: axes}) do
    Enum.reduce(axes, 1, fn {_name, size}, acc -> acc * size end)
  end
end


# =============================================================================
# NIF Bindings to add to exla/lib/exla/nif.ex
# =============================================================================
#
# Add these function declarations to the existing EXLA.NIF module:
#
# def mlir_all_reduce(_tensor, _op, _replica_groups), do: :erlang.nif_error(:not_loaded)
# def mlir_all_gather(_tensor, _axis, _replica_groups), do: :erlang.nif_error(:not_loaded)
# def mlir_reduce_scatter(_tensor, _op, _axis, _replica_groups), do: :erlang.nif_error(:not_loaded)
# def mlir_all_to_all(_tensor, _split_axis, _concat_axis, _replica_groups), do: :erlang.nif_error(:not_loaded)


# =============================================================================
# C++ Implementation Sketch for exla/c_src/exla/exla.cc
# =============================================================================
#
# ERL_NIF_TERM mlir_all_reduce(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
#   // argv[0]: MLIRFunction* containing the builder context
#   // argv[1]: mlir::Value input tensor
#   // argv[2]: atom for reduction op (:sum, :mean, :max, :min)
#   // argv[3]: replica_groups as list of lists of integers
#
#   MLIRFunction* function;
#   if (!get_function(env, argv[0], &function)) {
#     return make_error(env, "invalid function");
#   }
#
#   mlir::Value input;
#   if (!get_mlir_value(env, argv[1], &input)) {
#     return make_error(env, "invalid input tensor");
#   }
#
#   // Parse reduction operation
#   std::string op_str;
#   if (!get_atom(env, argv[2], &op_str)) {
#     return make_error(env, "invalid reduction op");
#   }
#
#   // Parse replica groups
#   std::vector<std::vector<int64_t>> replica_groups;
#   if (!parse_replica_groups(env, argv[3], &replica_groups)) {
#     return make_error(env, "invalid replica groups");
#   }
#
#   // Create replica groups attribute
#   auto replica_groups_attr = get_replica_groups_attr(function->builder(), replica_groups);
#
#   // Create channel handle for cross-device communication
#   auto channel_handle = mlir::stablehlo::ChannelHandleAttr::get(
#     function->context(),
#     /*handle=*/function->next_channel_id(),
#     /*type=*/1  // DEVICE_TO_DEVICE
#   );
#
#   // Create the AllReduceOp
#   auto loc = function->builder().getUnknownLoc();
#   auto all_reduce_op = function->builder().create<mlir::stablehlo::AllReduceOp>(
#     loc,
#     input.getType(),
#     input,
#     replica_groups_attr,
#     channel_handle,
#     /*use_global_device_ids=*/mlir::UnitAttr()
#   );
#
#   // Set the reduction computation (sum, mean, etc.)
#   // This creates a region with the reduction function
#   set_reduction_computation(all_reduce_op, op_str);
#
#   return wrap_mlir_value(env, all_reduce_op.getResult());
# }
#
# // Similar implementations for mlir_all_gather, mlir_reduce_scatter, mlir_all_to_all
# // using stablehlo::AllGatherOp, stablehlo::ReduceScatterOp, stablehlo::AllToAllOp
