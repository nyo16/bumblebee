defmodule Bumblebee.Layers.Moe do
  @moduledoc false

  import Nx.Defn
  import Bumblebee.Utils.Model, only: [join: 2]

  @doc """
  Builds a sparse Mixture of Experts block.

  This is the main entry point for creating an MoE layer. It combines
  routing and expert computation into a single block that can replace
  a standard FFN in a transformer.

  ## Options

    * `:num_experts` - the total number of experts

    * `:num_experts_per_tok` - the number of experts to activate per token

    * `:hidden_size` - the input/output hidden size

    * `:moe_intermediate_size` - the intermediate size for each expert's FFN

    * `:activation` - the activation function (e.g., :silu)

    * `:norm_topk_prob` - whether to normalize top-k routing probabilities

    * `:name` - the layer name prefix

  Returns a container with:
    * `:hidden_state` - the expert output
    * `:router_logits` - the router logits (for auxiliary loss computation)

  """
  def sparse_moe_block(hidden_state, opts \\ []) do
    num_experts = Keyword.fetch!(opts, :num_experts)
    num_experts_per_tok = Keyword.fetch!(opts, :num_experts_per_tok)
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    moe_intermediate_size = Keyword.fetch!(opts, :moe_intermediate_size)
    activation = Keyword.fetch!(opts, :activation)
    norm_topk_prob = Keyword.get(opts, :norm_topk_prob, false)
    name = opts[:name]

    # Step 1: Router - compute routing weights and select top-k experts
    router_logits =
      Axon.dense(hidden_state, num_experts,
        name: join(name, "gate"),
        use_bias: false
      )

    # Step 2: Build all expert networks
    # Each expert is a gated FFN: output = down(gate(hidden) * activation(up(hidden)))
    expert_outputs =
      for expert_idx <- 0..(num_experts - 1) do
        build_expert(hidden_state, expert_idx, moe_intermediate_size, hidden_size, activation,
          name: name
        )
      end

    # Stack expert outputs using container: (num_experts, batch, seq_len, hidden_size)
    # We'll use Axon.layer to manually stack the outputs
    stacked_experts =
      Axon.layer(
        fn inputs, _opts ->
          # inputs is a list of tensors, one per expert
          Nx.stack(inputs)
        end,
        expert_outputs,
        name: join(name, "stack_experts"),
        op_name: :stack_experts
      )

    # Step 3: Apply routing and combine expert outputs
    Axon.layer(
      fn stacked_experts, router_logits, _opts ->
        route_and_combine(
          stacked_experts,
          router_logits,
          num_experts_per_tok,
          norm_topk_prob
        )
      end,
      [stacked_experts, router_logits],
      name: join(name, "route_and_combine"),
      op_name: :moe_route_and_combine
    )
    |> then(fn combined_output ->
      # Return both output and router logits for auxiliary loss
      Axon.container(%{
        hidden_state: combined_output,
        router_logits: router_logits
      })
    end)
  end

  defp build_expert(hidden_state, expert_idx, intermediate_size, output_size, activation, opts) do
    name = opts[:name]
    expert_name = "experts.#{expert_idx}"

    # Gated FFN structure: gate_proj and up_proj
    gate =
      Axon.dense(hidden_state, intermediate_size,
        name: join(name, join(expert_name, "gate_proj")),
        use_bias: false
      )

    up =
      Axon.dense(hidden_state, intermediate_size,
        name: join(name, join(expert_name, "up_proj")),
        use_bias: false
      )

    # Combine: up * activation(gate)
    intermediate = Axon.multiply(up, Axon.activation(gate, activation))

    # Down projection
    Axon.dense(intermediate, output_size,
      name: join(name, join(expert_name, "down_proj")),
      use_bias: false
    )
  end

  defnp route_and_combine(stacked_experts, router_logits, num_experts_per_tok, norm_topk_prob) do
    # stacked_experts shape: (num_experts, batch, seq_len, hidden_size)
    # router_logits shape: (batch, seq_len, num_experts)

    # Compute routing probabilities
    routing_probs = Axon.Activations.softmax(router_logits, axis: -1)

    # Get top-k experts per token
    top_k_output = Nx.top_k(routing_probs, k: num_experts_per_tok)
    top_k_weights = top_k_output.values
    top_k_indices = top_k_output.indices

    # Normalize top-k weights if requested
    top_k_weights =
      if norm_topk_prob do
        sum = Nx.sum(top_k_weights, axes: [-1], keep_axes: true)
        Nx.divide(top_k_weights, sum)
      else
        top_k_weights
      end

    # Get dimensions
    {_num_experts, batch_size, seq_len, hidden_size} = Nx.shape(stacked_experts)

    # Reshape for gathering: (batch, seq_len, num_experts, hidden_size)
    experts_reshaped = Nx.transpose(stacked_experts, axes: [1, 2, 0, 3])

    # Initialize output
    output = Nx.broadcast(0.0, {batch_size, seq_len, hidden_size})

    # For each of the top-k positions, gather the selected expert output
    # and weight it by the routing weight
    {output, _} =
      while {output, k = 0}, k < num_experts_per_tok do
        # Get expert indices and weights for this k position
        # (batch, seq_len)
        expert_idx = top_k_indices[[.., .., k]]
        # (batch, seq_len)
        weight = top_k_weights[[.., .., k]]

        # Gather expert outputs for the selected experts
        # We need to gather from experts_reshaped using expert_idx
        expert_output = gather_expert_outputs(experts_reshaped, expert_idx)

        # Weight the expert output
        weight_expanded = Nx.new_axis(weight, -1)
        weighted_output = Nx.multiply(expert_output, weight_expanded)

        # Accumulate
        {Nx.add(output, weighted_output), k + 1}
      end

    output
  end

  defnp gather_expert_outputs(experts_reshaped, expert_indices) do
    # experts_reshaped: (batch, seq_len, num_experts, hidden_size)
    # expert_indices: (batch, seq_len)
    # Returns: (batch, seq_len, hidden_size)

    {batch_size, seq_len, num_experts, hidden_size} = Nx.shape(experts_reshaped)

    # Flatten experts for easier gathering
    # Reshape to (batch*seq_len, num_experts, hidden_size)
    experts_flat = Nx.reshape(experts_reshaped, {batch_size * seq_len, num_experts, hidden_size})

    # Flatten expert indices to (batch*seq_len,)
    indices_flat = Nx.reshape(expert_indices, {batch_size * seq_len})

    # Gather: for each position, take the output from the selected expert
    # Use Nx.take to select along expert dimension
    output_flat =
      Nx.take_along_axis(experts_flat, Nx.reshape(indices_flat, {:auto, 1, 1}), axis: 1)

    # Remove the singleton dimension and reshape back
    output_flat = Nx.squeeze(output_flat, axes: [1])
    Nx.reshape(output_flat, {batch_size, seq_len, hidden_size})
  end
end
