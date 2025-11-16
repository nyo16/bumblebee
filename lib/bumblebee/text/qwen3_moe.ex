defmodule Bumblebee.Text.Qwen3Moe do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 151_936,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 262_144,
        doc: """
        the vocabulary size of the position embedding. This corresponds to the maximum sequence
        length that this model can process. Typically this is set to a large value just in case,
        such as 512, 1024 or 2048
        """
      ],
      hidden_size: [
        default: 2048,
        doc: "the dimensionality of hidden layers"
      ],
      intermediate_size: [
        default: 9728,
        doc: "the dimensionality of intermediate layers in dense FFN blocks"
      ],
      attention_head_size: [
        default: 128,
        doc: """
        the size of the key, value, and query projection per attention head.
        """
      ],
      num_blocks: [
        default: 48,
        doc: "the number of Transformer blocks in the model"
      ],
      num_attention_heads: [
        default: 32,
        doc: "the number of attention heads for each attention layer in the model"
      ],
      num_key_value_heads: [
        default: 4,
        doc: "the number of key value heads for each attention layer in the model"
      ],
      activation: [
        default: :silu,
        doc: "the activation function"
      ],
      rotary_embedding_base: [
        default: 10_000,
        doc: "base for computing rotary embedding frequency"
      ],
      rotary_embedding_scaling_strategy: [
        default: nil,
        doc: """
        scaling configuration for rotary embedding. Currently the supported values are:

          * `%{type: :linear, factor: number()}`

          * `%{type: :dynamic, factor: number()}`

        For more details see https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases
        """
      ],
      layer_norm_epsilon: [
        default: 1.0e-6,
        doc: "the epsilon used by RMS normalization layers"
      ],
      initializer_scale: [
        default: 0.02,
        doc:
          "the standard deviation of the normal initializer used for initializing kernel parameters"
      ],
      tie_word_embeddings: [
        default: true,
        doc: "whether to tie input and output embedding weights"
      ],
      use_qk_norm: [
        default: true,
        doc: "whether to use RMS normalization on query and key projections"
      ],
      # MoE-specific options
      num_experts: [
        default: 128,
        doc: "the number of expert networks in MoE layers"
      ],
      num_experts_per_tok: [
        default: 8,
        doc: "the number of experts to activate per token (top-k routing)"
      ],
      moe_intermediate_size: [
        default: 768,
        doc: "the intermediate size for expert FFN layers"
      ],
      decoder_sparse_step: [
        default: 1,
        doc: "frequency of MoE layers. 1 means every layer has MoE"
      ],
      router_aux_loss_coef: [
        default: 0.001,
        doc: "coefficient for auxiliary load balancing loss"
      ],
      norm_topk_prob: [
        default: false,
        doc: "whether to normalize top-k routing probabilities"
      ],
      mlp_only_layers: [
        default: [],
        doc: "list of layer indices (0-based) that use dense MLP instead of MoE"
      ],
      output_router_logits: [
        default: false,
        doc: "whether to return router logits in model outputs for auxiliary loss computation"
      ]
    ] ++
      Shared.common_options([:num_labels, :id_to_label]) ++
      Shared.token_options(pad_token_id: 151_643)

  @moduledoc """
  Qwen3 Mixture of Experts (MoE) model family.

  ## Architectures

    * `:base` - plain Qwen3 MoE without any head on top

    * `:for_causal_language_modeling` - Qwen3 MoE with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - Qwen3 MoE with a sequence
      classification head. The head returns logits corresponding to
      possible classes

  ## Inputs

    * `"input_ids"` - `{batch_size, sequence_length}`

      Indices of input sequence tokens in the vocabulary.

    * `"attention_mask"` - `{batch_size, sequence_length}`

      Mask indicating which tokens to attend to. This is used to ignore
      padding tokens, which are added when processing a batch of sequences
      with different length.

    * `"position_ids"` - `{batch_size, sequence_length}`

      Indices of positions of each input sequence tokens in the position
      embeddings.

    * `"attention_head_mask"` - `{encoder_num_blocks, encoder_num_attention_heads}`

      Mask to nullify selected heads of the self-attention blocks in
      the encoder.

    * `"input_embeddings"` - `{batch_size, sequence_length, hidden_size}`

      Embedded representation of `"input_ids"`, which can be specified
      for more control over how `"input_ids"` are embedded than the
      model's internal embedding lookup. If `"input_embeddings"` are present,
      then `"input_ids"` will be ignored.

    * `"cache"`

      A container with cached layer results used to speed up sequential
      decoding (autoregression). With cache, certain hidden states are
      taken from the cache, rather than recomputed on every decoding
      pass. The cache should be treated as opaque and initialized with
      `Bumblebee.Text.Generation.init_cache/4`.

  ## Global layer options

  #{Shared.global_layer_options_doc([:output_hidden_states, :output_attentions])}

  ## Configuration

  #{Shared.options_doc(options)}

  ## References

    * [Qwen3 Technical Report](https://arxiv.org/abs/2507.xxxx)
    * [Qwen3-30B-A3B-Instruct Model Card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)

  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable
  @behaviour Bumblebee.Text.Generation

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(),
    do: [
      :base,
      :for_causal_language_modeling,
      :for_sequence_classification
    ]

  @impl true
  def config(spec, opts) do
    spec
    |> Shared.put_config_attrs(opts)
    |> Shared.validate_label_options()
  end

  @impl true
  def input_template(_spec) do
    %{
      "input_ids" => Nx.template({1, 1}, :s64)
    }
  end

  @impl true
  def init_cache(spec, batch_size, max_length, _inputs) do
    Layers.Decoder.init_cache(batch_size, max_length,
      hidden_size: spec.hidden_size,
      attention_head_size: spec.attention_head_size,
      decoder_num_attention_heads: spec.num_attention_heads,
      decoder_num_blocks: spec.num_blocks
    )
  end

  @impl true
  def traverse_cache(_spec, cache, fun) do
    Layers.Decoder.traverse_cache(cache, fun)
  end

  @impl true
  def model(%__MODULE__{architecture: :base} = spec) do
    inputs = inputs(spec)

    inputs
    |> core(spec)
    |> Layers.output()
  end

  def model(%__MODULE__{architecture: :for_causal_language_modeling} = spec) do
    inputs = inputs(spec)

    outputs = core(inputs, spec)
    logits = language_modeling_head(outputs.hidden_state, spec, name: "language_modeling_head")

    output_map = %{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions,
      cache: outputs.cache
    }

    # Optionally include router logits for auxiliary loss
    output_map =
      if spec.output_router_logits and Map.has_key?(outputs, :router_logits) do
        Map.put(output_map, :router_logits, outputs.router_logits)
      else
        output_map
      end

    Layers.output(output_map)
  end

  def model(%__MODULE__{architecture: :for_sequence_classification} = spec) do
    inputs = inputs(spec)

    outputs = core(inputs, spec)

    logits =
      Axon.dense(outputs.hidden_state, spec.num_labels,
        kernel_initializer: kernel_initializer(spec),
        name: "sequence_classification_head.output",
        use_bias: false
      )

    pooled_logits =
      Layers.if_present inputs["input_ids"] do
        Axon.layer(
          fn logits, input_ids, _opts ->
            indices =
              input_ids
              |> Nx.not_equal(spec.pad_token_id)
              |> Nx.sum(axes: [-1])
              |> Nx.subtract(1)
              |> Nx.as_type({:s, 64})

            Bumblebee.Utils.Nx.batched_take(logits, indices)
          end,
          [logits, inputs["input_ids"]]
        )
      else
        Layers.take_token(logits, axis: 1, index: -1)
      end

    Layers.output(%{
      logits: pooled_logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions,
      cache: outputs.cache
    })
  end

  defp inputs(spec) do
    shape = {nil, nil}
    hidden_shape = {nil, nil, spec.hidden_size}

    attention_head_mask_shape = {spec.num_blocks, spec.num_attention_heads}

    Bumblebee.Utils.Model.inputs_to_map([
      Axon.input("input_ids", optional: true, shape: shape),
      Axon.input("attention_mask", optional: true, shape: shape),
      Axon.input("position_ids", optional: true, shape: shape),
      Axon.input("attention_head_mask", optional: true, shape: attention_head_mask_shape),
      Axon.input("input_embeddings", optional: true, shape: hidden_shape),
      Axon.input("cache", optional: true)
    ])
  end

  defp core(inputs, spec) do
    embeddings =
      embedder(
        inputs["input_ids"],
        inputs["input_embeddings"],
        spec,
        name: "embedder"
      )

    position_ids =
      Layers.default inputs["position_ids"] do
        Layers.default_position_ids(embeddings)
      end

    decoder_outputs =
      decoder(
        embeddings,
        position_ids,
        inputs["attention_mask"],
        inputs["attention_head_mask"],
        inputs["cache"],
        spec,
        name: "decoder"
      )

    hidden_state =
      Layers.rms_norm(decoder_outputs.hidden_state,
        name: "output_norm",
        epsilon: spec.layer_norm_epsilon
      )

    %{
      hidden_state: hidden_state,
      hidden_states: Layers.append(decoder_outputs.hidden_states, hidden_state),
      attentions: decoder_outputs.attentions,
      cache: decoder_outputs.cache,
      router_logits: Map.get(decoder_outputs, :router_logits)
    }
  end

  defp embedder(input_ids, input_embeddings, spec, opts) do
    name = opts[:name]

    Layers.default input_embeddings do
      Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "token_embedding")
      )
    end
  end

  defp decoder(
         hidden_state,
         position_ids,
         attention_mask,
         attention_head_mask,
         cache,
         spec,
         opts
       ) do
    name = opts[:name]

    # Build query and key normalization functions
    query_norm =
      if spec.use_qk_norm do
        &Layers.rms_norm(&1, epsilon: spec.layer_norm_epsilon, channel_index: -1, name: &2)
      else
        nil
      end

    key_norm =
      if spec.use_qk_norm do
        &Layers.rms_norm(&1, epsilon: spec.layer_norm_epsilon, channel_index: -1, name: &2)
      else
        nil
      end

    # Handle attention mask caching and offset
    {attention_mask, cache} = Layers.Decoder.cached_attention_mask(attention_mask, cache)
    offset = Layers.Decoder.get_cache_offset(cache)

    # Build MoE blocks with heterogeneous layer support
    {final_hidden_state, all_hidden_states, all_attentions, final_cache, all_router_logits} =
      build_moe_blocks(
        hidden_state,
        attention_mask,
        attention_head_mask,
        cache,
        offset,
        position_ids,
        query_norm,
        key_norm,
        spec,
        name: name
      )

    # Update cache offset
    final_cache = Layers.Decoder.update_cache_offset(final_cache, hidden_state)

    %{
      hidden_state: final_hidden_state,
      hidden_states: all_hidden_states,
      attentions: all_attentions,
      cache: final_cache,
      router_logits: all_router_logits
    }
  end

  defp build_moe_blocks(
         hidden_state,
         attention_mask,
         attention_head_mask,
         cache,
         offset,
         position_ids,
         query_norm,
         key_norm,
         spec,
         opts
       ) do
    name = opts[:name]

    # Initialize state using Axon containers (similar to Layers.Transformer.blocks)
    state = %{
      hidden_state: hidden_state,
      hidden_states: Axon.container({hidden_state}),
      attentions: Axon.container({}),
      cache: cache,
      router_logits: Axon.container({})
    }

    # Process each block
    outputs =
      for block_idx <- 0..(spec.num_blocks - 1), reduce: state do
        state ->
          block_name = join(name, "blocks.#{block_idx}")

          # Get block cache
          block_cache = Layers.Decoder.get_block_cache(state.cache, block_idx)

          # Get attention head mask for this block
          block_attention_head_mask = Axon.nx(attention_head_mask, & &1[block_idx])

          # Determine if this layer should use MoE or dense FFN
          use_moe = block_idx not in spec.mlp_only_layers

          # Build the transformer block with appropriate FFN
          block_output =
            transformer_block(
              state.hidden_state,
              attention_mask,
              block_attention_head_mask,
              block_cache,
              offset,
              block_idx,
              position_ids,
              query_norm,
              key_norm,
              use_moe,
              spec,
              name: block_name
            )

          # Extract outputs
          new_hidden_state = Axon.nx(block_output, & &1.hidden_state)
          new_cache = Axon.nx(block_output, & &1.cache)
          attention = Axon.nx(block_output, & &1.attention)

          # Update cache
          updated_cache = Layers.Decoder.put_block_cache(state.cache, block_idx, new_cache)

          # Collect router logits
          router_logit = Axon.nx(block_output, & &1.router_logits)
          router_logits = Layers.append(state.router_logits, router_logit)

          %{
            hidden_state: new_hidden_state,
            hidden_states: Layers.append(state.hidden_states, new_hidden_state),
            attentions: Layers.append(state.attentions, attention),
            cache: updated_cache,
            router_logits: router_logits
          }
      end

    {outputs.hidden_state, outputs.hidden_states, outputs.attentions, outputs.cache,
     outputs.router_logits}
  end

  defp transformer_block(
         hidden_state,
         attention_mask,
         attention_head_mask,
         cache,
         offset,
         block_idx,
         position_ids,
         query_norm,
         key_norm,
         use_moe,
         spec,
         opts
       ) do
    name = opts[:name]

    # Self-attention norm
    normalized_hidden_state =
      Layers.rms_norm(hidden_state,
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "self_attention_norm")
      )

    # Self-attention
    {self_attention_cache, _cross_attention_cache} =
      Layers.Decoder.get_attention_caches(cache)

    {attention_hidden_state, attention_output, new_self_attention_cache} =
      self_attention(
        normalized_hidden_state,
        attention_mask,
        attention_head_mask,
        self_attention_cache,
        offset,
        block_idx,
        position_ids,
        query_norm,
        key_norm,
        spec,
        name: name
      )

    # Residual connection
    hidden_state = Axon.add(hidden_state, attention_hidden_state)

    # FFN norm
    normalized_hidden_state =
      Layers.rms_norm(hidden_state,
        epsilon: spec.layer_norm_epsilon,
        name: join(name, "output_norm")
      )

    # FFN (MoE or dense)
    {ffn_output, router_logits} =
      if use_moe do
        moe_ffn(normalized_hidden_state, spec, name: name)
      else
        {gated_ffn(normalized_hidden_state, spec.intermediate_size, spec.hidden_size,
           name: join(name, "ffn"),
           activation: spec.activation
         ), Layers.none()}
      end

    # Residual connection
    hidden_state = Axon.add(hidden_state, ffn_output)

    # Update cache
    new_cache =
      Layers.Decoder.put_attention_caches(
        cache,
        new_self_attention_cache,
        Layers.none()
      )

    # Return outputs
    Axon.container(%{
      hidden_state: hidden_state,
      cache: new_cache,
      attention: attention_output,
      router_logits: router_logits
    })
  end

  defp self_attention(
         hidden_state,
         attention_mask,
         attention_head_mask,
         cache,
         offset,
         _block_idx,
         position_ids,
         query_norm,
         key_norm,
         spec,
         opts
       ) do
    name = opts[:name]

    # Query, key, value projections
    query =
      Axon.dense(hidden_state, spec.num_attention_heads * spec.attention_head_size,
        name: join(name, "self_attention.query"),
        kernel_initializer: kernel_initializer(spec),
        use_bias: false
      )

    key =
      Axon.dense(hidden_state, spec.num_key_value_heads * spec.attention_head_size,
        name: join(name, "self_attention.key"),
        kernel_initializer: kernel_initializer(spec),
        use_bias: false
      )

    value =
      Axon.dense(hidden_state, spec.num_key_value_heads * spec.attention_head_size,
        name: join(name, "self_attention.value"),
        kernel_initializer: kernel_initializer(spec),
        use_bias: false
      )

    # Apply query and key normalization if specified
    query =
      if query_norm do
        query_norm.(query, join(name, "self_attention.query_norm"))
      else
        query
      end

    key =
      if key_norm do
        key_norm.(key, join(name, "self_attention.key_norm"))
      else
        key
      end

    # Reshape for multi-head attention
    query = Layers.split_heads(query, spec.num_attention_heads)
    key = Layers.split_heads(key, spec.num_key_value_heads)
    value = Layers.split_heads(value, spec.num_key_value_heads)

    # Apply rotary embeddings
    rotary_opts = [
      name: join(name, "self_attention.rotary_embedding"),
      max_positions: spec.max_positions,
      base: spec.rotary_embedding_base,
      scaling_strategy: spec.rotary_embedding_scaling_strategy
    ]

    {query, key} =
      Layers.rotary_embedding(
        query,
        key,
        position_ids,
        attention_mask,
        spec.attention_head_size,
        rotary_opts
      )

    # Cache key and value
    {key, value, new_cache} =
      Layers.Decoder.cached_attention_key_values(key, value, cache, offset)

    # Grouped query attention - repeat key/value to match query heads
    num_key_value_groups = div(spec.num_attention_heads, spec.num_key_value_heads)
    key = repeat_states(key, num_key_value_groups)
    value = repeat_states(value, num_key_value_groups)

    # Compute attention
    {attention_output, attention_weights} =
      Layers.attention(
        query,
        key,
        value,
        attention_mask,
        attention_head_mask,
        # attention_relative_bias
        Layers.none(),
        offset,
        causal: true
      )

    # Flatten trailing dimensions and output projection
    attention_output = Layers.flatten_trailing(attention_output)

    attention_output =
      Axon.dense(attention_output, spec.hidden_size,
        name: join(name, "self_attention.output"),
        kernel_initializer: kernel_initializer(spec),
        use_bias: false
      )

    {attention_output, attention_weights, new_cache}
  end

  defp moe_ffn(hidden_state, spec, opts) do
    name = opts[:name]

    moe_output =
      Layers.Moe.sparse_moe_block(hidden_state,
        num_experts: spec.num_experts,
        num_experts_per_tok: spec.num_experts_per_tok,
        hidden_size: spec.hidden_size,
        moe_intermediate_size: spec.moe_intermediate_size,
        activation: spec.activation,
        norm_topk_prob: spec.norm_topk_prob,
        name: join(name, "moe")
      )

    # Extract hidden state and router logits
    hidden_state_out = Axon.nx(moe_output, & &1.hidden_state)
    router_logits = Axon.nx(moe_output, & &1.router_logits)

    {hidden_state_out, router_logits}
  end

  defp gated_ffn(hidden_state, intermediate_size, output_size, opts) do
    name = opts[:name]
    activation = opts[:activation]

    intermediate =
      Axon.dense(hidden_state, intermediate_size,
        name: join(name, "intermediate"),
        use_bias: false
      )

    gate = Axon.dense(hidden_state, intermediate_size, name: join(name, "gate"), use_bias: false)

    hidden_state = Axon.multiply(intermediate, Axon.activation(gate, activation))

    Axon.dense(hidden_state, output_size, name: join(name, "output"), use_bias: false)
  end

  defp language_modeling_head(hidden_state, spec, opts) do
    name = opts[:name]

    Layers.dense_transposed(hidden_state, spec.vocab_size,
      kernel_initializer: kernel_initializer(spec),
      name: join(name, "output")
    )
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defp repeat_states(state, 1), do: state

  defp repeat_states(state, times) do
    Layers.repeat_interleave(state, times, axis: 2)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      scaling_strategy_converter = fn _name, value ->
        case value do
          %{"type" => "linear", "factor" => factor} when is_number(factor) ->
            {:ok, %{type: :linear, factor: factor}}

          %{"type" => "dynamic", "factor" => factor} when is_number(factor) ->
            {:ok, %{type: :dynamic, factor: factor}}

          nil ->
            {:ok, nil}

          _other ->
            {:ok, nil}
        end
      end

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          tie_word_embeddings: {"tie_word_embeddings", boolean()},
          max_positions: {"max_position_embeddings", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          num_key_value_heads: {"num_key_value_heads", number()},
          attention_head_size: {"head_dim", number()},
          intermediate_size: {"intermediate_size", number()},
          activation: {"hidden_act", activation()},
          rotary_embedding_base: {"rope_theta", number()},
          rotary_embedding_scaling_strategy:
            {"rope_scaling", optional(scaling_strategy_converter)},
          initializer_scale: {"initializer_range", number()},
          layer_norm_epsilon: {"rms_norm_eps", number()},
          # MoE-specific options
          num_experts: {"num_experts", number()},
          num_experts_per_tok: {"num_experts_per_tok", number()},
          moe_intermediate_size: {"moe_intermediate_size", number()},
          decoder_sparse_step: {"decoder_sparse_step", optional(number())},
          router_aux_loss_coef: {"router_aux_loss_coef", optional(number())},
          norm_topk_prob: {"norm_topk_prob", optional(boolean())},
          mlp_only_layers: {"mlp_only_layers", optional(list(number()))},
          output_router_logits: {"output_router_logits", optional(boolean())}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(spec) do
      # Base mappings (embeddings, norms, attention)
      base_mappings = %{
        "embedder.token_embedding" => "model.embed_tokens",
        "output_norm" => "model.norm",
        "language_modeling_head.output" =>
          if(spec.tie_word_embeddings, do: "model.embed_tokens", else: "lm_head"),
        "sequence_classification_head.output" => "score"
      }

      # Per-layer mappings (attention + FFN/MoE)
      layer_mappings =
        for n <- 0..(spec.num_blocks - 1), reduce: %{} do
          acc ->
            # Common attention mappings
            attention_mappings = %{
              "decoder.blocks.#{n}.self_attention.query" => "model.layers.#{n}.self_attn.q_proj",
              "decoder.blocks.#{n}.self_attention.key" => "model.layers.#{n}.self_attn.k_proj",
              "decoder.blocks.#{n}.self_attention.value" => "model.layers.#{n}.self_attn.v_proj",
              "decoder.blocks.#{n}.self_attention.output" => "model.layers.#{n}.self_attn.o_proj",
              "decoder.blocks.#{n}.self_attention.query_norm" =>
                "model.layers.#{n}.self_attn.q_norm",
              "decoder.blocks.#{n}.self_attention.key_norm" =>
                "model.layers.#{n}.self_attn.k_norm",
              "decoder.blocks.#{n}.self_attention_norm" => "model.layers.#{n}.input_layernorm",
              "decoder.blocks.#{n}.self_attention.rotary_embedding" =>
                "model.layers.#{n}.self_attn.rotary_emb",
              "decoder.blocks.#{n}.output_norm" => "model.layers.#{n}.post_attention_layernorm"
            }

            # FFN mappings: MoE or dense depending on layer
            ffn_mappings =
              if n in spec.mlp_only_layers do
                # Dense FFN
                %{
                  "decoder.blocks.#{n}.ffn.gate" => "model.layers.#{n}.mlp.gate_proj",
                  "decoder.blocks.#{n}.ffn.intermediate" => "model.layers.#{n}.mlp.up_proj",
                  "decoder.blocks.#{n}.ffn.output" => "model.layers.#{n}.mlp.down_proj"
                }
              else
                # MoE FFN
                moe_base_mappings = %{
                  "decoder.blocks.#{n}.moe.gate" => "model.layers.#{n}.mlp.gate"
                }

                # Add expert mappings
                expert_mappings =
                  for expert_idx <- 0..(spec.num_experts - 1), reduce: %{} do
                    expert_acc ->
                      Map.merge(expert_acc, %{
                        "decoder.blocks.#{n}.moe.experts.#{expert_idx}.gate_proj" =>
                          "model.layers.#{n}.mlp.experts.#{expert_idx}.gate_proj",
                        "decoder.blocks.#{n}.moe.experts.#{expert_idx}.up_proj" =>
                          "model.layers.#{n}.mlp.experts.#{expert_idx}.up_proj",
                        "decoder.blocks.#{n}.moe.experts.#{expert_idx}.down_proj" =>
                          "model.layers.#{n}.mlp.experts.#{expert_idx}.down_proj"
                      })
                  end

                Map.merge(moe_base_mappings, expert_mappings)
              end

            Map.merge(acc, Map.merge(attention_mappings, ffn_mappings))
        end

      Map.merge(base_mappings, layer_mappings)
    end
  end
end
