defmodule Bumblebee.Text.Qwen3 do
  alias Bumblebee.Shared

  options =
    [
      vocab_size: [
        default: 151669,
        doc: """
        the vocabulary size of the token embedding. This corresponds to the number of distinct
        tokens that can be represented in model input and output
        """
      ],
      max_positions: [
        default: 32768,
        doc: """
        the vocabulary size of the position embedding. This corresponds to the maximum sequence
        length that this model can process. Typically this is set to a large value just in case,
        such as 512, 1024 or 2048
        """
      ],
      hidden_size: [
        default: 1024,
        doc: "the dimensionality of hidden layers"
      ],
      intermediate_size: [
        default: 3072,
        doc: "the dimensionality of intermediate layers"
      ],
      attention_head_size: [
        default: nil,
        doc: """
        the size of the key, value, and query projection per attention head.
        Defaults to `div(hidden_size, num_attention_heads)`
        """
      ],
      num_blocks: [
        default: 28,
        doc: "the number of Transformer blocks in the model"
      ],
      num_attention_heads: [
        default: 16,
        doc: "the number of attention heads for each attention layer in the model"
      ],
      num_key_value_heads: [
        default: 8,
        doc: "the number of key value heads for each attention layer in the model"
      ],
      activation: [
        default: :silu,
        doc: "the activation function"
      ],
      rotary_embedding_base: [
        default: 1_000_000,
        doc: "base for computing rotary embedding frequency"
      ],
      rotary_embedding_scaling_strategy: [
        default: nil,
        doc: """
        scaling configuration for rotary embedding. Currently the supported values are:

          * `%{type: :linear, factor: number()}`

          * `%{type: :dynamic, factor: number()}`

          * `%{type: :llama3, factor: number(), low_frequency_factor: number(), high_frequency_factor: number(), original_max_positions: pos_integer()}`

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
      ]
    ] ++
      Shared.common_options([:output_hidden_states, :output_attentions, :num_labels, :id_to_label]) ++ Shared.token_options(pad_token_id: 151643)

  @moduledoc """
  Qwen3 model family.

  ## Architectures

    * `:base` - plain Qwen3 without any head on top

    * `:for_causal_language_modeling` - Qwen3 with a language modeling
      head. The head returns logits for each token in the original
      sequence

    * `:for_sequence_classification` - Qwen3 with a sequence
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
  """

  defstruct [architecture: :base] ++ Shared.option_defaults(options)

  @behaviour Bumblebee.ModelSpec
  @behaviour Bumblebee.Configurable
  @behaviour Bumblebee.Text.Generation

  import Bumblebee.Utils.Model, only: [join: 2]

  alias Bumblebee.Layers

  @impl true
  def architectures(), do: [:base, :for_causal_language_modeling, :for_sequence_classification]

  @impl true
  def config(spec, opts) do
    spec
    |> Shared.put_config_attrs(opts)
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
    outputs = core(inputs, spec)

    # For base architecture, only return the final hidden state and optionally hidden_states
    base_output = %{
      hidden_state: outputs.hidden_state,
      cache: outputs.cache
    }
    
    base_output = 
      if spec.output_attentions do
        Map.put(base_output, :attentions, outputs.attentions)
      else
        base_output
      end
    
    base_output = 
      if spec.output_hidden_states do
        Map.put(base_output, :hidden_states, outputs.hidden_states)
      else
        base_output
      end

    Layers.output(base_output)
  end

  def model(%__MODULE__{architecture: :for_causal_language_modeling} = spec) do
    inputs = inputs(spec)

    outputs = core(inputs, spec)

    logits =
      outputs.hidden_state
      |> Axon.dense(spec.vocab_size,
        use_bias: false,
        kernel_initializer: kernel_initializer(spec),
        name: "language_modeling_head.output"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions,
      cache: outputs.cache
    })
  end

  def model(%__MODULE__{architecture: :for_sequence_classification} = spec) do
    inputs = inputs(spec)

    outputs = core(inputs, spec)

    logits =
      outputs.hidden_state
      |> Layers.take_token(index: -1, axis: 1)
      |> Axon.dense(spec.num_labels,
        kernel_initializer: kernel_initializer(spec),
        name: "sequence_classification_head.output"
      )

    Layers.output(%{
      logits: logits,
      hidden_states: outputs.hidden_states,
      attentions: outputs.attentions
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
        inputs["attention_mask"],
        position_ids,
        inputs["attention_head_mask"],
        inputs["cache"],
        spec
      )

    pooled_state =
      decoder_outputs.hidden_state
      |> Layers.rms_norm(
        name: "norm",
        epsilon: spec.layer_norm_epsilon
      )

    %{
      hidden_state: pooled_state,
      hidden_states: Layers.append(decoder_outputs.hidden_states, pooled_state),
      attentions: decoder_outputs.attentions,
      cache: decoder_outputs.cache
    }
  end

  defp decoder(
         hidden_state,
         attention_mask,
         position_ids,
         attention_head_mask,
         cache,
         spec
       ) do
    Layers.Transformer.blocks(hidden_state,
      attention_mask: attention_mask,
      attention_head_mask: attention_head_mask,
      attention_head_size: spec.attention_head_size,
      cache: cache,
      num_blocks: spec.num_blocks,
      num_attention_heads: spec.num_attention_heads,
      num_key_value_heads: spec.num_key_value_heads,
      hidden_size: spec.hidden_size,
      kernel_initializer: kernel_initializer(spec),
      layer_norm: &Layers.rms_norm(&1, name: &2, epsilon: spec.layer_norm_epsilon),
      ffn:
        &gated_ffn(&1, spec.intermediate_size, spec.hidden_size,
          name: &2,
          activation: spec.activation
        ),
      block_type: :norm_first,
      causal: true,
      rotary_embedding: [
        position_ids: position_ids,
        max_positions: spec.max_positions,
        base: spec.rotary_embedding_base,
        scaling_strategy: spec.rotary_embedding_scaling_strategy
      ],
      query_use_bias: false,
      key_use_bias: false,
      value_use_bias: false,
      output_use_bias: false,
      name: "decoder"
    )
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

  defp embedder(input_ids, input_embeddings, spec, opts) do
    name = opts[:name]

    # TODO: Axon needs a way to specify ignoring pad tokens
    # in gradient
    Layers.default input_embeddings do
      Axon.embedding(input_ids, spec.vocab_size, spec.hidden_size,
        kernel_initializer: kernel_initializer(spec),
        name: join(name, "token_embedding")
      )
    end
  end

  defp kernel_initializer(spec) do
    Axon.Initializers.normal(scale: spec.initializer_scale)
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(spec, data) do
      import Shared.Converters

      opts =
        convert!(data,
          vocab_size: {"vocab_size", number()},
          max_positions: {"max_position_embeddings", number()},
          hidden_size: {"hidden_size", number()},
          num_blocks: {"num_hidden_layers", number()},
          num_attention_heads: {"num_attention_heads", number()},
          num_key_value_heads: {"num_key_value_heads", number()},
          attention_head_size: {"head_dim", number()},
          intermediate_size: {"intermediate_size", number()},
          layer_norm_epsilon: {"rms_norm_eps", number()},
          rotary_embedding_base: {"rope_theta", number()},
          initializer_scale: {"initializer_range", number()},
          tie_word_embeddings: {"tie_word_embeddings", boolean()}
        ) ++ Shared.common_options_from_transformers(data, spec)

      @for.config(spec, opts)
    end
  end

  defimpl Bumblebee.HuggingFace.Transformers.Model do
    def params_mapping(_spec) do
      %{
        "qwen3.embedder.token_embedding" => "model.embed_tokens",
        "qwen3.decoder.{n}.self_attention.query" => "model.layers.{n}.self_attn.q_proj",
        "qwen3.decoder.{n}.self_attention.key" => "model.layers.{n}.self_attn.k_proj",
        "qwen3.decoder.{n}.self_attention.value" => "model.layers.{n}.self_attn.v_proj",
        "qwen3.decoder.{n}.self_attention.output" => "model.layers.{n}.self_attn.o_proj",
        "qwen3.decoder.{n}.self_attention.rotary_embedding" =>
          "model.layers.{n}.self_attn.rotary_emb",
        "qwen3.decoder.{n}.ffn.gate" => "model.layers.{n}.mlp.gate_proj",
        "qwen3.decoder.{n}.ffn.intermediate" => "model.layers.{n}.mlp.up_proj",
        "qwen3.decoder.{n}.ffn.output" => "model.layers.{n}.mlp.down_proj",
        "qwen3.decoder.{n}.output_norm" => "model.layers.{n}.input_layernorm",
        "qwen3.decoder.{n}.self_attention_norm" =>
          "model.layers.{n}.post_attention_layernorm",
        "qwen3.norm" => "model.norm",
        "qwen3.language_modeling_head.output" => "lm_head",
        "qwen3.sequence_classification_head.output" => "score"
      }
    end
  end
end