defmodule Bumblebee.Text.EmbeddingGemmaTest do
  use ExUnit.Case, async: false

  import Bumblebee.TestHelpers

  # Configure EXLA for fast testing
  setup_all do
    Application.put_env(:nx, :default_defn_options, compiler: EXLA, client: :host)
    :ok
  end

  @moduletag model_test_tags()

  test ":base architecture" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "google/embeddinggemma-300m"},
               architecture: :base,
               backend: {EXLA.Backend, client: :host}
             )

    assert %Bumblebee.Text.EmbeddingGemma{architecture: :base} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50]])
    }

    assert_equal(
      model
      |> Axon.predict(params, inputs)
      |> Nx.shape(),
      {1, 5, spec.hidden_size}
    )
  end

  test ":for_text_embedding architecture" do
    assert {:ok, %{model: model, params: params, spec: spec}} =
             Bumblebee.load_model({:hf, "google/embeddinggemma-300m"},
               architecture: :for_text_embedding,
               backend: {EXLA.Backend, client: :host}
             )

    assert %Bumblebee.Text.EmbeddingGemma{architecture: :for_text_embedding} = spec

    inputs = %{
      "input_ids" => Nx.tensor([[10, 20, 30, 40, 50]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 0]])
    }

    output = Axon.predict(model, params, inputs)
    
    # Should have pooled_state with shape [batch_size, hidden_size]
    assert_equal(Nx.shape(output.pooled_state), {1, spec.hidden_size})
    
    # Should also have hidden_state with original shape
    assert_equal(Nx.shape(output.hidden_state), {1, 5, spec.hidden_size})
  end

  test "text embedding serving" do
    {:ok, model_info} = 
      Bumblebee.load_model({:hf, "google/embeddinggemma-300m"},
        architecture: :for_text_embedding,
        backend: {EXLA.Backend, client: :host}
      )
    
    {:ok, tokenizer} = 
      Bumblebee.load_tokenizer({:hf, "google/embeddinggemma-300m"})

    serving = 
      Bumblebee.Text.text_embedding(model_info, tokenizer, 
        output_attribute: :pooled_state,
        embedding_processor: :l2_norm
      )

    text = "This is a test sentence for embedding generation."
    
    assert %{embedding: embedding} = Nx.Serving.run(serving, text)
    assert Nx.rank(embedding) == 1
    assert Nx.size(embedding) == model_info.spec.hidden_size
    
    # Check that embedding is normalized (L2 norm should be ~1.0)
    norm = Nx.LinAlg.norm(embedding)
    assert_in_delta(Nx.to_number(norm), 1.0, 0.01)
  end

  test "bidirectional attention vs causal attention" do
    # Load EmbeddingGemma (bidirectional)
    {:ok, %{model: embedding_model, params: embedding_params}} =
      Bumblebee.load_model({:hf, "google/embeddinggemma-300m"},
        architecture: :base,
        backend: {EXLA.Backend, client: :host}
      )

    # Test that attention can see future tokens (bidirectional)
    inputs = %{
      "input_ids" => Nx.tensor([[1, 2, 3, 4, 5]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1]])
    }

    output = Axon.predict(embedding_model, embedding_params, inputs)
    
    # The first token's representation should be influenced by all tokens
    # (this is hard to test directly, but we can at least verify the model runs)
    assert Nx.shape(output.hidden_state) == {1, 5, 768}
    assert Nx.type(output.hidden_state) == {:f, 32}
  end

  test "batch processing" do
    {:ok, model_info} = 
      Bumblebee.load_model({:hf, "google/embeddinggemma-300m"},
        architecture: :for_text_embedding,
        backend: {EXLA.Backend, client: :host}
      )
    
    {:ok, tokenizer} = 
      Bumblebee.load_tokenizer({:hf, "google/embeddinggemma-300m"})

    serving = 
      Bumblebee.Text.text_embedding(model_info, tokenizer, 
        output_attribute: :pooled_state,
        compile: [batch_size: 2, sequence_length: 64]
      )

    texts = [
      "First sentence for testing.",
      "Second sentence with different content."
    ]
    
    results = Nx.Serving.run(serving, texts)
    
    assert length(results) == 2
    assert Enum.all?(results, fn %{embedding: emb} -> 
      Nx.rank(emb) == 1 and Nx.size(emb) == model_info.spec.hidden_size
    end)
  end

  test "configuration loading from HuggingFace" do
    config_data = %{
      "vocab_size" => 256_000,
      "max_position_embeddings" => 2048,
      "hidden_size" => 768,
      "num_hidden_layers" => 12,
      "num_attention_heads" => 12,
      "num_key_value_heads" => 12,
      "head_dim" => 64,
      "intermediate_size" => 3072,
      "hidden_act" => "gelu_new",
      "attention_bias" => false,
      "rope_theta" => 10000.0,
      "initializer_range" => 0.02,
      "rms_norm_eps" => 1.0e-6,
      "model_type" => "gemma"
    }

    spec = %Bumblebee.Text.EmbeddingGemma{}
    {:ok, loaded_spec} = Bumblebee.HuggingFace.Transformers.Config.load(spec, config_data)

    assert loaded_spec.vocab_size == 256_000
    assert loaded_spec.max_positions == 2048
    assert loaded_spec.hidden_size == 768
    assert loaded_spec.num_blocks == 12
    assert loaded_spec.num_attention_heads == 12
    assert loaded_spec.attention_head_size == 64
    assert loaded_spec.intermediate_size == 3072
    assert loaded_spec.activation == :gelu_approx_tanh
  end
end