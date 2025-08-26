defmodule Bumblebee.Text.TextEmbedding do
  @moduledoc false

  alias Bumblebee.Shared

  def text_embedding(model_info, tokenizer, opts \\ []) do
    %{model: model, params: params, spec: _spec} = model_info

    opts =
      Keyword.validate!(opts, [
        :compile,
        output_attribute: :pooled_state,
        output_pool: nil,
        embedding_processor: nil,
        defn_options: [],
        preallocate_params: false
      ])

    output_attribute = opts[:output_attribute]
    output_pool = opts[:output_pool]
    embedding_processor = opts[:embedding_processor]
    preallocate_params = opts[:preallocate_params]
    defn_options = opts[:defn_options]

    compile =
      if compile = opts[:compile] do
        compile
        |> Keyword.validate!([:batch_size, :sequence_length])
        |> Shared.require_options!([:batch_size, :sequence_length])
      end

    batch_size = compile[:batch_size]
    sequence_length = compile[:sequence_length]

    tokenizer =
      Bumblebee.configure(tokenizer, length: sequence_length, return_token_type_ids: false)

    {_init_fun, encoder} = Axon.build(model)

    embedding_fun = fn params, inputs ->
      output = encoder.(params, inputs)

      output =
        case output do
          %{^output_attribute => output} ->
            output

          %{} ->
            keys = output |> Map.keys() |> Enum.sort()

            raise ArgumentError,
                  "key #{inspect(output_attribute)} not found in the output map," <>
                    " you may want to set :output_attribute to one of the map keys: #{inspect(keys)}"

          _ ->
            output
        end

      if output_pool != nil and Nx.rank(output) != 3 do
        raise ArgumentError,
              "expected the output tensor to have rank 3 to apply :output_pool, got: #{Nx.rank(output)}." <>
                " You should either disable pooling or pick a different output using :output_attribute"
      end

      output =
        case output_pool do
          nil ->
            output

          :cls_token_pooling ->
            Nx.take(output, 0, axis: 1)

          :mean_pooling ->
            input_mask_expanded = Nx.new_axis(inputs["attention_mask"], -1)

            output
            |> Nx.multiply(input_mask_expanded)
            |> Nx.sum(axes: [1])
            |> Nx.divide(Nx.sum(input_mask_expanded, axes: [1]))

          :last_token_pooling ->
            last_token_pool(output, inputs["attention_mask"])

          other ->
            raise ArgumentError,
                  "expected :output_pool to be one of :cls_token_pooling, :mean_pooling, :last_token_pooling or nil, got: #{inspect(other)}"
        end

      output =
        case embedding_processor do
          nil ->
            output

          :l2_norm ->
            Bumblebee.Utils.Nx.normalize(output)

          other ->
            raise ArgumentError,
                  "expected :embedding_processor to be one of nil or :l2_norm, got: #{inspect(other)}"
        end

      output
    end

    batch_keys = Shared.sequence_batch_keys(sequence_length)

    Nx.Serving.new(
      fn batch_key, defn_options ->
        params = Shared.maybe_preallocate(params, preallocate_params, defn_options)

        scope = {:embedding, batch_key}

        embedding_fun =
          Shared.compile_or_jit(embedding_fun, scope, defn_options, compile != nil, fn ->
            {:sequence_length, sequence_length} = batch_key

            inputs = %{
              "input_ids" => Nx.template({batch_size, sequence_length}, :u32),
              "attention_mask" => Nx.template({batch_size, sequence_length}, :u32)
            }

            [params, inputs]
          end)

        fn inputs ->
          inputs = Shared.maybe_pad(inputs, batch_size)
          embedding_fun.(params, inputs) |> Shared.serving_post_computation()
        end
      end,
      defn_options
    )
    |> Nx.Serving.batch_size(batch_size)
    |> Nx.Serving.process_options(batch_keys: batch_keys)
    |> Nx.Serving.client_preprocessing(fn input ->
      {texts, multi?} = Shared.validate_serving_input!(input, &Shared.validate_string/1)

      inputs =
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
          Bumblebee.apply_tokenizer(tokenizer, texts)
        end)

      batch_key = Shared.sequence_batch_key_for_inputs(inputs, sequence_length)
      batch = [inputs] |> Nx.Batch.concatenate() |> Nx.Batch.key(batch_key)

      {batch, multi?}
    end)
    |> Nx.Serving.client_postprocessing(fn {embeddings, _metadata}, multi? ->
      for embedding <- Bumblebee.Utils.Nx.batch_to_list(embeddings) do
        %{embedding: embedding}
      end
      |> Shared.normalize_output(multi?)
    end)
  end

  defp last_token_pool(hidden_states, attention_mask) do
    # Check if left padding is used by seeing if all sequences end with a valid token
    last_column = Nx.slice_along_axis(attention_mask, -1, 1, axis: 1) |> Nx.squeeze(axes: [1])
    batch_size = Nx.axis_size(attention_mask, 0)
    left_padding = Nx.sum(last_column) |> Nx.equal(batch_size)

    if Nx.to_number(left_padding) == 1 do
      # Left padding: take the last token (index -1)
      Nx.slice_along_axis(hidden_states, -1, 1, axis: 1)
      |> Nx.squeeze(axes: [1])
    else
      # Right padding: find the last non-padded token for each sequence
      sequence_lengths = 
        attention_mask
        |> Nx.sum(axes: [1])
        |> Nx.subtract(1)  # Convert from length to 0-based index

      # Use gather to get the last token for each sequence  
      hidden_states
      |> Nx.to_batched(1)
      |> Enum.zip(Nx.to_list(sequence_lengths))
      |> Enum.map(fn {batch_hidden, seq_len} ->
        Nx.slice_along_axis(batch_hidden, seq_len, 1, axis: 1) |> Nx.squeeze(axes: [0, 1])
      end)
      |> Nx.stack()
    end
  end
end
