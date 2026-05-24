defmodule Bumblebee.Multimodal.ImageTextToText do
  @moduledoc """
  Generation helper for vision-language models like Qwen3-VL.

  This wraps featurization, prompt expansion, and `Bumblebee.Text.Generation`
  in a single call. Each call recompiles the generation graph if the
  image or prompt produces a different total patch count or sequence
  length, which makes this best suited for interactive or one-shot use.
  For high-throughput serving with batched, varying image sizes, see
  the static-shape padding follow-up.
  """

  alias Bumblebee.Text

  @placeholder "<|image_pad|>"

  @doc """
  Generates text from a prompt that includes a `<|image_pad|>` marker
  and an image.

  ## Required arguments

    * `model_info` - a loaded `Bumblebee.Multimodal.Qwen3VL` (or compatible)
      model
    * `featurizer` - a configured `Bumblebee.Vision.Qwen3VLFeaturizer`
    * `tokenizer` - a loaded tokenizer for the same model
    * `generation_config` - a `Bumblebee.Text.GenerationConfig`
    * `text` - the user prompt containing exactly one `<|image_pad|>` marker
    * `image` - an image tensor or `t:StbImage.t/0`

  ## Returns

      %{text: "<generated text>", token_ids: [...]}

  ## Example

      {:ok, model_info} = Bumblebee.load_model({:hf, "Qwen/Qwen3-VL-2B-Instruct"})
      {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-VL-2B-Instruct"})

      {:ok, featurizer} =
        Bumblebee.load_featurizer({:hf, "Qwen/Qwen3-VL-2B-Instruct"},
          module: Bumblebee.Vision.Qwen3VLFeaturizer
        )

      featurizer = Bumblebee.configure(featurizer, quality: :low)
      {:ok, gen_config} = Bumblebee.load_generation_config({:hf, "Qwen/Qwen3-VL-2B-Instruct"})
      gen_config = Bumblebee.configure(gen_config, max_new_tokens: 64)

      Bumblebee.Multimodal.ImageTextToText.generate(
        model_info, featurizer, tokenizer, gen_config,
        "<|im_start|>user\\n<|vision_start|><|image_pad|><|vision_end|>What is in this image?<|im_end|>\\n<|im_start|>assistant\\n",
        image
      )
  """
  def generate(
        model_info,
        featurizer,
        tokenizer,
        %Text.GenerationConfig{} = generation_config,
        text,
        image
      ) do
    %{model: model, params: params, spec: spec} = model_info

    unless Map.has_key?(spec, :image_token_id) do
      raise ArgumentError,
            "expected a multimodal model with :image_token_id, got #{inspect(spec.__struct__)}"
    end

    merge_size =
      case spec do
        %{vision_spec: %{spatial_merge_size: ms}} -> ms
        _ -> 1
      end

    image_inputs = Bumblebee.apply_featurizer(featurizer, image)
    visual_tokens = visual_tokens_for(image_inputs["image_grid_thw"], merge_size)
    expanded_text = expand_marker(text, visual_tokens)

    tokenizer = Bumblebee.configure(tokenizer, return_token_type_ids: false)
    text_inputs = Bumblebee.apply_tokenizer(tokenizer, expanded_text)

    inputs =
      text_inputs
      |> Map.merge(image_inputs)
      |> Map.put("seed", Nx.tensor([:erlang.system_time()], type: :s64))

    generate_fun = Text.Generation.build_generate(model, spec, generation_config)
    %{token_ids: token_ids} = generate_fun.(params, inputs)

    decoded =
      token_ids
      |> Nx.to_batched(1)
      |> Enum.map(&Bumblebee.Tokenizer.decode(tokenizer, Nx.to_flat_list(&1)))
      |> hd()

    %{text: decoded, token_ids: token_ids}
  end

  defp expand_marker(text, visual_tokens) do
    case String.split(text, @placeholder) do
      [_only] ->
        raise ArgumentError,
              "the prompt must contain a #{@placeholder} marker where the image " <>
                "embedding should be spliced in, got: #{inspect(text)}"

      [prefix, suffix] ->
        prefix <> String.duplicate(@placeholder, visual_tokens) <> suffix

      _multiple ->
        raise ArgumentError,
              "expected exactly one #{@placeholder} marker in the prompt"
    end
  end

  defp visual_tokens_for(grid_thw, merge_size) do
    grid_thw
    |> Nx.to_list()
    |> Enum.map(fn [t, h, w] ->
      t * div(h, merge_size) * div(w, merge_size)
    end)
    |> Enum.sum()
  end
end
