from transformers import PreTrainedModel, PreTrainedTokenizer

from utils import compute


class E2EQGPipeline:
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
    ):

        self.model = model
        self.tokenizer = tokenizer

        self.device = compute.get_device()
        self.model.to(self.device)

        self.default_generate_kwargs = {
            "max_length": 512,
            "num_beams": 8,
            "length_penalty": 1,
            "no_repeat_ngram_size": 4,
            "early_stopping": False,
            "num_return_sequences": 2
        }

    def __call__(self, context: str, **generate_kwargs):
        inputs = self._prepare_inputs_for_e2e_qg(context)

        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs

        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device),
            attention_mask=inputs['attention_mask'].to(self.device),
            **generate_kwargs
        )

        predictions = [self.tokenizer.decode(out, skip_special_tokens=True) for out in outs]
        return predictions

    def _prepare_inputs_for_e2e_qg(self, context):
        source_text = context
        if not context.startswith("generate questions:"):
            source_text = f"generate questions: {context}"
        source_text = source_text + " </s>"

        inputs = self._tokenize([source_text], padding=False)
        return inputs

    def _tokenize(
            self,
            inputs,
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=512
    ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs
