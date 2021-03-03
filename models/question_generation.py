import datasets
from transformers import PreTrainedModel, PreTrainedTokenizer

from data import datasets_loading
from utils import compute

_generate_question_prefix = 'generate questions:'

beam_search_args = {
    "max_length": 512,
    "num_beams": 8,
    "length_penalty": 1,
    "no_repeat_ngram_size": 3,
    "early_stopping": False,
    "num_return_sequences": 2
}

top_k_args = {
    "max_length": 512,
    "do_sample": True,
    "top_k": 50,
    "num_return_sequences": 2
}

top_p_args = {
    "max_length": 512,
    "do_sample": True,
    "top_k": 25,
    'top_p': 0.85,
    "num_return_sequences": 4
}


class QuestionGenerator:
    def __init__(
            self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.device = compute.get_device()
        self.model.to(self.device)

        self.default_generate_kwargs = beam_search_args

    def __call__(self, context: str, **generate_kwargs):
        inputs = self._prepare_inputs_for_e2e_qg(context)

        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs

        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device),
            attention_mask=inputs['attention_mask'].to(self.device),
            **generate_kwargs
        )

        return [self.tokenizer.decode(out, skip_special_tokens=True) for out in outs]

    def _prepare_inputs_for_e2e_qg(self, context):
        source_text = context
        if not context.startswith(_generate_question_prefix):
            source_text = f"generate questions: {context}"
        source_text = source_text + " </s>"

        inputs = self._tokenize([source_text], padding=False)
        return inputs

    def _tokenize(self, inputs, padding=True, truncation=True, add_special_tokens=True, max_length=512):
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


def generate_questions(model, tokenizer, boolq_generation_dataset, num_texts):
    """ input is boolq split(probably validation only)
    returns dicts in the form of boolq qa dataset: passage:list[str], question:list[str]
    texts may appear multiple times in passage, with different question in the questions list"""

    pipe = QuestionGenerator(model, tokenizer)
    generated_questions = {'passage': [], 'question': [], 'answer': []}

    for i in range(num_texts):
        text = boolq_generation_dataset[i]['source_text']
        clean_text = text[len(_generate_question_prefix) + 1:]

        questions = list(set(pipe(text)))
        generated_questions['passage'] += [clean_text] * len(questions)
        generated_questions['question'] += questions
        # hack: adding labels to avoid problems with processing down the road
        generated_questions['answer'] += [9] * len(questions)

    return datasets.Dataset.from_dict(generated_questions)


def generate_boolq_dataset(model, tokenizer, split='validation', num_questions=0):
    boolq_generation = datasets_loading.get_boolq_generation_dataset(tokenizer)

    split_ = boolq_generation[split]
    num_texts = num_questions if num_questions else len(split_)
    return generate_questions(model, tokenizer, split_, num_texts)
