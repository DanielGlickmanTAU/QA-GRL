from transformers import PreTrainedModel, PreTrainedTokenizer

from config import ExperimentVariables
from data import datasets_loading

from utils import compute, model_loading
import datasets

_generate_question_prefix = 'generate questions:'


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
        if not context.startswith(_generate_question_prefix):
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


def generate_questions(model, tokenizer, boolq_generation_dataset, num_texts):
    """ input is boolq split(probably validation only)
    returns dicts in the form of boolq qa dataset: passage:list[str], question:list[str]
    texts may appear multiple times in passage, with different question in the questions list"""

    pipe = E2EQGPipeline(model, tokenizer)
    generated_questions = {'passage': [], 'question': [], 'label': []}

    for i in range(num_texts):
        text = boolq_generation_dataset[i]['source_text']
        clean_text = text[len(_generate_question_prefix) + 1:]

        questions = pipe(text)
        generated_questions['passage'] += [clean_text] * len(questions)
        generated_questions['question'] += questions
        #hack: adding labels to avoid problems with processing down the road
        generated_questions['label'] = [9] * len(questions)

    return datasets.Dataset.from_dict(generated_questions)


def generate_boolq_dataset(split='validation', num_questions=0):
    generation_task_name = 'question-generation'

    generation_model_params = ExperimentVariables._t5_qg

    model, tokenizer = model_loading.get_last_model_and_tokenizer(generation_task_name, generation_model_params)
    boolq_generation = datasets_loading.get_boolq_generation_dataset(tokenizer)

    split_ = boolq_generation[split]
    num_texts = num_questions if num_questions else len(split_)
    generated_questions = generate_questions(model, tokenizer, split_, num_texts)
    return generated_questions.map(
        lambda examples: tokenizer(examples['passage'], examples['question'], truncation=True,
                                   padding=True)
    )
