from data.TaskParams import TaskParams
from train.training import get_trainer
from utils import compute

torch = compute.get_torch()
from transformers import PreTrainedModel, PreTrainedTokenizer

from data import question_generation_dataset, data_collator

from config import ExperimentVariables
from config.ExperimentVariables import hyperparams
from utils.model_loading import get_model_and_tokenizer_for_qa_generation, get_save_path

model_params = hyperparams.model_params

from unittest import TestCase


class Test(TestCase):
    def test_diff_between_models(self):
        task_name = 'question-generation'
        model_params = ExperimentVariables._t5_qg

        model_type = 't5'
        model, tokenizer = get_model_and_tokenizer_for_qa_generation(model_params)
        boolq = question_generation_dataset.get_processed_boolq_dataset(tokenizer)

        # original_texts = set(boolq['train']['source_text'])
        # boolq['validation'] = boolq['validation'].filter(lambda example: example['source_text'] not in original_texts)

        # pipe = E2EQGPipeline(model, tokenizer)
        # for i in range(5):
        #     t = boolq['validation'][i]['source_text']
        #     print(t)
        #     print(pipe(t))
        #     print('\n')

        metric_name = "accuracy"

        task_params = TaskParams(boolq, model, tokenizer, task_name)
        save_dir = get_save_path(task_name, model_params)
        trainer = get_trainer(save_dir, model_params, task_params, True, None, metric_name,
                              False, data_collator=data_collator.T2TDataCollator(tokenizer))
        trainer.train()
        print(3)

        pipe = E2EQGPipeline(model, tokenizer)
        for i in range(5):
            t = boolq['validation'][i]['source_text']
            print(t)
            print(pipe(t))
            print('\n')


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
            "max_length": 256,
            "num_beams": 4,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }

    def __call__(self, context: str, **generate_kwargs):
        inputs = self._prepare_inputs_for_e2e_qg(context)

        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs

        input_length = inputs["input_ids"].shape[-1]

        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device),
            attention_mask=inputs['attention_mask'].to(self.device),
            **generate_kwargs
        )

        prediction = self.tokenizer.decode(outs[0], skip_special_tokens=True)
        questions = prediction.split("<sep>")
        questions = [question.strip() for question in questions[:-1]]
        return questions

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
