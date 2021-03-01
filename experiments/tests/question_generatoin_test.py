from utils import compute
from data import question_generation_dataset

torch = compute.get_torch()
from config import ExperimentVariables
from config.ExperimentVariables import hyperparams
from utils.model_loading import get_model_and_tokenizer_for_qa_generation

model_params = hyperparams.model_params

from unittest import TestCase


class Test(TestCase):
    def test_diff_between_models(self):
        task = 'boolq-classification'
        model_params = ExperimentVariables._t5_qg

        model_type = 't5'
        model, tokenizer = get_model_and_tokenizer_for_qa_generation(model_params)
        boolq = question_generation_dataset.get_processed_boolq_dataset(tokenizer)

        print(tokenizer)
