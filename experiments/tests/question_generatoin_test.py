from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

from experiments import experiment
from collections import defaultdict

from datasets import load_from_disk

from train.training import get_trainer
from utils import compute

torch = compute.get_torch()
from data.DatasetPostMapper import DataSetPostMapper
from config import ExperimentVariables
from config.ExperimentVariables import hyperparams
from data import datasets_loading
from data.TaskParams import TaskParams
from utils.model_loading import get_last_model_and_tokenizer, get_save_path
from utils import model_loading
from data import boolq_utils

model_params = hyperparams.model_params

from unittest import TestCase


class Test(TestCase):
    def test_diff_between_models(self):
        task = 'boolq-classification'
        model_params = ExperimentVariables._t5_qg

        model_type = 't5'
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_params.model_name,
            # cache_dir=,
        )

        tokenizer.add_tokens(['<sep>', '<hl>'])
        model.resize_token_embeddings(len(tokenizer))

        print(tokenizer)
