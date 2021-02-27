from utils import compute

torch = compute.get_torch()
from data.DatasetPostMapper import DataSetPostMapper
from transformers import TrainingArguments, Trainer
from config import ExperimentVariables
from config.ExperimentVariables import hyperparams
from data import datasets_loading
from data.TaskParams import TaskParams
from experiments import experiment
from utils.model_loading import get_last_model_and_tokenizer
from train import training
from data import boolq_utils
model_params = hyperparams.model_params
model_name = model_params.model_name
# torch, experiment = experiment.start_experiment(tags=[model_name, hyperparams.task_name],
#                                                 hyperparams=hyperparams)
from unittest import TestCase
import os


class Test(TestCase):
    def test_diff_between_models(self):
        task = 'boolq-classification'
        model_params = ExperimentVariables._distilbert_squad
        model_params.num_epochs = 0
        model, tokenizer = get_last_model_and_tokenizer(task, model_params)

        ds = datasets_loading.get_boolq_dataset(tokenizer)

        task_params = TaskParams(ds, model, tokenizer, 'trash')

        mapper = DataSetPostMapper(task_params)
        mapped_ds = ds.map(mapper.add_is_correct_and_probs, batched=True, batch_size=10, writer_batch_size=10)

        sorted_ds = mapped_ds.sort('probs')
        best = [boolq_utils.get_t_q_a(example) for example in sorted_ds['validation'][:5]]
        worst = [boolq_utils.get_t_q_a(example) for example in sorted_ds['validation'][-5:]]

        print('best: ', best)
        print('worst: ', worst)
