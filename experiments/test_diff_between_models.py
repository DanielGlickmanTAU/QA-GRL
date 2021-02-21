from utils import compute
torch = compute.get_torch()
from transformers import TrainingArguments, Trainer
from config import ExperimentVariables
from config.ExperimentVariables import hyperparams
from data import datasets_loading
from data.TaskParams import TaskParams
from experiments import experiment
from utils.model_loading import get_last_model_and_tokenizer
from train import training

model_params = hyperparams.model_params
model_name = model_params.model_name
# torch, experiment = experiment.start_experiment(tags=[model_name, hyperparams.task_name],
#                                                 hyperparams=hyperparams)
from unittest import TestCase
import os


class Test(TestCase):
    def test_diff_between_models(self):
        task = 'race-classification'
        model_params = ExperimentVariables._distilbert_squad
        model_params.num_epochs = 0
        model, tokenizer = get_last_model_and_tokenizer(task, model_params)
        ds = datasets_loading.get_race_dataset(tokenizer)

        task_params = TaskParams(ds, model, tokenizer, 'trash')
        trainer = training.get_trainer('trash', model_params, task_params, False)
                                       #, experiment=experiment)
        m1_out_train = trainer.predict(ds['train'])
        m1_out_valid = trainer.predict(ds['validation'])
        m1_out_train = trainer.predict(ds['train'])
        print(m1_out_valid)
        """PredictionOutput(predictions=array([[-0.7137633 ,  0.6313474 ],
       [-0.05348547,  0.16040793],
       [-0.79939294,  0.72003627],
       ...,
       [-1.5532022 ,  1.2381634 ],
       [-2.634107  ,  2.0046153 ],
       [-0.35718027,  0.42043307]], dtype=float32), label_ids=array([0, 1, 1, ..., 0, 1, 0]), metrics={'eval_loss': 0.44448697566986084, 'eval_accuracy': 0.7750875260611305})"""

        change_dir = '' if hyperparams.use_unique_seperator_for_answer else '/using_sep'
        save_dir = task + '/' + model_name + change_dir
        # model_params = ExperimentVariables._roberta_squad
        # model, tokenizer = get_last_model_and_tokenizer(task, model_params)
