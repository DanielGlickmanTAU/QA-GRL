from experiments import experiment
from data.TaskParams import TaskParams
from models.question_generation import E2EQGPipeline
from utils import compute

torch = compute.get_torch()
from config.ExperimentVariables import hyperparams

torch, experiment = experiment.start_experiment(tags=['AWS', 't5-small', 'question-generation'],
                                                hyperparams=hyperparams)

from data import question_generation_dataset, data_collator

from config import ExperimentVariables
from config.ExperimentVariables import hyperparams
from utils.model_loading import get_model_and_tokenizer_for_qa_generation, get_save_path
from train import training

model_params = hyperparams.model_params

from unittest import TestCase


class Test(TestCase):
    def train_question_generating_model(self):
        task_name = 'question-generation'
        model_params = ExperimentVariables._t5_qg

        model, tokenizer = get_model_and_tokenizer_for_qa_generation(model_params)
        boolq = question_generation_dataset.get_processed_boolq_dataset(tokenizer)

        task_params = TaskParams(boolq, model, tokenizer, task_name)
        save_dir = get_save_path(task_name, model_params)

        trainer = training.get_generator_trainer(save_dir, model_params, task_params, load_best_model_at_end=True,
                                                 data_collator=data_collator.T2TDataCollator(tokenizer))
        trainer.train()

        pipe = E2EQGPipeline(model, tokenizer)
        for i in range(5):
            t = boolq['validation'][i]['source_text']
            print(t)
            print(pipe(t))
            print('\n')
