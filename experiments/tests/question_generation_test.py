from experiments import experiment
from data.TaskParams import TaskParams
from models import question_generation
from models.question_generation import QuestionGenerator
from utils import compute

torch = compute.get_torch()
from config.ExperimentVariables import hyperparams

torch, experiment = experiment.start_experiment(tags=['AWS', 't5-small', 'question-generation'],
                                                hyperparams=hyperparams)

from data import question_generation_dataset, data_collator, datasets_loading

from config import ExperimentVariables
from config.ExperimentVariables import hyperparams
import models.model_loading as model_loading
from train import training

model_params = hyperparams.model_params

from unittest import TestCase


class Test(TestCase):
    task_name = 'question-generation'
    model_params = ExperimentVariables._t5_qg
    save_dir = model_loading.get_save_path(task_name, model_params)

    def train_question_generating_model(self):
        model, tokenizer = model_loading.get_model_and_tokenizer_for_qa_generation(model_params)
        boolq = question_generation_dataset.get_processed_boolq_dataset(tokenizer)

        task_params = TaskParams(boolq, model, tokenizer, self.task_name)
        trainer = training.get_generator_trainer(self.save_dir, model_params, task_params, load_best_model_at_end=True,
                                                 data_collator=data_collator.T2TDataCollator(tokenizer))
        trainer.train()

    def test_run_trained_question_generation_model(self):
        model, tokenizer = model_loading.get_last_model_and_tokenizer(self.task_name, self.model_params)
        boolq = datasets_loading.get_boolq_generation_dataset(tokenizer)
        generator = QuestionGenerator(model, tokenizer)

        for i in range(20):
            t = boolq['validation'][i]['source_text']
            print(t)
            print(generator(t))
            print(boolq['validation'][i]['target_text'])
            print('\n')

    def test_generating_questions(self):
        qs = question_generation.generate_boolq_dataset(num_questions=10)
        print(qs)
