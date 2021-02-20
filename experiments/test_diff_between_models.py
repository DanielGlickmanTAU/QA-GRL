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
        model_params = ExperimentVariables._distilbert_squad
        task = 'race-classification'
        print(os.getcwd())
        model, tokenizer = get_last_model_and_tokenizer(task, model_params)
        ds = datasets_loading.get_race_dataset(tokenizer)

        change_dir = '' if hyperparams.use_unique_seperator_for_answer else '/using_sep'
        save_dir = task + '/' + model_name + change_dir

        # args = TrainingArguments(
        #     '/trash',
        #     learning_rate=hyperparams.model_params.learning_rate,
        #     per_device_train_batch_size=hyperparams.model_params.batch_size,
        #     per_device_eval_batch_size=hyperparams.model_params.batch_size,
        #     num_train_epochs=0,
        # )
        #
        # trainer = Trainer(
        #     model,
        #     args,
        #     train_dataset=ds["train"],
        #     eval_dataset=ds["validation"],
        #     tokenizer=tokenizer,
        #     # compute_metrics=compute_metrics
        # )

        hyperparams.model_params.num_epochs = 0
        hyperparams.model_params.batch_size = 1
        task_params = TaskParams(ds,model,tokenizer,'trash')
        trainer = training.get_trainer('trash', hyperparams.model_params, task_params, False, experiment=experiment)

        out = trainer.predict(ds['test'])
        print(model)

        # model_params = ExperimentVariables._roberta_squad
        # model, tokenizer = get_last_model_and_tokenizer(task, model_params)
