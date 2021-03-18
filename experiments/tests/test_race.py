from experiments import experiment
from config.ExperimentVariables import hyperparams
import config.ExperimentVariables as variables
from models.model_loading import get_save_path

from train.training import get_trainer
from unittest import TestCase

from data import tasks


class Test(TestCase):
    def test_race_classification_params(self):
        model_params = variables._roberta_squad.clone()
        model_name = model_params.model_name
        torch, exp = experiment.start_experiment(tags=[hyperparams.env, model_name, hyperparams.task_name],
                                                 hyperparams=hyperparams)

        params = tasks.task_to_params_getter['boolq'](model_params.model_name, model_params.model_tokenizer)
        save_dir = get_save_path(params.benchmark_folder_name, model_params)
        print('saving to ', save_dir)

        metric_name = "accuracy"

        model_params.num_epochs = 3
        trainer = get_trainer(save_dir, hyperparams.model_params, params, True, exp, metric_name,
                              hyperparams.disable_tqdm)

        # results = trainer.train(save_dir + '/checkpoint-84500')
        # results = trainer.train(save_dir + '/checkpoint-8474')
        results = trainer.train()
        print('done train')
        print(results)


if __name__ == '__main__':
    print('starting test')
    Test().test_race_classification_params()
