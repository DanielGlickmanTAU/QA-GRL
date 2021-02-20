from experiments import experiment
from config.ExperimentVariables import hyperparams

model_params = hyperparams.model_params
model_name = model_params.model_name
torch, experiment = experiment.start_experiment(tags=[hyperparams.env, model_name, hyperparams.task_name],
                                                hyperparams=hyperparams)

from train.training import get_trainer
from unittest import TestCase

from data import tasks


class Test(TestCase):
    def test_race_classification_params(self):
        params = tasks.task_to_params_getter[hyperparams.task_name]()
        change_dir = '' if hyperparams.use_unique_seperator_for_answer else '/using_sep'
        save_dir = params.benchmark_folder_name + '/' + model_name + change_dir
        print('saving to ', save_dir)

        metric_name = "accuracy"

        trainer = get_trainer(save_dir, hyperparams.model_params, params, True, experiment, metric_name,
                              hyperparams.disable_tqdm)

        # results = trainer.train(save_dir + '/checkpoint-84500')
        # results = trainer.train(save_dir + '/checkpoint-8474')
        results = trainer.train()
        print('done train')
        print(results)


if __name__ == '__main__':
    print('starting test')
    Test().test_race_classification_params()
