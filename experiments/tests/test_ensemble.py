import os

from experiments import experiment
from datasets import load_from_disk

from data.DatasetPostMapper import DataSetPostMapper
from config.ExperimentVariables import hyperparams
import config.ExperimentVariables as variables
from models.model_loading import get_save_path, get_best_model_and_tokenizer

from train.training import get_trainer
from unittest import TestCase

from data import tasks, datasets_loading
import gc

import torch


class Test(TestCase):
    # def train_boolq_test(self):
    #     model_params = variables._distilbert_squad.clone()
    #     task_name = 'boolq@1'
    #     model_params.num_epochs = 1
    #
    #     self.run_exp(model_params, task_name)

    def test_mark_prob_being_correct(self):
        def aggregate_scores():
            pass

        tasks = ['boolq-classification1', 'boolq-classification2', 'boolq-classification3', 'boolq-classification4',
                 'boolq-classification5']
        model_params = variables._distilbert_squad.clone()

        dataset = None
        for task in tasks:
            path = get_save_path(task, model_params)
            answer_model, answer_tokenizer = get_best_model_and_tokenizer(task, model_params)
            # first time, load unprocessed dataset
            if not dataset:
                dataset = datasets_loading.get_boolq_dataset(answer_tokenizer, remove_duplicates=False)
                dataset = dataset['validation']
            dataset = self.process_dataset(answer_model, answer_tokenizer, path, dataset)
            print('avg being correct:', sum(dataset[path]) / len(dataset[path]))
            print('avg being correct when label is 0:',
                  sum([x for x, l in zip(dataset[path], dataset['label']) if l == 0]) / dataset['label'].count(0))

    def disabledtest_train_boolq_multiplem_models(self):
        model_params = variables._roberta_squad.clone()
        done_txt_file = 'so_far.txt' if 'distilbert' in model_params.model_name \
            else 'so_far_roberta.txt' if 'roberta' in model_params.model_name else None

        hyperparams.model_params = model_params
        model_params.num_epochs = 2
        tasks = ['boolq@1', 'boolq@2', 'boolq@3']
        try:
            with open(done_txt_file, 'r') as f:
                so_far = f.read()
        except:
            so_far = []
        with open(done_txt_file, 'a') as f:
            for task in tasks:
                if task not in so_far:
                    self.run_exp(model_params, task)
                    f.write(task)
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    print('skipping', task)

    def run_exp(self, model_params, task_name):
        model_name = model_params.model_name
        torch, exp = experiment.start_experiment(tags=[hyperparams.env, model_name, task_name],
                                                 hyperparams=hyperparams)
        params = tasks.get_task_params(task_name, model_params.model_name, model_params.model_tokenizer)
        save_dir = get_save_path(params.benchmark_folder_name, model_params)
        print('saving to ', save_dir)
        metric_name = "accuracy"
        trainer = get_trainer(save_dir, hyperparams.model_params, params, True, exp, metric_name,
                              hyperparams.disable_tqdm)
        results = trainer.train()
        print('done train')
        print(results)

    def process_dataset(self, model, tokenizer, path, dataset):
        save_path = '%s/processed_dataset' % path

        if os.path.isdir(save_path):
            print('loading dataset from ', save_path)
            return load_from_disk(save_path)

        print('processing dataset from model', path)
        mapper = DataSetPostMapper(model, tokenizer)
        mapped_ds = dataset.map(lambda examples: mapper.add_prob_to_be_correct(examples, path),
                                batched=True,
                                batch_size=20,
                                writer_batch_size=20)
        print('saving to:', save_path)
        mapped_ds.save_to_disk(save_path)
        return mapped_ds


if __name__ == '__main__':
    print('starting test')
    Test().train_boolq_test()
