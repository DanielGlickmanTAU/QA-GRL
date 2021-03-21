import gc
import os
from collections import namedtuple
from unittest import TestCase

import torch
from datasets import load_from_disk

import config.ExperimentVariables as variables
from config.ExperimentVariables import hyperparams
from data import tasks, datasets_loading
from data.DatasetPostMapper import DataSetPostMapper
from experiments import experiment
from models.model_loading import get_save_path, get_best_model_and_tokenizer
from train.training import get_trainer

sanity_asserts = False


class Test(TestCase):
    def test_mark_prob_being_correct(self):
        distilbert_tasks = ['boolq-classification1', 'boolq-classification2', 'boolq-classification3',
                            'boolq-classification4', 'boolq-classification5']
        distilbert_model_params = variables._distilbert_squad.clone()
        distilbert_model_params.batch_size *= 3

        roberta_tasks = ['boolq-classification1', 'boolq-classification2', 'boolq-classification3']
        roberta_model_params = variables._roberta_squad.clone()
        roberta_model_params.batch_size *= 3

        # different runs because tokenizers for each set are different
        dataset_stupid = self.iterate_tasks(distilbert_model_params, distilbert_tasks)
        dataset_smart = self.iterate_tasks(roberta_model_params, roberta_tasks)
        assert len(dataset_smart['scores']) == len(dataset_stupid['scores'])

        ScoredQuestion = namedtuple('ScoredQuestion', ['text', 'question', 'smart_scores', 'stupid_scores'])
        scored = []
        for smart, stupid in [(dataset_smart[i], dataset_stupid[i]) for i in range(len(dataset_smart['scores']))]:
            if sanity_asserts:
                assert smart['passage'] == stupid['passage']
                assert smart['question'] == stupid['question']
            scored.append(ScoredQuestion(smart['passage'], smart['question'], smart['scores'], stupid['scores']))

        def aggregate_scores(smart_scores, stupid_scores):
            normalizer = len(stupid_scores) / len(smart_scores)
            return (normalizer * sum(smart_scores)) - sum(stupid_scores)

        self.print_nicely(scored, aggregate_scores)

    def print_nicely(self, scored, scoring_function, ):
        def _print(example):
            print('-' * 50)
            print('question:', example['question'])
            print('text:', example['text'])
            # print('score:', score)
            print('smart scores:', example.smart_scores)
            print('stupid scores:', example.stupid_scores)
            print('-' * 50)

            """
            :param scoring_function: gets list_smart_scores, list_stupid_scores and returns the rank
            :return:
            """

        scored.sort(
            key=lambda scored_question: scoring_function(scored_question.smart_scores,
                                                         scored_question.stupid_scores))
        print('TOP')
        for x in scored[:3]: _print(x)
        print('BOT')
        for x in scored[-3:]: _print(x)

    def iterate_tasks(self, model_params, tasks):
        final_model_path = get_save_path('scored_is', model_params)
        dataset = None
        paths = []
        if os.path.isdir(final_model_path):
            return load_from_disk(final_model_path)
        for task in tasks:
            path = get_save_path(task, model_params)
            answer_model, answer_tokenizer = get_best_model_and_tokenizer(task, model_params)
            # first time, load unprocessed dataset
            if not dataset:
                dataset = datasets_loading.get_boolq_dataset(answer_tokenizer, remove_duplicates=False, keep_text=True)
                dataset = dataset['validation']
            dataset = self.process_dataset(answer_model, answer_tokenizer, path, dataset)
            paths.append(path)
            print('avg being correct:', sum(dataset[path]) / len(dataset[path]))

        scored_ds = dataset.map(lambda example: {'scores': [example[key] for key in paths]})
        scored_ds.save_to_disk(final_model_path)
        return scored_ds

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