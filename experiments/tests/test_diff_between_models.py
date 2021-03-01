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
model_name = model_params.model_name
torch, experiment = experiment.start_experiment(tags=[model_name, "error prediction"],
                                                hyperparams=hyperparams)
from unittest import TestCase


def get_top_examples(k, ds, tokenizer, reverse=False):
    return [(boolq_utils.get_t_q_a(tokenizer, example), example['prob']) for example in
            [ds[(-i - 1) if reverse else i] for i in range(k)]]


class Test(TestCase):
    def test_diff_between_models(self):
        task = 'boolq-classification'
        model_params = ExperimentVariables._roberta_squad

        mapped_ds, answer_model, answer_tokenizer, = self.get_processed_dataset(task, model_params)
        confidence_model, confidence_tokenizer = model_loading.get_model_and_tokenizer_for_classification(
            model_params.model_name, model_params.model_tokenizer)

        mapper = DataSetPostMapper(confidence_model, confidence_tokenizer)
        error_ds = mapped_ds.map(mapper.change_labels)

        metric_name = "accuracy"
        save_dir = get_save_path('error-prediction', model_params)

        task_params = TaskParams(error_ds, confidence_model, confidence_tokenizer, 'error-prediction')
        trainer = get_trainer(save_dir, model_params, task_params, True, experiment, metric_name,
                              hyperparams.disable_tqdm)

        # results = trainer.train(save_dir + '/checkpoint-84500')
        # results = trainer.train(save_dir + '/checkpoint-8474')
        results = trainer.train()
        print('done')

        # self.print_by_probability_ratio(mapped_ds, tokenizer)

    def print_by_probability_ratio(self, mapped_ds, tokenizer):
        sorted_ds = mapped_ds.sort('prob')
        top = get_top_examples(k=20, ds=sorted_ds['validation'], tokenizer=tokenizer)
        buttom = get_top_examples(k=20, ds=sorted_ds['validation'], tokenizer=tokenizer, reverse=True)
        print('\n\n'.join(['question:' + x[0][1] + '\ntext:' + x[0][0] + '\nconfidence:' + str(x[1]) for x in top]))
        print('\n\n'.join(['question:' + x[0][1] + '\ntext:' + x[0][0] + '\nconfidence:' + str(x[1]) for x in buttom]))

    def map_texts_to_questions(self, dataset_split, tokenizer):
        d = defaultdict(list)
        l = [boolq_utils.get_t_q_a(tokenizer, example) for example in
             [dataset_split[i] for i in range(len(dataset_split))]]

        for t, q, a in l:
            d[t].append(q)
        return d

    load_processed_ds_from_disk = True

    def get_processed_dataset(self, task, model_params):
        model, tokenizer = get_last_model_and_tokenizer(task, model_params)
        save_path = '%s/processed_dataset' % get_save_path(task, model_params)

        if self.load_processed_ds_from_disk:
            return load_from_disk(save_path), model, tokenizer

        ds = datasets_loading.get_boolq_dataset(tokenizer)
        task_params = TaskParams(ds, model, tokenizer, 'trash')
        mapper = DataSetPostMapper(model, tokenizer)
        mapped_ds = ds.map(mapper.add_is_correct_and_probs, batched=True, batch_size=20, writer_batch_size=20)

        train_dict = self.map_texts_to_questions(mapped_ds['train'], tokenizer)
        validation_dict = self.map_texts_to_questions(mapped_ds['validation'], tokenizer)

        def _filter(example):
            t, q, a = boolq_utils.get_t_q_a(tokenizer, example)
            # just a trick to not filter the training set, only valid
            return len(train_dict[t]) == 0 or q in train_dict[t]

        mapped_ds = mapped_ds.filter(_filter)

        mapped_ds.save_to_disk(save_path)
        return mapped_ds, model, tokenizer
