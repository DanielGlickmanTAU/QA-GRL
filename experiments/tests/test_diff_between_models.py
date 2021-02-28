from collections import defaultdict

from datasets import load_from_disk

from utils import compute

torch = compute.get_torch()
from data.DatasetPostMapper import DataSetPostMapper
from config import ExperimentVariables
from config.ExperimentVariables import hyperparams
from data import datasets_loading
from data.TaskParams import TaskParams
from utils.model_loading import get_last_model_and_tokenizer, get_save_path
from data import boolq_utils

model_params = hyperparams.model_params
model_name = model_params.model_name
# torch, experiment = experiment.start_experiment(tags=[model_name, hyperparams.task_name],
#                                                 hyperparams=hyperparams)
from unittest import TestCase


class Test(TestCase):
    def test_diff_between_models(self):
        mapped_ds, tokenizer = self.get_processed_dataset()
        train_dict = self.map_texts_to_questions(mapped_ds['train'], tokenizer)
        validation_dict = self.map_texts_to_questions(mapped_ds['validation'], tokenizer)

        overlap_texts = [t for t in validation_dict if len(train_dict[t])]
        print(len(overlap_texts)/ len(validation_dict))

        # sorted_ds = mapped_ds.sort('prob')
        # worst = [boolq_utils.get_t_q_a(tokenizer, example) for example in
        #          [sorted_ds['validation'][i] for i in range(5)]]
        # best = [boolq_utils.get_t_q_a(tokenizer, example) for example in
        #         [sorted_ds['validation'][-i - 1] for i in range(5)]]
        #
        # print('best: ', best)
        # print('worst: ', worst)

    def map_texts_to_questions(self, dataset_split, tokenizer):
        d = defaultdict(list)
        l = [boolq_utils.get_t_q_a(tokenizer, example) for example in
             [dataset_split[i] for i in range(len(dataset_split))]]

        for t, q, a in l:
            d[t].append(q)
        return d

    load_processed_ds_from_disk = True

    def get_processed_dataset(self):
        task = 'boolq-classification'
        model_params = ExperimentVariables._roberta_squad
        model, tokenizer = get_last_model_and_tokenizer(task, model_params)
        save_path = '%s/processed_dataset' % get_save_path(task, model_params)

        if load_from_disk:
            return load_from_disk(save_path), tokenizer

        ds = datasets_loading.get_boolq_dataset(tokenizer)
        task_params = TaskParams(ds, model, tokenizer, 'trash')
        mapper = DataSetPostMapper(task_params)
        mapped_ds = ds.map(mapper.add_is_correct_and_probs, batched=True, batch_size=20, writer_batch_size=20)
        mapped_ds.save_to_disk(save_path)
        return mapped_ds, tokenizer
