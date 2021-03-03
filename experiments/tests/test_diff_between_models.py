from experiments import experiment
from collections import defaultdict

from datasets import load_from_disk

from models.confidence import get_last_confidence_model
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

        answer_model, answer_tokenizer = get_last_model_and_tokenizer(task, model_params)
        mapped_qa_ds = self.get_processed_dataset(task, model_params, answer_model, answer_tokenizer)

        error_prediction_model_params = ExperimentVariables._roberta_squad
        confidence_model, confidence_tokenizer = get_last_confidence_model(error_prediction_model_params)
        error_prediction_task_name = 'error-prediction'

        mapped_error_ds = self.get_processed_error_dataset(confidence_model, confidence_tokenizer,
                                                           error_prediction_model_params,
                                                           error_prediction_task_name, mapped_qa_ds)

        print('train acc:', sum(mapped_error_ds['train']['correct']) / len(mapped_error_ds['train']))
        print('validation acc:', sum(mapped_error_ds['validation']['correct']) / len(mapped_error_ds['validation']))

        self.print_by_probability_ratio(mapped_error_ds['validation'], confidence_tokenizer)
        self.print_by_probability_ratio(mapped_error_ds['validation'], confidence_tokenizer)

    def test_simple_confidence_model_example(self):
        error_prediction_model_params = ExperimentVariables._roberta_squad
        confidence_model, confidence_tokenizer = get_last_confidence_model(error_prediction_model_params)
        error_prediction_task_name = 'error-prediction'

        dataset = datasets_loading.get_boolq_dataset(confidence_tokenizer)
        assert len(dataset['validation']) < 3000

        mapper = DataSetPostMapper(confidence_model, confidence_tokenizer)

        mapped_error_ds = dataset['validation'].map(mapper.add_is_correct_and_probs, batched=True, batch_size=50,
                                                    writer_batch_size=50)

        # JUST NEED TO FIGURE OUT WHAT DATA THE CONFIDENCE MODEL EXCPECTS...
        # PROBABLY JUST NORMAL BOOLQ DATA T + Q...PROBABLY so i need to get regular boolq dataset. and map it..
        # for each example, put another of example of the generated one and tag from where it is a
        # original_boolq.map(lambda examples: gen_model.generate(example['txt'][question]))

    def get_processed_error_dataset(self, confidence_model, confidence_tokenizer, error_prediction_model_params,
                                    error_prediction_task_name, mapped_qa_ds):
        error_save_path = '%s/processed_dataset' % get_save_path(error_prediction_task_name,
                                                                 error_prediction_model_params)
        if self.load_error_ds_from_disk:
            return load_from_disk(error_save_path)

        else:
            raise Exception()
        # return predict_confidence_on_boolq(confidence_model, confidence_tokenizer, mapped_qa_ds, error_save_path)

    def print_by_probability_ratio(self, mapped_ds, tokenizer, k=100):
        sorted_ds = mapped_ds.sort('prob')
        top = get_top_examples(k=k, ds=sorted_ds, tokenizer=tokenizer)
        buttom = get_top_examples(k=k, ds=sorted_ds, tokenizer=tokenizer, reverse=True)
        print('\n\n'.join(['question:' + x[0][1] + '\ntext:' + x[0][0] + '\nconfidence:' + str(x[1]) for x in top]))
        print('\n')
        print('\n\n'.join(['question:' + x[0][1] + '\ntext:' + x[0][0] + '\nconfidence:' + str(x[1]) for x in buttom]))

    def map_texts_to_questions(self, dataset_split, tokenizer):
        d = defaultdict(list)
        l = [boolq_utils.get_t_q_a(tokenizer, example) for example in
             [dataset_split[i] for i in range(len(dataset_split))]]

        for t, q, a in l:
            d[t].append(q)
        return d

    load_error_ds_from_disk = True
    load_processed_ds_from_disk = True

    def get_processed_dataset(self, task, model_params, model, tokenizer):

        save_path = '%s/processed_dataset' % get_save_path(task, model_params)

        if self.load_processed_ds_from_disk:
            return load_from_disk(save_path)

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
        return mapped_ds
