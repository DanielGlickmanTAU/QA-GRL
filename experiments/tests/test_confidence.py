from experiments import experiment
from collections import defaultdict

from datasets import load_from_disk

from models.confidence import get_last_confidence_model
from models.question_generation import generate_boolq_dataset
from utils import compute
from models import model_loading

torch = compute.get_torch()
from data.DatasetPostMapper import DataSetPostMapper
from config import ExperimentVariables
from config.ExperimentVariables import hyperparams
from data import datasets_loading
from data.TaskParams import TaskParams
from models.model_loading import get_best_model_and_tokenizer, get_save_path
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
    def test_confidence_ranking(self):
        task = 'boolq-classification'
        model_params = ExperimentVariables._roberta_squad

        answer_model, answer_tokenizer = get_best_model_and_tokenizer(task, model_params)
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

    def test_generate_and_rank_confidence(self):
        error_prediction_model_params = ExperimentVariables._roberta_squad
        confidence_model, confidence_tokenizer = get_last_confidence_model(error_prediction_model_params)
        mapper = DataSetPostMapper(confidence_model, confidence_tokenizer)
        error_prediction_task_name = 'error-prediction'

        boolq = datasets_loading.get_boolq_dataset(confidence_tokenizer)
        assert len(boolq['validation']) < 3000

        generation_task_name = 'question-generation'
        generation_model_params = ExperimentVariables._t5_qg

        gen_model, gen_tokenizer = model_loading.get_best_model_and_tokenizer(generation_task_name,
                                                                              generation_model_params)
        generated_questions = generate_boolq_dataset(gen_model, gen_tokenizer, num_questions=2)

        generated_questions = generated_questions.map(
            lambda examples: datasets_loading.tokenize_boolq(examples, confidence_tokenizer), batched=True)

        # generated_questions = boolq['validation'].select(range(4)).map(mapper.add_is_correct_and_probs, batched=True)
        # , batch_size=50,writer_batch_size=50)
        generated_questions = generated_questions.map(mapper.add_probs, batched=True, batch_size=50,
                                                      writer_batch_size=50)

        l = [(boolq_utils.get_t_q_a(confidence_tokenizer, example), example['prob']) for example in
             [generated_questions[i] for i in range(len(generated_questions))]]

        d = defaultdict(list)
        probs = {}
        for (t, q, a), prob in l:
            d[t].append(q)
            probs[q] = prob

        results = []
        for t in d:
            questions = d[t]
            if len(questions) < 2:
                continue
            for i in range(len(questions)):
                for j in range(1, len(questions)):
                    q1 = questions[i]
                    p1 = probs[q1]
                    q2 = questions[j]
                    p2 = probs[q2]
                    if not q1 == q2:
                        if p1 > p2:
                            results.append((t, q1, p1, q2, p2, p1 - p2))
                        else:
                            results.append((t, q2, p2, q1, p1, p2 - p1))

        results.sort(key=lambda x: -x[-1])

        with open('results_topk', 'w+') as f:
            f.write('\n\n'.join(self.format_results(results)))

        print('top:', '\n\n'.join(self.format_results(results[:100])))
        print('bot:', '\n\n'.join(self.format_results(results[-100:])))
        import pickle
        import time
        filename = 'results_pickle_' + str(time.time())
        filename_last = 'results_pickle_last'
        pickle.dump([hyperparams, results], open(filename, "wb"))
        pickle.dump([hyperparams, results], open(filename_last, "wb"))

    def format_results(self, results):
        return ['text:' + x[0] + '\n\n'
                + 'question1:' + x[1]
                + '\n' + 'prob1:' + str(x[2])
                + '\n' + 'question2:' + x[3]
                + '\n' + 'prob2:' + str(x[4])
                + '\n' + 'diff:' + str(x[5])
                + '\n' + '-' * 60
                for x in results]

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
