from unittest import TestCase

from config import ExperimentVariables
from data.datasets_loading import get_race_dataset, get_boolq_dataset
from data.special_tokens import get_answer_seperator
from utils.model_loading import get_model_and_tokenizer_for_classification


class Test(TestCase):
    def test_preprocess_function_boolq(self):
        ExperimentVariables.task_name = "boolq"
        ExperimentVariables.model_params = ExperimentVariables._distilbert_squad

        def get_t_q_a(tokenizer, example):
            def remove_sep_if_starting_with_sep(str):
                return str[len(sep):] if str.startswith(sep) else str

            sep = tokenizer.sep_token
            q_seperator = get_answer_seperator(tokenizer)

            str = tokenizer.decode(example['input_ids'])
            str = remove_sep_if_starting_with_sep(str)

            idx = str.index(sep) + len(sep)
            t = str[:idx]
            str = str[idx:]
            str = remove_sep_if_starting_with_sep(str)

            idx = str.index(q_seperator) + len(q_seperator)
            q = str[:idx]

            a = example['label']

            return t, q, a

        model, tokenizer = get_model_and_tokenizer_for_classification()
        dataset_race = get_boolq_dataset(tokenizer)

        r1 = dataset_race['train'][0]
        r2 = dataset_race['train'][1]
        r3 = dataset_race['train'][2]
        r4 = dataset_race['train'][3]

        t1, q1, a1 = get_t_q_a(tokenizer, r1)
        t2, q2, a2 = get_t_q_a(tokenizer, r2)
        t3, q3, a3 = get_t_q_a(tokenizer, r3)
