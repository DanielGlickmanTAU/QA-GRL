from unittest import TestCase

from config import ExperimentVariables
from data.datasets_loading import get_boolq_dataset
from data.special_tokens import get_answer_seperator
from utils.model_loading import get_model_and_tokenizer_for_classification


class Test(TestCase):
    def test_preprocess_function_boolq(self):
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

            return remove_speical_tokens(tokenizer, t), remove_speical_tokens(tokenizer, q), a

        def remove_speical_tokens(tokenizer, example: str):
            for special in list(tokenizer.special_tokens_map.values()):
                example = example.replace(special, '')
            return example

        ExperimentVariables.hyperparams.task_name = "boolq"

        ExperimentVariables.hyperparams.model_params = ExperimentVariables._distilbert_squad

        model, tokenizer = get_model_and_tokenizer_for_classification()
        dataset_race = get_boolq_dataset(tokenizer)

        r1 = dataset_race['train'][0]

        t1, q1, a1 = get_t_q_a(tokenizer, r1)
        self.assertNotIsInstance(r1['label'], bool)

        ExperimentVariables.hyperparams.model_params = ExperimentVariables._roberta_squad
        model, tokenizer = get_model_and_tokenizer_for_classification()
        dataset_race = get_boolq_dataset(tokenizer)

        r1 = dataset_race['train'][0]

        t1, q1, a1 = get_t_q_a(tokenizer, r1)
        print(t1,q1)
