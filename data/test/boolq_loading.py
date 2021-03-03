from unittest import TestCase

from config import ExperimentVariables
from data.boolq_utils import get_t_q_a
from data.datasets_loading import get_boolq_dataset
from models.model_loading import get_model_and_tokenizer_for_classification


class Test(TestCase):
    def test_preprocess_function_boolq(self):
        ExperimentVariables.hyperparams.task_name = "boolq"

        ExperimentVariables.hyperparams.model_params = ExperimentVariables._distilbert_squad

        model, tokenizer = get_model_and_tokenizer_for_classification()
        dataset_race = get_boolq_dataset(tokenizer)

        r1 = dataset_race['train'][0]

        t1, q1, a1 = get_t_q_a(r1)
        self.assertNotIsInstance(r1['label'], bool)

        ExperimentVariables.hyperparams.model_params = ExperimentVariables._roberta_squad
        model, tokenizer = get_model_and_tokenizer_for_classification()
        dataset_race = get_boolq_dataset(tokenizer)

        r1 = dataset_race['train'][0]

        t1, q1, a1 = get_t_q_a(r1)
        print(t1,q1)
