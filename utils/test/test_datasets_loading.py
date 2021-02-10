from unittest import TestCase

from utils.datasets_loading import get_race_dataset, get_swag_dataset
from utils.model_loading import get_model_and_tokenizer_for_classification


class Test(TestCase):
    def test_preprocess_function_race(self):
        model, tokenizer = get_model_and_tokenizer_for_classification()
        dataset_race = get_race_dataset(tokenizer)
        dataset_swag = get_swag_dataset(tokenizer)

        r1 = dataset_race['train'][0]
        s1 = dataset_swag['train'][0]

        print(r1,s1)
