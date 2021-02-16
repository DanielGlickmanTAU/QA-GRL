from dataclasses import dataclass

from experiments.TaskParams import TaskParams
from utils.datasets_loading import get_race_dataset, get_sst_dataset, get_swag_dataset
from utils.model_loading import get_model_and_tokenizer_for_classification


@dataclass
class SwagClassificationParams(TaskParams):
    benchmark_folder_name: str = "swag-classification"


@dataclass
class RACEClassificationParams(TaskParams):
    benchmark_folder_name: str = "race-classification"


@dataclass
class SSTClassificationParams(TaskParams):
    benchmark_folder_name: str = "sst-classification"


model, tokenizer = get_model_and_tokenizer_for_classification()


def get_race_classification_params():
    encoded_dataset = get_race_dataset(tokenizer)
    return RACEClassificationParams(encoded_dataset, model, tokenizer)


def get_sst_params():
    encoded_dataset = get_sst_dataset(tokenizer)
    return SSTClassificationParams(encoded_dataset, model, tokenizer)


def get_swag_params():
    encoded_dataset = get_swag_dataset(tokenizer)
    return SwagClassificationParams(encoded_dataset, model, tokenizer)


task_to_params_getter = {
    'swag': get_swag_params,
    'race': get_race_classification_params,
    'sst': get_sst_params
}
