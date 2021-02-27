from dataclasses import dataclass

from datasets import DatasetDict

from data.TaskParams import TaskParams
from data.datasets_loading import get_race_dataset, get_sst_dataset, get_swag_dataset, get_boolq_dataset
from utils.model_loading import get_model_and_tokenizer_for_classification
import datasets


@dataclass
class SwagClassificationParams(TaskParams):
    benchmark_folder_name: str = "swag-classification"


@dataclass
class RACEClassificationParams(TaskParams):
    benchmark_folder_name: str = "race-classification"


@dataclass
class SSTClassificationParams(TaskParams):
    benchmark_folder_name: str = "sst-classification"

@dataclass
class BoolQClassificationParams(TaskParams):
    benchmark_folder_name: str = "boolq-classification"


@dataclass
class CombinedClassificationParams(TaskParams):
    benchmark_folder_name: str = "combined-classification"


def get_race_classification_params():
    model, tokenizer = get_model_and_tokenizer_for_classification()
    encoded_dataset = get_race_dataset(tokenizer)
    return RACEClassificationParams(encoded_dataset, model, tokenizer)


def get_sst_params():
    model, tokenizer = get_model_and_tokenizer_for_classification()
    encoded_dataset = get_sst_dataset(tokenizer)
    return SSTClassificationParams(encoded_dataset, model, tokenizer)


def get_swag_params():
    model, tokenizer = get_model_and_tokenizer_for_classification()
    encoded_dataset = get_swag_dataset(tokenizer)
    return SwagClassificationParams(encoded_dataset, model, tokenizer)

def get_boolq_params():
    model, tokenizer = get_model_and_tokenizer_for_classification()
    encoded_dataset = get_boolq_dataset(tokenizer)
    return BoolQClassificationParams(encoded_dataset, model, tokenizer)


def get_combined_params():
    model, tokenizer = get_model_and_tokenizer_for_classification()
    all_datasets = [get_swag_dataset(tokenizer), get_race_dataset(tokenizer), get_sst_dataset(tokenizer)]

    train = datasets.concatenate_datasets([ds['train'] for ds in all_datasets])
    validation = datasets.concatenate_datasets([ds['validation'] for ds in all_datasets])
    test = datasets.concatenate_datasets([ds['test'] for ds in all_datasets])

    encoded_dataset = DatasetDict({'train':train, 'validation':validation, 'test': test})
    return CombinedClassificationParams(encoded_dataset, model, tokenizer)


task_to_params_getter = {
    'swag': get_swag_params,
    'race': get_race_classification_params,
    'sst': get_sst_params,
    'combined': get_combined_params
}
