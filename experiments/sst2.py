from dataclasses import dataclass

from experiments.TaskParams import TaskParams
from utils.datasets_loading import  get_sst_dataset
from utils.model_loading import get_model_and_tokenizer_for_classification


model, tokenizer = get_model_and_tokenizer_for_classification()

encoded_dataset = get_sst_dataset(tokenizer)


@dataclass
class SSTClassificationParams(TaskParams):
    benchmark_folder_name: str = "sst-classification"


classificationParams = SSTClassificationParams(encoded_dataset, model, tokenizer)