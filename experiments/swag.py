from dataclasses import dataclass

from experiments.TaskParams import TaskParams
from utils.datasets_loading import get_swag_dataset
from utils.model_loading import get_model_and_tokenizer_for_classification

model, tokenizer = get_model_and_tokenizer_for_classification()

encoded_dataset = get_swag_dataset(tokenizer)

benchmark_folder_name = "swag-classification"


@dataclass
class SwagClassificationParams(TaskParams):
    benchmark_folder_name: str = "swag-classification"


swagClassificationParams = SwagClassificationParams(encoded_dataset, model, tokenizer)

# trainer.train()
