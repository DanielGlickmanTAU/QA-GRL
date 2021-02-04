from dataclasses import dataclass

from experiments.TaskParams import TaskParams
from utils.datasets_loading import get_race_dataset
from utils.model_loading import get_model_and_tokenizer_for_classification

model, tokenizer = get_model_and_tokenizer_for_classification()

encoded_dataset = get_race_dataset(tokenizer)



@dataclass
class SwagClassificationParams(TaskParams):
    benchmark_folder_name: str = "race-classification"


classificationParams = SwagClassificationParams(encoded_dataset, model, tokenizer)

# trainer.train()
