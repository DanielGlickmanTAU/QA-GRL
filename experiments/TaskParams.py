from dataclasses import dataclass

from tokenizers import Tokenizer
from tokenizers.models import Model
from torch.utils.data import Dataset


@dataclass
class TaskParams:
    dataset: Dataset
    model: Model
    tokenizer: Tokenizer
    benchmark_folder_name: str