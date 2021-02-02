from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML
from torch.utils.data import Dataset


def show_random_elements(dataset: Dataset, num_examples=10,picks= []):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset. Make sure its specific set(i.e train/dev)"
    if len(picks) == 0:
        for _ in range(num_examples):
            pick = random.randint(0, len(dataset) - 1)
            while pick in picks:
                pick = random.randint(0, len(dataset) - 1)
            picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))