from transformers import DistilBertForQuestionAnswering

from utils.debug import answer_question
from utils.model_loading import get_model_and_tokenizer_for_qa
import utils.visualization as visualization

model, tokenizer = get_model_and_tokenizer_for_qa()

assert type(model) == DistilBertForQuestionAnswering

from datasets import load_dataset, load_metric

dataset = load_dataset("swag", "regular")

# %%

import utils.visualization as visualization
import utils.datasets_loading as datasets_loading

x = dataset['train']
# %%
# print([t['sent2'] for t in x[:3]])


a = x[:10]
b = datasets_loading.preprocess_function(a, tokenizer)
print('asd')

import utils.decorators as decorators


@decorators.measure_time
def preprocess():
    to_remove = list(dataset['train'][0].keys())
    to_remove.remove('label')
    return dataset.map(lambda examples: datasets_loading.preprocess_function(examples, tokenizer), batched=True, remove_columns=to_remove)


p = preprocess()
print(p)
