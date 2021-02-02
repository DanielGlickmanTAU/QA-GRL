from transformers import DistilBertForQuestionAnswering

from utils.debug import answer_question
from utils.model_loading import get_model_and_tokenizer_for_qa,get_model_and_tokenizer_for_classification
import utils.visualization as visualization

model, tokenizer = get_model_and_tokenizer_for_classification()

# assert type(model) == DistilBertFor#QuestionAnswering

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

# import sklearn
@decorators.measure_time
def preprocess():
    to_remove = list(dataset['train'][0].keys())
    to_remove.remove('label')
    return dataset.map(lambda examples: datasets_loading.preprocess_function(examples, tokenizer), batched=True,
                       remove_columns=to_remove)


encoded_dataset = preprocess()
print(encoded_dataset)

from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

metric_name = "accuracy"
batch_size = 12

benchmark_folder_name = "swag-classification"
args = TrainingArguments(
    benchmark_folder_name,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=50,
    weight_decay=0.01,
    load_best_model_at_end=True,
    #todo fix one use accuracy metric(use sklearn)
    # metric_for_best_model=metric_name,
)

#todo fix accuracy
# metric = load_metric(metric_name)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return predictions == labels


validation_key = "validation"

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"].select(range(20)),
                                #todo fit on real validation data
    eval_dataset=encoded_dataset["train"].select(range(20)),
    # eval_dataset=encoded_dataset[validation_key].select(range(10)),
    tokenizer=tokenizer,
    #todo fix accyracy
    # compute_metrics=compute_metrics
)

trainer.train()