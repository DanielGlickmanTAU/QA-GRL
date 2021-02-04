from datasets import load_metric

from utils.datasets_loading import get_swag_dataset
from utils.model_loading import get_model_and_tokenizer_for_classification
from transformers import TrainingArguments, Trainer

model, tokenizer = get_model_and_tokenizer_for_classification()

encoded_dataset = get_swag_dataset()

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
    metric_for_best_model=metric_name,
)

metric = load_metric(metric_name)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return metric.compute(predictions=predictions, references=labels)


validation_key = "validation"

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"].select(range(1000)),
    eval_dataset=encoded_dataset[validation_key].select(range(100)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
