from unittest import TestCase

from datasets import load_metric
from transformers import TrainingArguments, Trainer
import swag


class Test(TestCase):
    def test_swag_classification_params(self):
        params = swag.swagClassificationParams
        batch_size = 12
        metric_name = "accuracy"
        metric = load_metric(metric_name)

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions.argmax(axis=1)
            return metric.compute(predictions=predictions, references=labels)


        args = TrainingArguments(

            params.benchmark_folder_name,
            evaluation_strategy="epoch",
            learning_rate=2e-4,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=10,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
        )

        trainer = Trainer(
            params.model,
            args,
            train_dataset=params.dataset["train"].select(range(60)),
            eval_dataset=params.dataset["train"].select(range(10)),
            tokenizer=params.tokenizer,
            compute_metrics=compute_metrics
        )

        results = trainer.train()
        print(results)

