from unittest import TestCase

from utils import compute

torch = compute.get_torch()
from datasets import load_metric
from transformers import TrainingArguments, Trainer

import race


class Test(TestCase):
    def test_race_classification_params(self):
        params = race.classificationParams
        batch_size = 16
        metric_name = "accuracy"
        metric = load_metric(metric_name)

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions.argmax(axis=1)
            return metric.compute(predictions=predictions, references=labels)

        args = TrainingArguments(

            params.benchmark_folder_name,
            evaluation_strategy="epoch",
            learning_rate=8e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=100,
            weight_decay=0.01,
            # load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            # overwrite_output_dir=True
            save_total_limit=2
        )

        trainer = Trainer(
            params.model,
            args,
            train_dataset=params.dataset["train"],
            eval_dataset=params.dataset["validation"],
            tokenizer=params.tokenizer,
            compute_metrics=compute_metrics
        )

        # resume_from_checkpoint=params.benchmark_folder_name
        results = trainer.train(params.benchmark_folder_name + '/checkpoint-105500')
        print('done train')
        print(results)
