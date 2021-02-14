print('0')
from unittest import TestCase

from utils import compute
from utils.compute import get_index_of_free_gpus

torch = compute.get_torch()
from datasets import load_metric
from transformers import TrainingArguments, Trainer

print('1')
import race

print('2')
# import pytorch_lightning as pl
from dataclasses import dataclass
import torch.functional as F
from experiments.TaskParams import TaskParams

print('3')
# from models.ClassificationModel import ClassificationModel

print('3.5')
from utils.datasets_loading import get_race_dataset

print('4')
from utils.model_loading import get_model_and_tokenizer_for_classification

print('yoyo')


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

        model, tokenizer = get_model_and_tokenizer_for_classification()
        dataset = get_race_dataset(tokenizer)

        def get_data_loader(split):
            return torch.utils.data.DataLoader(dataset[split], batch_size=2, num_workers=2)

        # qaClassificationModel = ClassificationModel(model, tokenizer)

        # # trainer = pl.Trainer(gpus=len(get_index_of_free_gpus()))
        # trainer = pl.Trainer(gpus=len(get_index_of_free_gpus()),distributed_backend='dp')
        # trainer.fit(qaClassificationModel, get_data_loader('train'))

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

        #resume_from_checkpoint=params.benchmark_folder_name
        results = trainer.train(params.benchmark_folder_name + '/checkpoint-105500')
        print('done train')
        print(results)
