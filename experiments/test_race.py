from unittest import TestCase

from utils import compute
from utils.compute import get_index_of_free_gpus

torch = compute.get_torch()
from datasets import load_metric
from transformers import TrainingArguments, Trainer
import race
import pytorch_lightning as pl
from dataclasses import dataclass
import torch.functional as F
from experiments.TaskParams import TaskParams
from models.ClassificationModel import ClassificationModel
from utils.datasets_loading import get_race_dataset
from utils.model_loading import get_model_and_tokenizer_for_classification


class Test(TestCase):
    def test_race_classification_params(self):
        params = race.classificationParams
        batch_size = 1
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

        qaClassificationModel = ClassificationModel(model, tokenizer)

        # trainer = pl.Trainer(gpus=len(get_index_of_free_gpus()))
        trainer = pl.Trainer(gpus=len(get_index_of_free_gpus()),distributed_backend='dp')
        trainer.fit(qaClassificationModel, get_data_loader('train'))

        # args = TrainingArguments(
        #
        #     params.benchmark_folder_name,
        #     evaluation_strategy="epoch",
        #     learning_rate=2e-5,
        #     per_device_train_batch_size=batch_size,
        #     per_device_eval_batch_size=batch_size,
        #     num_train_epochs=30,
        #     weight_decay=0.01,
        #     load_best_model_at_end=True,
        #     metric_for_best_model=metric_name,
        # )
        #
        # trainer = Trainer(
        #     params.model,
        #     args,
        #     train_dataset=params.dataset["train"].select(range(10_00)),
        #     eval_dataset=params.dataset["train"].select(range(100)),
        #     tokenizer=params.tokenizer,
        #     compute_metrics=compute_metrics
        # )
        #
        # results = trainer.train()
        # print('done train')
        # print(results)
