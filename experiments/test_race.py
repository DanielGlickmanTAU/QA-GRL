import experiment
from config.ExperimentVariables import hyperparams

torch, experiment = experiment.start_experiment(tags=[hyperparams.model_name.model_name, hyperparams.task_name],
                                                hyperparams=hyperparams)
from unittest import TestCase
from datasets import load_metric
from transformers import TrainingArguments, Trainer

import race


class Test(TestCase):
    def test_race_classification_params(self):
        # params = race.get_race_classification_params()
        params = race.task_to_params_getter[hyperparams.task_name]()
        batch_size = 20
        metric_name = "accuracy"
        metric = load_metric(metric_name)

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions.argmax(axis=1)
            accuracy = metric.compute(predictions=predictions, references=labels)
            experiment.log_metrics(accuracy)
            return accuracy

        args = TrainingArguments(

            params.benchmark_folder_name,
            evaluation_strategy="epoch",
            learning_rate=1e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=100,
            weight_decay=0.01,
            # load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            # overwrite_output_dir=True
            save_total_limit=1
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
        results = trainer.train(params.benchmark_folder_name + '/checkpoint-84500')
        # results = trainer.train()
        print('done train')
        print(results)


if __name__ == '__main__':
    print('starting test')
    Test().test_race_classification_params()
