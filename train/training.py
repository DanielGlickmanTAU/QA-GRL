from datasets import load_metric
from transformers import TrainingArguments, Trainer

from data.TaskParams import TaskParams


def get_trainer(save_dir, model_params, model_and_dataset: TaskParams, load_best_model_at_end, experiment=None,
                metric_name='accuracy', disable_tqdm=False, data_collator=None):
    metric = load_metric(metric_name)

    def compute_metrics(eval_pred):
        print('computing metric')
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=1)
        accuracy = metric.compute(predictions=predictions, references=labels)
        if experiment:
            experiment.log_metrics({'accuracy_on_eval': accuracy['accuracy']})
        return accuracy

    args = TrainingArguments(

        save_dir,
        evaluation_strategy="epoch",
        learning_rate=model_params.learning_rate,
        per_device_train_batch_size=(model_params.batch_size),
        per_device_eval_batch_size=(model_params.batch_size),
        num_train_epochs=model_params.num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_name,
        save_total_limit=2,
        disable_tqdm=disable_tqdm
    )
    trainer = Trainer(
        model_and_dataset.model,
        args,
        train_dataset=model_and_dataset.dataset["train"],
        eval_dataset=model_and_dataset.dataset["validation"],
        tokenizer=model_and_dataset.tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    return trainer
