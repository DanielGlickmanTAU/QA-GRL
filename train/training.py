from utils import compute

torch = compute.get_torch()

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

    args = get_training_args(disable_tqdm, load_best_model_at_end, metric_name, model_params, save_dir)

    class MyTrainer(Trainer):
        def _save_checkpoint(self, model, trial, metrics=None):
            print('here you go', metrics)
            super(MyTrainer, self)._save_checkpoint()._save_checkpoint(model, trial, metrics)

    trainer = MyTrainer(
        model_and_dataset.model,
        args,
        train_dataset=model_and_dataset.dataset["train"],
        eval_dataset=model_and_dataset.dataset["validation"],
        tokenizer=model_and_dataset.tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    return trainer


class GenTrainer(Trainer):

    def _training_step(self, model, inputs, optimizer) -> float:
        model.train()
        device = compute.get_device()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        # Our model outputs do not work with DataParallel, so forcing return tuple.
        if isinstance(model, torch.nn.DataParallel):
            inputs["return_tuple"] = True

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if len(compute.get_index_of_free_gpus()) > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        loss.backward()
        return loss.item()


def get_generator_trainer(save_dir, model_params, model_and_dataset: TaskParams, load_best_model_at_end,
                          disable_tqdm=False, data_collator=None):
    args = get_training_args(disable_tqdm, load_best_model_at_end, None, model_params, save_dir)

    return GenTrainer(
        model_and_dataset.model,
        args,
        train_dataset=model_and_dataset.dataset["train"],
        eval_dataset=model_and_dataset.dataset["validation"],
        tokenizer=model_and_dataset.tokenizer,
        data_collator=data_collator
    )


def get_training_args(disable_tqdm, load_best_model_at_end, metric_name, model_params, save_dir):
    return TrainingArguments(
        save_dir,
        evaluation_strategy="epoch",
        learning_rate=model_params.learning_rate,
        per_device_train_batch_size=model_params.batch_size,
        per_device_eval_batch_size=model_params.batch_size,
        num_train_epochs=model_params.num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=load_best_model_at_end,
        # metric_for_best_model=metric_name,
        save_total_limit=1 if load_best_model_at_end else 2,
        disable_tqdm=disable_tqdm,
        prediction_loss_only=True
    )
