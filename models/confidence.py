from config.ExperimentVariables import hyperparams
from data.DatasetPostMapper import DataSetPostMapper
from data.TaskParams import TaskParams
from train.training import get_trainer
from utils import model_loading
from utils.model_loading import get_last_model_and_tokenizer, get_save_path

task_name = 'error-prediction'


def train_confidence_model(mapped_qa_ds, model_params, experiment=None):
    confidence_model, confidence_tokenizer = model_loading.get_model_and_tokenizer_for_classification(
        model_params.model_name, model_params.model_tokenizer)
    train_confidence_model(confidence_model, confidence_tokenizer, mapped_qa_ds, model_params, task_name,
                           experiment)

    return get_last_confidence_model(model_params)


def get_last_confidence_model(model_params):
    return get_last_model_and_tokenizer(task_name, model_params)


def train_confidence_model(confidence_model, confidence_tokenizer, mapped_qa_ds, model_params, task_name, experiment):
    error_ds = get_error_dataset(confidence_model, confidence_tokenizer, mapped_qa_ds)
    metric_name = "accuracy"
    task_params = TaskParams(error_ds, confidence_model, confidence_tokenizer, 'error-prediction')
    save_dir = get_save_path(task_name, model_params)
    trainer = get_trainer(save_dir, model_params, task_params, True, experiment, metric_name,
                          hyperparams.disable_tqdm)
    trainer.train()


def get_error_dataset(confidence_model, confidence_tokenizer, mapped_qa_ds):
    mapper = DataSetPostMapper(confidence_model, confidence_tokenizer)
    error_ds = mapped_qa_ds.map(mapper.change_labels)
    return error_ds


def predict_confidence_on_dataset(confidence_model, confidence_tokenizer, save_path, dataset):
    error_ds = get_error_dataset(confidence_model, confidence_tokenizer, dataset)
    mapper = DataSetPostMapper(confidence_model, confidence_tokenizer)
    mapped_error_ds = error_ds.map(mapper.add_is_correct_and_probs, batched=True, batch_size=50,
                                   writer_batch_size=50)
    mapped_error_ds.save_to_disk(save_path)
    return mapped_error_ds