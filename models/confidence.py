from config.ExperimentVariables import hyperparams
from data.DatasetPostMapper import DataSetPostMapper
from data.TaskParams import TaskParams
from train.training import get_trainer
from utils import model_loading
from utils.model_loading import get_last_model_and_tokenizer, get_save_path


def get_confidence_model(mapped_qa_ds, model_params, train=False, experiment = None):
    task_name = 'error-prediction'
    if train:
        confidence_model, confidence_tokenizer = model_loading.get_model_and_tokenizer_for_classification(
            model_params.model_name, model_params.model_tokenizer)
        train_confidence_model(confidence_model, confidence_tokenizer, mapped_qa_ds, model_params, task_name, experiment)

    return get_last_model_and_tokenizer(task_name, model_params)


def train_confidence_model(confidence_model, confidence_tokenizer, mapped_qa_ds, model_params, task_name,experiment):
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