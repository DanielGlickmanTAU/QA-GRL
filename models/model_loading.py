import os
from os import listdir

import utils.compute as compute
import data.special_tokens as special_tokens
from config.ExperimentVariables import hyperparams

dl_glickman_cache = compute.get_cache_dir()

if compute.is_university_server():
    try:
        # change transofrmers cache dir, cause defalut store in university is not enough
        os.environ['TRANSFORMERS_CACHE'] = dl_glickman_cache
    except:
        print('failed changing transformers cache dir')
from transformers import AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering, AutoModelForSequenceClassification, \
    AutoModelForSeq2SeqLM, T5Tokenizer

device = compute.get_device()
print('using device ', device)


def get_model_and_tokenizer_for_qa(model_name=hyperparams.model_params.model_name,
                                   toknizer_model_name=hyperparams.model_params.model_tokenizer):
    return _get_model_and_toknizer(model_name, toknizer_model_name, AutoModelForQuestionAnswering)


def get_model_and_tokenizer_for_classification(model_name=None, toknizer_model_name=None):
    if not model_name:
        model_name = hyperparams.model_params.model_name
    if not toknizer_model_name:
        toknizer_model_name = hyperparams.model_params.model_tokenizer
    return _get_model_and_toknizer(model_name, toknizer_model_name, AutoModelForSequenceClassification)


def get_model_and_tokenizer_for_generation(model_name, toknizer_model_name):
    if not model_name:
        model_name = hyperparams.model_params.model_name
    if not toknizer_model_name:
        toknizer_model_name = hyperparams.model_params.model_tokenizer
    return _get_model_and_toknizer(model_name, toknizer_model_name, AutoModelForSeq2SeqLM)


def _get_model_and_toknizer(model_name, toknizer_model_name, autoModelClass):
    def _get_and_save_pretrained_tokenizer(name):
        first_time_running_model = not os.path.exists("%s/" % name)
        if first_time_running_model:
            os.makedirs("%s/" % name)
            config = AutoConfig.from_pretrained(name)
            config.save_pretrained("%s/" % name)

        tokenizer = AutoTokenizer.from_pretrained("%s" % toknizer_model_name, cache_dir=dl_glickman_cache,
                                                  return_token_type_ids=True,
                                                  use_fast=True)
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens.special_tokens})
        if 't5' in toknizer_model_name:
            tokenizer.add_tokens(['<sep>', '<hl>'])
        if first_time_running_model:
            tokenizer.save_pretrained("%s/" % name)

        return tokenizer

    tokenizer = _get_and_save_pretrained_tokenizer(toknizer_model_name)
    model = autoModelClass.from_pretrained(model_name, cache_dir=dl_glickman_cache).to(device=device)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def get_last_checkpoint_in_path(path):
    files = listdir(path)
    return _get_last_checkpoint(files)


def get_best_checkpoint_in_path(path):
    files = listdir(path)
    return _get_best_checkpoint(files)


def _get_last_checkpoint(files):
    files = [f for f in files if 'checkpoint-' in f]
    return sorted(files, key=lambda s: int(s[len('checkpoint-'):]))[-1]


def _get_best_checkpoint(files):
    """ best checkpoint is the lowest checkpoint. Assuming keep_last = 1 and """
    files = [f for f in files if 'checkpoint-' in f]
    return sorted(files, key=lambda s: int(s[len('checkpoint-'):]))[0]


def get_best_model_and_tokenizer(saved_path, model_params):
    path = get_save_path(saved_path, model_params)
    checkpoint = get_best_checkpoint_in_path(path)
    path_checkpoint = path + '/' + checkpoint
    print('getting model from checkpoint ', path_checkpoint)
    if 'generation' in saved_path:
        return get_model_and_tokenizer_for_qa_generation2(path_checkpoint, model_params.model_tokenizer)
    return get_model_and_tokenizer_for_classification(path_checkpoint, model_params.model_tokenizer)


def get_save_path(saved_path, model_params):
    sep = '' if hyperparams.use_unique_seperator_for_answer else '/using_sep'
    # path = '../experiments/' + saved_path + '/' + model_params.model_name + sep
    path = saved_path + '/' + model_params.model_name + sep
    return path


def get_model_and_tokenizer_for_qa_generation(model_params):
    return get_model_and_tokenizer_for_qa_generation2(model_params.model_name, model_params.model_tokenizer)


def get_model_and_tokenizer_for_qa_generation2(model_name, model_tokenizer):
    tokenizer = T5Tokenizer.from_pretrained(model_tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        # cache_dir=,
    )
    tokenizer.add_tokens(['<sep>', '<hl>'])
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer
