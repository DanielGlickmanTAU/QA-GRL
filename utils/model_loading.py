import os
import utils.special_tokens as special_tokens
import utils.compute as compute
from config import ExperimentVariables
from config.ExperimentVariables import hyperparams

dl_glickman_cache = compute.get_cache_dir()

if compute.is_university_server():
    try:
        # change transofrmers cache dir, cause defalut store in university is not enough
        os.environ['TRANSFORMERS_CACHE'] = dl_glickman_cache
    except:
        print('failed changing transformers cache dir')
from transformers import AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering, AutoModelForSequenceClassification

device = compute.get_device()
print('using device ', device)


def get_model_and_tokenizer_for_qa(model_name=hyperparams.model_params.model_params,
                                   toknizer_model_name=hyperparams.model_params.model_tokenizer):
    return _get_model_and_toknizer(model_name, toknizer_model_name, AutoModelForQuestionAnswering)


def get_model_and_tokenizer_for_classification(model_name=hyperparams.model_params.model_params,
                                               toknizer_model_name=hyperparams.model_params.model_tokenizer):
    return _get_model_and_toknizer(model_name, toknizer_model_name, AutoModelForSequenceClassification)


def _get_model_and_toknizer(model_name, toknizer_model_name, autoModelClass):
    def _get_and_save_pretrained_tokenizer(name):
        first_time_running_model = not os.path.exists("%s/" % name)
        if first_time_running_model:
            os.makedirs("%s/" % name)
            config = AutoConfig.from_pretrained(model_name)
            config.save_pretrained("%s/" % name)

        # NOTE: token_type_ids, seperates the question segment from text segment(its 0 and 1s array)
        # when using distilbert, it does not return token_type_ids, but the encoder adds [SEP] token
        tokenizer = AutoTokenizer.from_pretrained("%s" % toknizer_model_name, cache_dir=dl_glickman_cache, return_token_type_ids=True,
                                                  use_fast=True)
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens.special_tokens})
        if first_time_running_model:
            tokenizer.save_pretrained("%s/" % name)

        return tokenizer

    tokenizer = _get_and_save_pretrained_tokenizer(toknizer_model_name)
    model = autoModelClass.from_pretrained(model_name, cache_dir=dl_glickman_cache).to(device=device)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer
