import os
import special_tokens

dl_glickman_cache = '/specific/netapp5_3/ML_courses/students/DL2020/glickman1/cache'
try:
    # change transofrmers cache dir, cause defalut store in university is not enough
    os.environ['TRANSFORMERS_CACHE'] = dl_glickman_cache
except:
    print('failed changing transformers cache dir')
from transformers import AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
import utils.compute as compute

device = compute.get_device()
print('using device ', device)


def get_model_and_tokenizer_for_qa(model_name='distilbert-base-uncased-distilled-squad',
                                   toknizer_model_name="distilbert-base-uncased"):
    return _get_model_and_toknizer(model_name, toknizer_model_name, AutoModelForQuestionAnswering)


def get_model_and_tokenizer_for_classification(model_name='distilbert-base-uncased-distilled-squad',
                                               toknizer_model_name="distilbert-base-uncased"):
    return _get_model_and_toknizer(model_name, toknizer_model_name, AutoModelForSequenceClassification)


def _get_model_and_toknizer(model_name, toknizer_model_name, autoModelClass):
    def _get_and_save_pretrained_tokenizer(name):
        # NOTE: token_type_ids, seperates the question segment from text segment(its 0 and 1s array)
        # when using distilbert, it does not return token_type_ids, but the encoder adds [SEP] token
        tokenizer = AutoTokenizer.from_pretrained("%s" % name, cache_dir=dl_glickman_cache, return_token_type_ids=True,
                                                  use_fast=True)
        tokenizer.add_special_token(special_tokens.OPT)

        config = AutoConfig.from_pretrained(name)
        if not os.path.exists("%s/" % name):
            os.makedirs("%s/" % name)
            tokenizer.save_pretrained("%s/" % name)
            config.save_pretrained("%s/" % name)

        return tokenizer

    tokenizer = _get_and_save_pretrained_tokenizer(toknizer_model_name)
    model = autoModelClass.from_pretrained(model_name, cache_dir=dl_glickman_cache, ).to(device=device)
    return model, tokenizer
