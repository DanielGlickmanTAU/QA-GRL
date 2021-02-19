from config.ExperimentVariables import hyperparams

OPT = '[OPT]'

special_tokens = [OPT]


def get_answer_seperator(tokenizer):
    return special_tokens.OPT if hyperparams.use_unique_seperator_for_answer else tokenizer.sep_token
