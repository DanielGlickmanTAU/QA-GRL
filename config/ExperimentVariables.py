from dataclasses import dataclass


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __str__(self): return str(self.__dict__)

    def __repr__(self): return str(self.__dict__)


@dataclass(repr=False)
class _model_params(AttrDict):
    model_name: str
    model_tokenizer: str


_distilbert_squad = _model_params('distilbert-base-uncased-distilled-squad', 'distilbert-base-uncased')
_roberta_squad = _model_params('roberta-base-squad2', 'roberta-base-squad2')


@dataclass
class _race(AttrDict):
    negative_samples_per_question: int = 1

    def __str__(self): return str(self.__dict__)

    def __repr__(self): return str(self.__dict__)


hyperparams = AttrDict()
use_unique_seperator_for_answer = True
return_overflowing_tokens = False
race = _race()

hyperparams.race = race
hyperparams.model_name = _distilbert_squad
