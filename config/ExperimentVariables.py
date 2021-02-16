from dataclasses import dataclass, field

use_unique_seperator_for_answer = True
return_overflowing_tokens = False


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

@dataclass
class _hyperparam(AttrDict):
    hyperparams: AttrDict = field(default_factory=AttrDict)

    def __str__(self): return str(self.__dict__)

    def __repr__(self): return str(self.__dict__)

@dataclass
class _race(_hyperparam):
    negative_samples_per_question: int = 1

    def __str__(self): return str(self.__dict__)

    def __repr__(self): return str(self.__dict__)


race = _race()
