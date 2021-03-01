from dataclasses import dataclass, asdict, is_dataclass
import os


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __str__(self): return str(asdict(self)) if is_dataclass(self) else super().__str__()

    def __repr__(self): return str(asdict(self)) if is_dataclass(self) else super().__repr__()


@dataclass(repr=False)
class _model_params(AttrDict):
    model_name: str
    model_tokenizer: str
    batch_size: int
    learning_rate: float
    num_epochs: int = 4


_distilbert_squad = _model_params('distilbert-base-uncased-distilled-squad', 'distilbert-base-uncased', 32, 3e-5)
_roberta_squad = _model_params('deepset/roberta-base-squad2', 'roberta-base', 12, 1e-5)
_electra_squad = _model_params('ahotrod/electra_large_discriminator_squad2_512',
                               'ahotrod/electra_large_discriminator_squad2_512', 2, 1e-5)
_t5_qg = _model_params('valhalla/t5-small-e2e-qg', 'valhalla/t5-small-e2e-qg', 2, 1e-5)


@dataclass(repr=False)
class _race(AttrDict):
    negative_samples_per_question: int = 1


hyperparams = AttrDict()
race = _race()
hyperparams.env = 'UNI' if 'HOST' in os.environ and 'gamir' in os.environ[
    'HOST'] else 'LOCAL' if 'USERNAME' in os.environ else 'AWS'
hyperparams.task_name = 'boolq'
hyperparams.model_params = _roberta_squad

hyperparams.race = race
hyperparams.use_unique_seperator_for_answer = False
hyperparams.return_overflowing_tokens = False
hyperparams.disable_tqdm = True
print('using hyperparams:', hyperparams)
