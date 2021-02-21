import collections
from comet_ml import Experiment
from utils import compute


def start_experiment(tags=None, hyperparams=None):
    if hyperparams is None:
        hyperparams = {}
    if tags is None:
        tags = []

    def flatten(d):
        items = []
        for k, v in d.items():
            if isinstance(v, collections.MutableMapping):
                if hasattr(v, '__dict__'):
                    items.extend(flatten(v.__dict__).items())
                else:
                    items.extend(flatten(v).items())
            else:
                items.append((k, v))
        return dict(items)

    experiment = Experiment('FvAd5fm5rJLIj6TtmfGHUJm4b', project_name='dl', workspace="danielglickmantau")
    torch = compute.get_torch()

    if len(tags):
        experiment.add_tags(tags)
    if len(hyperparams):
        experiment.log_parameters(flatten(hyperparams))

    return torch, experiment
