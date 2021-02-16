import collections
def start_experiment(tags=[], hyperparams = {}):
    def flatten(d):
        items = []
        for k, v in d.items():
            if isinstance(v, collections.MutableMapping):
                if hasattr(v,'__dict__'):
                    items.extend(flatten(v.__dict__).items())
                else:
                    items.extend(flatten(v).items())
            else:
                items.append((k, v))
        return dict(items)

    from comet_ml import Experiment
    experiment = Experiment('FvAd5fm5rJLIj6TtmfGHUJm4b', project_name='dl', workspace="danielglickmantau")

    from utils import compute
    torch = compute.get_torch()

    if len(tags):
        experiment.add_tags(tags)
    if len(hyperparams):
        experiment.log_parameters(flatten(hyperparams))

    return torch, experiment
