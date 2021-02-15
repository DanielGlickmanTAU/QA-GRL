def start_experiment(tags=[], hyperparams = {}):
    from comet_ml import Experiment
    experiment = Experiment('FvAd5fm5rJLIj6TtmfGHUJm4b', project_name='dl', workspace="danielglickmantau")

    from utils import compute
    torch = compute.get_torch()

    if len(tags):
        experiment.add_tags(tags)
    if len(hyperparams):
        experiment.log_parameters(hyperparams)

    return torch, experiment
