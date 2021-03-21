def aggregate_scores(smart_scores, stupid_scores):
    normalizer = len(stupid_scores) / len(smart_scores)
    return (normalizer * sum(smart_scores)) - sum(stupid_scores)