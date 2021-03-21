def smart_minus_stupid(smart_scores, stupid_scores):
    normalizer = len(stupid_scores) / len(smart_scores)
    return (normalizer * sum(smart_scores)) - sum(stupid_scores)

def absolute_error(smart_scores,stupid_scores):
    return sum(smart_scores) + sum(stupid_scores)