import numpy


# smart minus stupid
def smart_is_right_and_stupid_is_wrong(smart_scores, stupid_scores):
    normalizer = ratio(smart_scores, stupid_scores)
    return (normalizer * sum(smart_scores)) - sum(stupid_scores)


def smart_is_right_and_stupid_are_not_sure(smart_scores, stupid_scores):
    return sum(smart_scores) - sum([numpy.abs(0.5 - x) for x in stupid_scores])


def absolute_error(smart_scores, stupid_scores):
    return sum(smart_scores) + sum(stupid_scores)


def var_combined(smart_scores, stupid_scores):
    return numpy.var(smart_scores + stupid_scores)


def var_stupid_minus_smart(smart_scores, stupid_scores):
    return numpy.var(stupid_scores) - numpy.var(smart_scores)


def ratio(smart_scores, stupid_scores):
    assert len(stupid_scores) >= len(smart_scores)
    return len(stupid_scores) / len(smart_scores)
