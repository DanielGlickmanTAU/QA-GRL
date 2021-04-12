import math

import numpy


def everyone_answers_wth_ease(smart_scores, stupid_scores):
    return sum(smart_scores) + sum(stupid_scores)


def smart_high_variance(smart_scores, stupid_scores):
    return numpy.var(smart_scores)


def smart_high_coefficient_of_variance(smart_scores, stupid_scores):
    return _coef_of_var(smart_scores)


def stupid_high_coefficient_of_variance(smart_scores, stupid_scores):
    return _coef_of_var(stupid_scores)


def diff_high_coefficient_of_variance(smart_scores, stupid_scores):
    return _coef_of_var(smart_scores) - _coef_of_var(stupid_scores)


def abs_diff_high_coefficient_of_variance(smart_scores, stupid_scores):
    return numpy.abs(_coef_of_var(smart_scores) - _coef_of_var(stupid_scores))


def _coef_of_var(scores):
    return numpy.std(scores) / numpy.mean(scores)


def stupid_high_variance(smart_scores, stupid_scores):
    return numpy.var(stupid_scores)


# smart minus stupid
def smart_is_right_and_stupid_is_wrong(smart_scores, stupid_scores):
    normalizer = ratio(smart_scores, stupid_scores)
    return (normalizer * sum(smart_scores)) - sum(stupid_scores)


def smart_is_right_and_stupid_are_not_sure(smart_scores, stupid_scores):
    return _avg(smart_scores) + _sure(stupid_scores)


def smart_is_sure_and_stupid_is_not_sure(smart_scores, stupid_scores):
    return _sure(stupid_scores) - _sure(smart_scores)


def smart_is_sure_and_stupid_is_wrong(smart_scores, stupid_scores):
    return _sure(smart_scores) - _avg(stupid_scores)


def smart_is_right_and_stupid_are_not_sure_normalizing_ratio(smart_scores, stupid_scores):
    return sum(smart_scores) * ratio(stupid_scores, stupid_scores) - _not_sure(stupid_scores)


def no_one_is_sure(smart_scores, stupid_scores):
    return _not_sure(smart_scores + stupid_scores)


def absolute_error(smart_scores, stupid_scores):
    return sum(smart_scores) + sum(stupid_scores)


def var_combined(smart_scores, stupid_scores):
    return numpy.var(smart_scores + stupid_scores)


def var_stupid_minus_smart(smart_scores, stupid_scores):
    return numpy.var(stupid_scores) - numpy.var(smart_scores)


def ratio(smart_scores, stupid_scores):
    assert len(stupid_scores) >= len(smart_scores)
    return len(stupid_scores) / len(smart_scores)


def random(smart_scores,stupid_scores):
    return numpy.random.random()


def _not_sure(stupid_scores):
    return -sum([numpy.abs(0.5 - x) for x in stupid_scores])


def _sure(stupid_scores):
    return _avg([numpy.abs(0.5 - x) for x in stupid_scores])


def _entropy(scores):
    scores = [s / sum(scores) for s in scores]
    return sum([-1 * x * numpy.log2(x) for x in scores])


def _avg(scores):
    return sum(scores) / len(scores)