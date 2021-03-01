file1 = './error_prediction_confidence'
file2 = './probability_ratio_as_confidence'


def split_by_prefix(lst: list, prefix: str) -> list:
    return [q[len(prefix):].replace('\n', '') for q in lst if q.startswith(prefix)]


def calc_match_percent(l1, l2, k):
    assert len(l1) == len(l2)
    match, not_match = get_matches(l1, l2, k)
    return len(match) / k


def get_matches(l1, l2, k):
    l1, l2 = l1[:k], l2[:k]
    return [x for x in l1 if x in l2], [x for x in l1 if x not in l2]


def split_in2(lst: list):
    n = len(lst)
    return lst[:int(n / 2)], lst[int(n / 2):]


lines1 = open(file1, encoding='utf8').readlines()
lines2 = open(file2, encoding='utf8').readlines()

q1 = split_by_prefix(lines1, 'question:')
hard1, easy1 = split_in2(q1)
conf1 = split_by_prefix(lines1, 'confidence:')

q2 = split_by_prefix(lines2, 'question:')
hard2, easy2 = split_in2(q2)
conf2 = split_by_prefix(lines2, 'confidence:')
