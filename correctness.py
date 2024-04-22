def accuracy(tp: int, fp: int, fn: int, tn: int):
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total


def precision(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fp)


def recall(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fn)


def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return 2 * p * r / (p + r)


acc = accuracy(70, 4930, 13930, 981070)
print(acc)


preci = precision(70, 4930, 13930, 981070)
print(preci)


rec = recall(70, 4930, 13930, 981070)
print(rec)


f1 = f1_score(70, 4930, 13930, 981070)
print(f1)
