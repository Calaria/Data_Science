import random
from typing import TypeVar, List, Tuple
X=TypeVar('X')
def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    data = data[:]
    random.shuffle(data)
    cut = int(len(data) * prob)
    return data[:cut], data[cut:]
data = [n for n in range(1000)]
train, test = split_data(data, 0.75)
assert len(train)==750
assert len(test)==250
assert sorted(train+test)==data
Y =TypeVar('Y')

def train_test_split(xs: List[X], ys: List[Y], test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)
    return ([xs[i] for i in train_idxs], [xs[i] for i in test_idxs], [ys[i] for i in train_idxs], [ys[i] for i in test_idxs])

xs= [x for x in range(1000)]
ys= [2*x for x in xs]
x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.25)
assert all(y==2*x for x, y in zip(x_train, y_train))
assert all(y==2*x for x, y in zip(x_test, y_test))

