from typing import List, Tuple


Vector = List[float]


def standard_deviation(xs: Vector) -> float:
    return variance(xs) ** 0.5


def variance(xs: Vector) -> float:
    assert len(xs) >= 2, "variance requires at least two elements"
    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)


def de_mean(xs: Vector) -> Vector:
    x_bar = mean(xs)
    return [x - x_bar for x in xs]


def mean(xs: Vector) -> float:
    return sum(xs) / len(xs)


def sum_of_squares(v: Vector) -> float:
    return dot(v, v)


def dot(v: Vector, w: Vector) -> float:
    assert len(v) == len(w), "vectors must be same length"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def covariance(xs: Vector, ys: Vector) -> float:
    assert len(xs) == len(ys), "xs and ys must have same number of elements"
    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)


def correlation(xs: Vector, ys: Vector) -> float:
    assert len(xs) == len(ys), "xs and ys must have same number of elements"
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / stdev_x / stdev_y
    else:
        return 0  # if no variation, correlation is zero


# Using multilinear regression to predict premium paid
import matplotlib.pyplot as plt
import math
import tqdm
import random


def calculate_mean(lst: List[float]) -> float:
    return sum(lst) / len(lst)


def vector_mean(vectors: List[Vector]) -> Vector:
    n = len(vectors)
    return scalar_multiply(
        1 / n, [sum(v[i] for v in vectors) for i in range(len(vectors[0]))]
    )


def calculate_standard_deviation(lst: List[float]) -> float:
    mean = calculate_mean(lst)
    squared_diff = [(x - mean) ** 2 for x in lst]
    variance = sum(squared_diff) / len(lst)
    return round(math.sqrt(variance), 3)


def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
    dim = len(data[0])
    means = vector_mean(data)
    stdevs = [
        calculate_standard_deviation([vector[i] for vector in data]) for i in range(dim)
    ]
    return means, stdevs


def rescale(data: List[Vector]) -> List[Vector]:
    dim = len(data[0])
    means, stdevs = scale(data)
    rescaled = [v[:] for v in data]
    for v in rescaled:
        for i in range(dim):
            if stdevs[i] > 0:
                v[i] = (v[i] - means[i]) / stdevs[i]
                v[i] = round(v[i], 3)
    return rescaled


def dot(v: Vector, w: Vector) -> float:
    assert len(v) == len(w)
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def predict(x: Vector, beta: Vector) -> float:
    return dot(x, beta)


def error(x: Vector, y: float, beta: Vector) -> float:
    return predict(x, beta) - y


def squared_error(x: Vector, y: float, beta: Vector) -> float:
    return error(x, y, beta) ** 2


def squared_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]


def scalar_multiply(c: float, v: Vector) -> Vector:
    return [c * v_i for v_i in v]


def add(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w)
    return [v_i + w_i for v_i, w_i in zip(v, w)]


def gradient_step(v: Vector, direction: Vector, step_size: float) -> Vector:
    assert len(v) == len(direction)
    step = scalar_multiply(step_size, direction)
    v = add(step, v)
    return v


def sqerror_gradient(x: Vector, y: float, beta: Vector):
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]


def my_error_function(X: List[Vector], y: List[float], beta: Vector) -> float:
    return sum(squared_error(x, y_i, beta) for x, y_i in zip(X, y))


def least_squares_fit(
    X: List[Vector],
    Y: List[float],
    learning_rate: float = 0.0000015,
    num_steps: int = 3000,
    batch_size: int = 1,
) -> Vector:
    guess = [random.random() for _ in X[0]]

    with tqdm.trange(num_steps) as t:
        for _ in t:
            for start in range(0, len(X), batch_size):
                batch_x = X[start : start + batch_size]
                batch_y = Y[start : start + batch_size]
                for x, y in zip(batch_x, batch_y):
                    grad = sqerror_gradient(x, y, guess)
                    guess = gradient_step(guess, grad, -learning_rate)
            current_error = my_error_function(X, Y, guess)
            t.set_description(f"Current Error: {current_error:.3f}")
    guess = [round(g, 5) for g in guess]
    return guess


import random
from typing import TypeVar, List, Tuple

X = TypeVar("X")
Y =TypeVar('Y')

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    data = data[:]
    random.shuffle(data)
    cut = int(len(data) * prob)
    return data[:cut], data[cut:]


def train_test_split(
    xs: List[X], ys: List[Y], test_pct: float
) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)
    return (
        [xs[i] for i in train_idxs],
        [xs[i] for i in test_idxs],
        [ys[i] for i in train_idxs],
        [ys[i] for i in test_idxs],
    )

def get_accuracy(tp: int, fp: int, fn: int, tn: int):
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total


def get_precision(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fp)


def get_recall(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fn)


def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
    p = get_precision(tp, fp, fn, tn)
    r = get_recall(tp, fp, fn, tn)
    return 2 * p * r / (p + r)
