from tools import Vector, correlation, mean, standard_deviation, mean
from typing import Tuple


def predict(alpha: float, beta: float, x_i: float):
    return beta * x_i + alpha


def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    return y_i - predict(alpha, beta, x_i)


def sum_of_sqerrors(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))


def least_squares_fit(x: Vector, y: Vector) -> Tuple[float, float]:
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta


def get_gradient(guess: Vector, x: Vector, y: Vector) -> Vector:
    alpha, beta = guess
    grad_alpha = sum(-2 * error(alpha, beta, x_i, y_i) for x_i, y_i in zip(x, y))
    grad_beta = sum(-2 * error(alpha, beta, x_i, y_i) * x_i for x_i, y_i in zip(x, y))
    return grad_alpha, grad_beta


def scalar_multiply(c: float, v: Vector) -> Vector:
    return [c * v_i for v_i in v]


def add(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w)
    return [v_i + w_i for v_i, w_i in zip(v, w)]


def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)


# create a simple dataset
num_friends = list(range(101))
daily_minutes = [0.3*x + 6 for x in num_friends]
# using gradient descent to find the best alpha and beta
from typing import List
import random
import tqdm
import matplotlib.pyplot as plt
from tools import de_mean

num_epochs = 100000
random.seed(0)
guess = [random.random(), random.random()]
learning_rate = 0.000001
with tqdm.trange(num_epochs) as t:
    for _ in t:
        alpha, beta = guess
        grad = get_gradient(guess, num_friends, daily_minutes)
        guess = gradient_step(guess, grad, -learning_rate)
        loss = sum_of_sqerrors(alpha, beta, num_friends, daily_minutes)
        t.set_description(f"loss: {loss:.3f}")
print(*guess)
#Testing the model
#using least square solution
alpha, beta = least_squares_fit(num_friends, daily_minutes)
print(alpha, beta)