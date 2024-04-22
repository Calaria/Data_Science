from typing import Callable


def partial_derivative(
    func: Callable[[list[float]], float], V: list[float], target: int, h: float
) -> float:
    new = [value + (h if i == target else 0) for i, value in enumerate(V)]
    return round((func(new) - func(V)) / h, 5)


def f(V: list[float]) -> float:
    return sum(x**2 for x in V)


# 计算梯度
def get_gradient(
    func: Callable[[list[float]], float], V: list[float], step: float
) -> list[float]:
    new = [partial_derivative(func, V, i, step) for i in range(len(V))]
    return new


# 更新我们的位置
def update_location(
    directional_vector: list[float], current_location: list[float], step: float
) -> None:
    for i in range(len(directional_vector)):
        move = -directional_vector[i] * step
        current_location[i] += move
        current_location[i] = round(current_location[i], 5)


V = [-100.5, 100.5,1000,1000,212230,12]
X = [V[0]]
Y = [V[1]]
Z = [f(V)]
for i in range(1000):
    directional_vector = get_gradient(f, V, 0.01)
    update_location(directional_vector, V, 0.01)
print(V)
