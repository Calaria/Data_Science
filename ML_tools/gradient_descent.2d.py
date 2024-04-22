from typing import Callable
import matplotlib.pyplot as plt
import numpy as np


def partial_derivative(
    func: Callable[[list[float]], float], V: list[float], target: int, h: float
) -> float:
    new = [value + (h if i == target else 0) for i, value in enumerate(V)]
    return round((func(new) - func(V)) / h, 5)


def f(V: list[float]) -> float:
    return sum(x**2 for x in V)


# 计算梯度
def get_gradient(func: Callable[[list[float]], float], V: list[float], step: float)->list[float]:
    new = [partial_derivative(func, V, i, step) for i in range(len(V))]
    return new


# 更新我们的位置
def update_location(directional_vector:list[float], current_location:list[float], step:float)->None:
    for i in range(len(directional_vector)):
        move = -directional_vector[i] * step
        current_location[i] += move
        current_location[i] = round(current_location[i], 5)


V = [-100.5, 100.5]
X = [V[0]]
Y = [V[1]]
Z = [f(V)]
for i in range(1000):
    directional_vector = get_gradient(f, V, 0.01)
    update_location(directional_vector, V, 0.01)
    X.append(V[0])
    Y.append(V[1])
    Z.append(f(V))


# 绘制函数图形
x_range = [i / 10.0 for i in range(-1000, 1000)]
y_range = [i / 10.0 for i in range(-1000, 1000)]
x, y = np.meshgrid(x_range, y_range)
z = x**2 + y**2
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x, y, z, cmap="viridis", edgecolor="none", alpha=0.5)


# 绘制梯度下降路径
ax.plot(X, Y, Z, color="r", marker=".", markersize=5)
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
ax.set_title("Gradient Descent Path on the Surface")
plt.show()
