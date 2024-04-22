from typing import List
from KNN import distance
import random
Vector=List[float]
def random_point(dim:int)->Vector:
    return [random.random() for  _ in range((dim))]

def random_distance(dim: int, num_pairs:int)->List[float]:
    return [distance(random_point(dim),random_point(dim)) for _ in range(num_pairs)]

import tqdm
dimensions = range(1,101)

avg_distances = []
min_distances = []
random.seed(42)
for dim in tqdm.tqdm(dimensions, desc="Curse of Dimensionality"):
    distances = random_distance(dim, 10000)
    avg_distances.append(sum(distances)/10000)
    min_distances.append(min(distances))

import matplotlib.pyplot as plt
plt.plot(dimensions, avg_distances, label='avg distance')
plt.plot(dimensions, min_distances, label='min distance')
plt.xlabel('Number of dimensions')
plt.ylabel('Distance')
plt.legend()
plt.show()