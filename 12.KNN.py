from typing import List
from collections import Counter

def majority_vote(labels: List[str])->str:
    vote_counts = Counter(labels)
    winner,winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count ==winner_count])
    if num_winners==1:
        return winner
    else:
        return majority_vote(labels[:-1])
    
assert(majority_vote(['a','b','c','b','a']))=='b'

from typing import NamedTuple
Vector=List[float]
def distance(v:Vector,w:Vector)->float:
    assert len(v)==len(w)
    return sum((v_i-w_i)**2 for v_i,w_i in zip(v,w))**0.5

class LabeledPoint(NamedTuple):
    point:Vector
    label:str

def knn_classify(k:int,
                 labeled_points:List[LabeledPoint],
                 new_point:Vector)->str:
    by_distance = sorted(labeled_points,
                         key=lambda lp:distance(lp.point,new_point))
    k_nearest_labels = [lp.label for lp in by_distance[:k]]
    
    return majority_vote(k_nearest_labels)


import os
import requests

if not os.path.exists('iris.dat'):
    print('Downloading----')
    data = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
    with open('iris.dat', 'w') as f:
        f.write(data.text)
else:
    print('File already exists.')

from typing import Dict
import csv
from collections import defaultdict
def parse_iris_row(row:List[str])->LabeledPoint:
    mesurements = [float(value) for value in row[:-1]]
    label = row[-1].split('-')[-1]
    return LabeledPoint(mesurements,label)
with open('iris.dat')as f:
    reader = csv.reader(f)
    iris_data = [parse_iris_row(row) for row in reader if row]

#Group the points by species
points_by_species:Dict[str, List[Vector]] = defaultdict(list)
for iris in iris_data:
    points_by_species[iris.label].append(iris.point)
    
#Plot
import matplotlib.pyplot as plt
metrics = ['sepal length','sepal width','petal length','petal width']
pairs=[(i,j) for i in range(4) for j in range(4) if i<j]
marks = ['+','o','x']
fig ,ax = plt.subplots(2,3)
for row in range(2):
    for col in range(3):
        i,j = pairs[3*row+col]
        ax[row][col].set_title(f"{metrics[i]} vs {metrics[j]}",fontsize=8)
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])
        
        for mark,(species,points) in zip(marks,points_by_species.items()):
            xs = [point[i] for point in points]
            ys = [point[j] for point in points]
            ax[row][col].scatter(xs,ys,marker=mark,label=species)

ax[-1][-1].legend(loc='lower right',fontsize=6)
plt.show()

import random
from split_data import split_data
random.seed(42)
iris_train,iris_test = split_data(iris_data,0.7)
assert(len(iris_train)==0.7*150)
assert(len(iris_test)==0.3*150)

from typing import Tuple
confusion_matrix:Dict[Tuple[str, str], int]=defaultdict(int)
num_correct=0
for iris in iris_test:
    predicted = knn_classify(5,iris_train,iris.point)
    actual=iris.label
    if predicted == actual:
        num_correct+=1
    confusion_matrix[(predicted, actual)]+=1
pct_correct = num_correct/len(iris_test)
print(pct_correct,confusion_matrix)
