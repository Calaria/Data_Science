import pandas as pd
from typing import List, Tuple
from tools import dot
import random
import tqdm


ans=[60,0.15,-8,30]

# Define the path to your file. Adjust the path based on where you saved the file.
file_path = r"phone_usage.txt"

# Read the dataset from the file
data = pd.read_csv(file_path, sep="\t")

# Display the first few rows of the data to verify it's read correctly
print(data.head())
Vector = List[float]

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
    v=add(step,v)
    return v

def sqerror_gradient(x:Vector, y:float,beta:Vector):
    err=error(x,y,beta)
    return [2*err *x_i for x_i in x]

    
def my_error_function(X: List[Vector], y: List[float], beta: Vector) -> float:
    return sum(squared_error(x, y_i, beta) for x, y_i in zip(X, y))

def least_squares_fit(X:List[Vector],
                      Y:List[float],
                      learning_rate:float=0.0000015,
                      num_steps: int =3000,
                      batch_size: int =1)->Vector:
    guess=[random.random() for _ in X[0]]

    with tqdm.trange(num_steps) as t:
        for _ in t:
            for start in range(0,len(X), batch_size):
                batch_x=X[start:start+batch_size]
                batch_y=Y[start:start+batch_size]
                for x,y in zip(batch_x,batch_y):
                    grad=sqerror_gradient(x,y,guess)
                    guess=gradient_step(guess,grad,-learning_rate)
            current_error=my_error_function(X,Y,guess)
            t.set_description(f"Current Error: {current_error:.3f}")       
    guess=[round(g,5) for g in guess]
    return guess

print(f"Loading data from {file_path}")
#Prepare data
X = data[['Friends Count', 'Daily Work Hours', 'Has PhD']].values.tolist()
Y = data['Time Spent on Phone (min)'].values.tolist()
X=[[1]+x for x in X]
#Test my_error_function

#Train the model
#疑惑：为什么t设置错误？为什么出现了lan？为什么shape不对？
"""

#Test the error and squared error functions
x = [1, 2, 3]
y = 30
beta = [4, 4, 4]
assert error(x, y, beta) == -6
assert squared_error(x, y, beta) == 36

#Test sqerror_gradient
assert sqerror_gradient(x,y,beta)==[-12,-24,-36]

#Test the error_gadient functions
assert squared_gradient(x, y, beta) == [-12, -24, -36]
"""
#guess=[random.random() for _ in X[0]]
#print(my_error_function(X,Y,guess))
#guess1=least_squares_fit(X,Y,num_steps=3000,learning_rate=0.000001 )
#print(guess1)
#guess2=least_squares_fit(X,Y,num_steps=2000,batch_size=10)
#print(guess2)
#guess3=least_squares_fit(X,Y,num_steps=1000,batch_size=5,learning_rate=0.000001)
#print(guess3)

##Bootstrap our model

from typing import TypeVar,Callable,List
import random

Xs=TypeVar('Xs')
Stat = TypeVar('Stat')

def bootstrap_sample(data:List[Xs])->List[Xs]:
    return [random.choice(data) for _ in data]

def bootstrap_statistic(data:List[Xs],stats_fn:Callable[[List[Xs]],Stat],num_samples:int)->List[Stat]:
    return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]

from typing import Tuple
import datetime
from statistics import stdev,mean
"""
print(f"This is our estimate: {ans}")
guess1=least_squares_fit(X,Y,num_steps=20000,batch_size=10)
print(guess1)
"""
def estimate_sample_beta(pairs: List[Tuple[Vector, float]]):
    x_sample = [x for x,_ in pairs]
    y_sample =[y for _,y in pairs]
    beta=least_squares_fit(x_sample,y_sample,num_steps=3000,batch_size=10)
    return beta

print(f"Doing bootstrap on our model...")
random.seed(0)
#bootstrap_betas=bootstrap_statistic(list(zip(X,Y)),estimate_sample_beta,50)
#bootstrap_standard_errors=[stdev([beta[i] for beta in bootstrap_betas]) for i in range(4)]
#print(f"Bootstrap standard errors: {bootstrap_standard_errors}")
#print(bootstrap_standard_errors)




#Calculate the p-value
from scipy.stats import norm

def p_value(beta_hat_j:float,se_hat_j:float)->float:
    if beta_hat_j>0:
        ans=(1-norm.cdf(beta_hat_j/se_hat_j))
    else:
        ans=2*norm.cdf(beta_hat_j/se_hat_j)
    return f"{ans:.5f}"
"""
print("P-values:")   
print(p_value(guess1[0],bootstrap_standard_errors[0]))
print(p_value(guess1[1],bootstrap_standard_errors[1]))
print(p_value(guess1[2],bootstrap_standard_errors[2]))
print(p_value(guess1[3],bootstrap_standard_errors[3]))
"""


#Regularization

def ridge_penalty(beta:Vector,alpha:float)->float:
    return alpha*dot(beta[1:],beta[1:])
def squared_error_ridge(x:Vector,y:float,beta:Vector,alpha:float)->float:
    return error(x,y,beta)**2+ridge_penalty(beta,alpha)

def ridge_penalty_gradient(beta:Vector,
                           alpha:float)->Vector:
    return [0.]+[2*alpha*beta_j for beta_j in beta[1:]]
def sqerror_ridge_gradient(x:Vector,
                           y:float,
                           beta:Vector,
                           alpha:float)->Vector:
    return add(sqerror_gradient(x,y,beta),
               ridge_penalty_gradient(beta,alpha))

def least_squares_fit_ridge(X:List[Vector],
                            Y:List[float],
                            learning_rate:float=0.0000015,
                            num_steps:int=3000,
                            batch_size:int=1,
                            alpha:float=0.0)->Vector:
    guess=[random.random() for _ in X[0]]
    with tqdm.trange(num_steps) as t:
        for _ in t:
            for start in range(0,len(X),batch_size):
                batch_x=X[start:start+batch_size]
                batch_y=Y[start:start+batch_size]
                for x,y in zip(batch_x,batch_y):
                    grad=sqerror_ridge_gradient(x,y,guess,alpha)
                    guess=gradient_step(guess,grad,-learning_rate)
            current_error=my_error_function(X,Y,guess)
            t.set_description(f"Current Error: {current_error:.3f}")
    guess=[round(g,5) for g in guess]
    return guess

#Test the regularization
beta_2=least_squares_fit_ridge(X,Y,alpha=0.2,num_steps=10000,batch_size=20,learning_rate=0.0000015)
beta_3=least_squares_fit_ridge(X,Y,alpha=1,num_steps=10000,batch_size=20,learning_rate=0.0000015)
beta_1=least_squares_fit_ridge(X,Y,alpha=0,num_steps=10000,batch_size=20,learning_rate=0.0000015)
beta_0=least_squares_fit(X,Y,num_steps=10000,batch_size=10)


#R squared
def multiple_r_squared(X:List[Vector],Y:List[float],beta:Vector)->float:
    sum_of_squared_errors=sum(error(x,y,beta)**2 for x,y in zip(X,Y))
    return 1.0-sum_of_squared_errors/sum(y**2 for y in Y)
random.seed(0)
R0=multiple_r_squared(X,Y,beta_0)
R1=multiple_r_squared(X,Y,beta_1)
R2=multiple_r_squared(X,Y,beta_2)
R3=multiple_r_squared(X,Y,beta_3)
#Compare the R squared
print("Comparing the R squared...")
import  pandas as pd
#show beta and R squared
df=pd.DataFrame({'Beta':[beta_0,beta_1,beta_2,beta_3],
                 'R Squared':[R0,R1,R2,R3]})
df = df.round(5)
print(df)
