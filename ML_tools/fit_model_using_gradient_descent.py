def sum_vector(V:list[list[float]])->list[float]:
    ndim=len(V[0])
    ans=[0]*ndim
    for i in range(ndim):
        for v in V:
            ans[i]+=v[i]
    return ans
def get_mean(V:list[list[float]]):
    n=len(V)
    V=sum_vector(V)
    k=len(V)
    for i in range(k):
        V[i]/=n
    return V
l=[(x,20*x+5) for x in range(-50,50)]
theta=(1,2)
def linear_gradient(x:float,y:float,theta:list[float])->list[float]:
    slop,intercept=theta
    predicted = slop*x+intercept
    error=predicted-y
    grad=[2*error*x,2*error]
    return grad

def update_location(
    directional_vector: list[float], current_location: list[float], step: float
) -> None:
    for i in range(len(directional_vector)):
        move = -directional_vector[i] * step
        current_location[i] += move
        current_location[i] = round(current_location[i], 5)
import random
random.seed(42)
theta=[random.uniform(-1,1),random.uniform(-1,1)]  
learning_rate=0.001
for epoch in range(3000):
    grad=get_mean([linear_gradient(x,y,theta) for x,y in l])
    update_location(directional_vector=grad,current_location=theta,step=learning_rate)
    if epoch%100==0:
        print(f"{epoch}: {theta}")
print(theta)

        
        