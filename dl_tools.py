from typing import List
import math
Vector=List[float]
# forward
def dot(v: Vector, w: Vector) -> float:
    assert len(v) == len(w)
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def step_function(x:float)->float:
    return 1 if x >= 0 else 0
def sigmoid(t:float)->float:
    return 1/(1+math.exp(-t))

def neuron_output(weights:Vector,inputs:Vector)->float:
    return sigmoid(dot(weights,inputs))

def perceptron_output(weights:Vector,inputs:Vector)->float:
    return step_function(dot(weights,inputs))

def feed_forward(neural_network: List[List[Vector]],
                 input_vector: Vector) -> List[Vector]:
    ouputs: List[Vector] = []
    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias)
                  for neuron in layer]
        ouputs.append(output)
        input_vector = output
    return ouputs


# optimizer
def sqerror_gradient(network:List[List[Vector]],
                     input_vector: Vector,
                     target_vector: Vector) -> List[List[Vector]]:
    assert len(network)==2,f"len of neetwork is {len(network)}"
    hidden_outputs, outputs = feed_forward(network, input_vector)
    output_deltas=[output*(1-output)*(output-target)
                   for output, target in zip(outputs, target_vector)]
    
    output_grads=[[output_deltas[i] *hidden_output
                   for hidden_output in hidden_outputs + [1]]
                  for i ,output_neuron in enumerate(network[-1])]
    
    hidden_deltas=[hidden_output*(1-hidden_output)*
                   dot(output_deltas,[n[i] for n in network[-1]])
                   for i, hidden_output in enumerate(hidden_outputs)]
    
    hidden_grads=[[hidden_deltas[i]*input for input in input_vector + [1]]
                    for i, hidden_neuron in enumerate(network[0])]
    return [hidden_grads, output_grads]

def get_error(network:List[List[Vector]],
                xs:List[Vector],
                ys:List[Vector])->float:
        error=0.0
        assert len(xs)==len(ys)
        for x,y in zip(xs,ys):
            predicted=feed_forward(network,x)[-1][0]
            error+= (predicted-y[0])**2
        return error
def gradient_step(vectors:List[Vector],gradients:List[Vector],step_size:float)->List[Vector]:
    assert len(vectors)==len(gradients)
    step=[step_size*gradient for gradient in gradients]
    return [vector+step for vector, step in zip(vectors,step)]

#Basics
def vector_subtract(v:Vector,w:Vector)->Vector:
    assert len(v)==len(w)
    return [v_i-w_i for v_i,w_i in zip(v,w)]

def squared_distance(v:Vector,w:Vector)->float:
    return sum_of_squares(vector_subtract(v,w))

def sum_of_squares(v:Vector)->float:
    return dot(v,v) 