def derivative(x):
    return 2*x
def estimate(x,h):
    return ((x+h)**2-x**2)/h
X=range(1,11)
actuals=[derivative(x) for x in range(1,11)]
pred=[estimate(x,0.001) for x in range(1,11)]

import matplotlib.pyplot as plt
plt.plot(X,actuals,"rx",label="actual derivative")
plt.plot(X,pred,"b+",label="estimate derivative")
plt.legend(loc=9)
plt.show()
