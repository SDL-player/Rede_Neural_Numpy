import numpy as np


def Identity(x):
    return x

def binary(x):
    if x > 0:
        return 1
    else:
        return 0
    
def Relu(x):
    if x > 0:
        return x
    else:
        return 0


sigmoid = lambda x: 1 / (1 + np.e ** -x)

def Exponential_linear_unit(x, alpha):
    if x > 0:
        return x
    else:
        return alpha * (np.e ** -x - 1)
    
def Leaky_Relu(x):
    if x > 0:
        return x
    else:
        return x * 0.01
    
def PRelu(x, alpha):
    if x > 0:
        return x
    else:
        return alpha * x
    
functions_ = {"Relu": Relu,
             "Elu": Exponential_linear_unit,
             "Identity": Identity,
             "Binary": binary,
             "Prelu": PRelu,
             "Leaky_Relu": Leaky_Relu,
             "Sigmoid": sigmoid
             }
    
    
