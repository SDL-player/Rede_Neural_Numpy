import numpy as np


random = np.random


class Layer:
    def __init__(self, input_, output, function, i=0.1):
        self.weights = random.rand(output, input_) * i
        self.biases = random.rand(output) * i
        self.temp_input = []
        self.temp_output = []
        self.f = function
        
    def __call__(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        self.temp_input = x
        x = np.dot(self.weights, x)
        f = np.vectorize(self.f)
        x = f(x)
        self.temp_output = x
        
        return x
    
