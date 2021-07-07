from functions import functions_, np


random = np.random

class Input:
    def __init__(self, input_, output, i=0.1):
        self.input = input_
        self.output = output
        self.weights = random.rand(output, input_) * i
        self.biases = random.rand(output) * i
        
    def __call__(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        self.temp_input = x
        x = np.dot(self.weights, x) + self.biases
        
        return x

class Layer:
    def __init__(self, input_, output, function, i=0.1):
        self.input = input_
        self.output = output
        self.name_of_function = function
        self.weights = random.rand(output, input_) * i
        self.biases = random.rand(output) * i
        self.temp_input = []
        self.f = functions_[function]
        
    def __call__(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        self.temp_input = x
        
        f = np.vectorize(self.f)
        x = f(x)
        x = np.dot(self.weights, x) + self.biases

        return x
