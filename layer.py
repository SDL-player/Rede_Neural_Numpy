from functions import functions_, np


random = np.random


class Layer:
    def __init__(self, input_, output, function, i=0.1):
        self.input = input_
        self.output = output
        self.name_of_function = function
        self.weights = random.rand(output, input_) * i
        self.biases = np.array([random.rand()] * output)
        self.temp_input = []
        self.f = functions_[function]
        
    def __call__(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        self.temp_input = x
        x = np.dot(self.weights, x) + self.biases
        f = np.vectorize(self.f)
        x = f(x)

        return x
