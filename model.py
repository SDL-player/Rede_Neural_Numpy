import numpy as np
from layer import Layer


class Model:
    def __init__(self):
        self.layers = []
        self.all_weights = []
        self.outputs = []
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def feedward(self, input_):
        x = input_
        for layer in self.layers:
            self.all_weights.append(layer.weights)
            x = layer(x)
        
        return x
    
    def backpropagation(self, target, output, alpha):
        
        size = len(self.layers)
        
        error = self.MTS(target, output)[0]
        
        dEg, dEu = self.delta_mts(target, output)
        
        for _id in range(size, 0, -1):
            id_ = _id - 1
            
            if _id == size:
                               
                inp = self.layers[id_].temp_input
                
                i = inp.shape[0]
                j = dEu.shape[0]
                
                dEu_ = np.resize(dEu, [i, j]).transpose()
                inp_ = np.resize(inp, [j, i])
                
                gradient_w = np.multiply(dEu_, inp_)
                
                self.layers[id_].weights -= gradient_w * alpha
                   
            else:
                
                w = self.layers[id_].weights
                
                dEg_ = np.resize(dEg, w.shape)
                
                gradient_w = np.multiply(dEg_, w)
                
                self.layers[id_].weights -= gradient_w * alpha
            
            s_b = self.layers[id_].biases
                
            gradient_b = np.multiply(np.resize(dEg, s_b.shape), self.layers[id_].biases)
                
            self.layers[id_].biases -= gradient_b * alpha 
                
            return error
                
    
    def delta_mts(self, target, output):
        f = lambda target, output: -(target - output) * output * (1 - output)
        a = f(target, output)
        return np.sum(a), a
        
    def MTS(self, target, output):
        values = -(target - output)
        f = lambda x: 1 / 2 * x ** 2
        a = f(values)
        return np.sum(a), a
    
    def train_model(self, inputs, targets, alpha = 0.5, epochs=2000):
        for epoch in range(epochs):
            
            media = 0
            
            q = 0
            
            for inp, tar in zip(inputs, targets):
               
               output = self.feedward(inp)
               
               error = self.backpropagation(tar, output, alpha)
               
               q += 1
               
               media += error
               
               print(f"Train data: {q} Epoch: {epoch + 1}/epochs Error: {error} Media: {media / q}")
               