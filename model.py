import json
import layer
import numpy as np


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
    
    def save_model(self, dir_):
        
        json_dict = {}
        
        for id_, layer in enumerate(self.layers):
            if layer.__class__.__name__ == "Layer":
                json_dict[f"Layer_{id_ + 1}"] = {"q_inputs": layer.input,
                                                 "q_outputs": layer.output,
                                                 "weights": layer.weights.tolist(),
                                                 "biases": layer.biases.tolist(),
                                                 "function": f"{layer.name_of_function}"}
                
            elif layer.__class__.__name__ == "Input":
                json_dict[f"Input"] = {"q_inputs": layer.input,
                                       "q_outputs": layer.output,
                                        "weights": layer.weights.tolist(),
                                        "biases": layer.biases.tolist()}
                
            elif layer.__class__.__name__ == "Output":
                json_dict[f"Output"] = {"q_inputs": layer.input,
                                        "q_outputs": layer.output,
                                        "function": f"{layer.name_of_function}"} 
            
            
        text = json.dumps(json_dict, indent=4)
            
        with open(dir_, "w") as file:
            file.writelines(text)
            file.close()
            
                
    def load_model(self, dir_):
        with open(dir_) as file:
            text = file.read()
            dict_json = json.loads(text)
            file.close()
            
        for key in dict_json.keys():
            
            if key == "Input":
                weights = dict_json[key]["weights"]
                biases = dict_json[key]["biases"]
                inputs = dict_json[key]["q_inputs"]
                outputs = dict_json[key]["q_outputs"]

                layer_ = layer.Input(inputs, outputs)
            
                layer_.weights = np.array(weights)
            
                layer_.biases = np.array(biases)
            
                self.layers.append(layer_)
                
            elif key == "Output":
                
                inputs = dict_json[key]["q_inputs"]
                outputs = dict_json[key]["q_outputs"]
                function = dict_json[key]["function"]
            
                layer_ = layer.Output(inputs, function)
            
            
                self.layers.append(layer_)
                
            else:
                weights = dict_json[key]["weights"]
                biases = dict_json[key]["biases"]
                inputs = dict_json[key]["q_inputs"]
                outputs = dict_json[key]["q_outputs"]
                function = dict_json[key]["function"]
            
                layer_ = layer.Layer(inputs, outputs, function)
            
                layer_.weights = np.array(weights)
            
                layer_.biases = np.array(biases)
            
                self.layers.append(layer_)
      
