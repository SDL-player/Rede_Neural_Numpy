# Rede_Neural_Numpy
Uma tentativa de fazer uma biblioteca de machine learning

# Como criar uma rede

```python
from Rede_Neural_Numpy import Layer # Importa a classe Layer.
from Rede_Neural_Numpy import Model # Importa a classe Model.
import numpy as np # Importa a biblioteca numpy como np

model = Model() # Cria um modelo.
model.add_layer(Layer(3, 5, "Relu")) # Cria uma camada com 3 entradas, 5 saídas, com a ativação Relu e adiciona a mesma ao modelo criado.
model.add_layer(Layer(5, 5, "Relu")) # Cria uma camada com 5 entradas, 5 saídas, com a ativação Relu e adiciona a mesma ao modelo criado.
model.add_layer(Layer(5, 5, "Relu")) # Cria uma camada com 5 entradas, 5 saídas, com a ativação Relu e adiciona a mesma ao modelo criado.
model.add_layer(Layer(5, 5, "Relu")) # Cria uma camada com 5 entradas, 5 saídas, com a ativação Relu e adiciona a mesma ao modelo criado.
model.add_layer(Layer(5, 2, "sigmoid")) # Cria uma camada com 5 entradas, 2 saídas, com a ativação Sigmoid e adiciona a mesma ao modelo criado.

input_ = [np.random.randint(0, 20, 3) for _ in range(10)] # Cria entradas aleatorias com shape (3, ).

target = [np.random.randint(0, 10, 2) for _ in range(10)] # Cria "targets" aleatorios com shape (2, ).

model.train_model(input_, target, 0.8, 2000) # Treina o modelo dado com uma taxa de 0.8 e com 2000 epocas.

```
# Como salvar e carregar o seus modelos

```python

model.save_model("model.json")
model.load_model("model.json")

```

# Requisitos
python>=3.7.0
numpy>=1.12.0

# Inspirações

Tive uma pequena inspiração nas bibliotecas Pytorch e Keras.
