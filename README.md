# Rede_Neural_Numpy
Uma tentativa de fazer uma biblioteca de machine learning

# Como criar uma rede

```python
from layer import Layer # Importa a classe Layer.
from model import Model, np # Importa a classe Model.
from functions import Relu, sigmoid # Importa as funções de ativação.


model = Model() # Cria um modelo.
model.add_layer(Layer(3, 5, Relu)) # Cria uma camada com 3 entradas, 5 saídas e com a ativação Relu.
model.add_layer(Layer(5, 5, Relu)) 
model.add_layer(Layer(5, 5, Relu))
model.add_layer(Layer(5, 5, Relu))
model.add_layer(Layer(5, 2, sigmoid))

input_ = [np.random.randint(0, 20, 3) for _ in range(10)] # Cria entradas aleatorias com shape (3, ).

target = [np.random.randint(0, 10, 2) for _ in range(10)]# Cria entradas aleatorias com shape (2, ).

model.train_model(input_, target, 0.8, 2000) # Treina o modelo dado com uma taxa de 0.8 e com 2000 epocas.
```
# Requisitos
python>=3.7.0
numpy>=1.12.0
