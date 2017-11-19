# NN RBM AE

### how to use this code
```python
for rate in [0.01, 0.2, 0.5]:
    for momentum in [0.0, 0.5, 0.9]:
        nn = NN(seed=2017, learning_rate=rate, dimensions=(784, 100, 10), active_function=sigmoid, momentum=momentum)
        nn.train(train_data_file='digitstrain.txt', validate_data_file='digitsvalid.txt', epoches=5000, batch=3000)
```
Graph will be generated in pictures directory.

### how to plot model
```python
from nn import NN
n = NN()
n.load_model('model_data_in_models_dir')
n.plot('model_data_in_models_dir')
```

### For homework2

RBM is implemented in `rbm.py`.

Autoencoder is implemented in `ae.py`.

Experiments for each question is in `__init__.py`. If you want to run the experient for specific question,
simply uncomment the code and run `python __init__.py` is ok.

### For homework3

My implementation for ngram is in `ngram.py`.
I used `GRU` and `pytorch` to implement ngram for extra credit. It is in `gru.py`

To run my code:
```python
python ngram.py
python gru.py
```

To run each of the sub problem, comment and uncomment the corresponding code will be ok.
