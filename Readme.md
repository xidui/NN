# NN

### how to use this code
```python
for rate in [0.01, 0.2, 0.5]:
    for momentum in [0.0, 0.5, 0.9]:
        nn = NN(seed=2017, learning_rate=rate, dimensions=(784, 100, 10), active_function=sigmoid, momentum=momentum)
        nn.train(train_data_file='digitstrain.txt', validate_data_file='digitsvalid.txt', epoches=5000, batch=3000)
```
Graph will be generated in pictures directory.