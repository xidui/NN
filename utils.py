import numpy as np
import matplotlib.pyplot as plt
import os
import json


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def softmax(X):
    tmp = np.exp(X)
    return tmp / np.sum(tmp, axis=1, keepdims=True)


def tanh(X):
    return np.tanh(X)


def relu(X):
    return np.maximum(X, 0)



def plot_model(model, title, save_file):
    fig, ax = plt.subplots(nrows=10, ncols=10)
    for r in range(10):
        for c in range(10):
            img = model[:, r * 10 + c].reshape((28, 28))
            ax[r][c].axis('off')
            ax[r][c].imshow(img, cmap='gray')
    if title:
        fig.suptitle(title)
    fig.savefig(save_file)
    plt.close()


def get_lines(file):
    f = open(file, 'r')
    lines = f.readlines()
    f.close()
    return lines
