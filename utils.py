import numpy as np
import matplotlib.pyplot as plt
import os
import json


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
