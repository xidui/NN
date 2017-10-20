import matplotlib.pyplot as plt
import numpy as np
import json
import os

eps = 0.01

def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def softmax(X):
    tmp = np.exp(X)
    return tmp / np.sum(tmp, axis=1, keepdims=True)


def tanh(X):
    return np.tanh(X)


def relu(X):
    return np.maximum(X, 0)


class AE:
    def __init__(self, learning_rate=0.1, dimensions=(784, 100, 10),
                 seed=2017, active_function=sigmoid, dropout=0.1):
        self.rs = np.random.RandomState(seed)
        self.seed = seed
        self.learning_rate = learning_rate
        self.dimensions = dimensions
        self.active_function = active_function
        self.dropout = dropout
        self.models = {
            'W': [],
            'B': [],
        }

        for i in range(len(self.dimensions) - 1):
            input_dim = self.dimensions[i]
            output_dim = self.dimensions[i + 1]
            self.models['W'].append(self.rs.normal(loc=0, scale=0.1, size=(input_dim, output_dim)))
            self.models['B'].append(np.zeros((1, output_dim)))

        for dir in ['ae_pictures', 'ae_models', 'ae_plot_models']:
            if not os.path.exists(dir):
                os.makedirs(dir)

    @staticmethod
    def read_data(filename):
        train_data = np.genfromtxt(filename, delimiter=',')
        return train_data[:, :-1], train_data[:, -1]

    def save_model(self, name):
        model = {
            'W': [W.tolist() for W in self.models['W']],
            'B': [B.tolist() for B in self.models['B']],
        }
        f = open('ae_models/{0}'.format(name), 'w')
        f.write(json.dumps(model))
        f.close()

    def plot_model(self, title):
        fig, ax = plt.subplots(nrows=10, ncols=10)
        for r in range(10):
            for c in range(10):
                img = self.models['W'][0][:, r * 10 + c].reshape((28, 28))
                ax[r][c].axis('off')
                ax[r][c].imshow(img, cmap='gray')
        if title:
            fig.suptitle(title)
        fig.savefig('ae_plot_models/{0}.png'.format(title))

    def plot_v_vhv_picture(self, v, vhv):
        fig, ax = plt.subplots(nrows=10, ncols=10)
        choice = [int(i) for i in self.rs.uniform(low=0, high=3000, size=50).tolist()]
        for key, value in enumerate(choice):
            origin = v[value,:].reshape((28, 28))
            after  = vhv[value,:].reshape((28, 28))
            ax[key / 5][key % 5 * 2].axis('off')
            ax[key / 5][key % 5 * 2 + 1].axis('off')
            ax[key / 5][key % 5 * 2].imshow(origin, cmap='gray')
            ax[key / 5][key % 5 * 2 + 1].imshow(after, cmap='gray')
        fig.savefig('aepicture.png')

    def forward_helper(self, X):
        intermediate_X = [X]
        for i in range(len(self.dimensions) - 1):
            W = self.models['W'][i]
            B = self.models['B'][i]
            X = intermediate_X[-1]
            U = X.dot(W) + B

            if i < len(self.dimensions) - 1:
                intermediate_X.append(self.active_function(U))
        return intermediate_X

    def average_loss(self, X):
        error = X - self.forward_helper(X)[-1]
        loss = 1.0 / 2.0 * (error ** 2).sum() / error.shape[0]
        return loss

    def train(self, train_data_file, epoches, start_from_latest=True, batch=32):
        start_from = 0
        X_train, Y_train = AE.read_data(train_data_file)  # X1: (3000, 784), Y (3000, 1)
        Y_train = Y_train.astype(np.int64)
        training_size = Y_train.size

        for epoch in range(start_from + 1, epoches + 1):
            batch_start = 0
            while batch_start < training_size:
                batch_end = batch_start + batch
                if batch_start + batch > training_size:
                    batch_end = training_size
                X_train_batch = X_train[range(batch_start, batch_end), :]
                Y_train_batch = Y_train[range(batch_start, batch_end)]
                training_size_batch = Y_train_batch.size
                # forward
                # dropout
                X_train_batch = X_train_batch * self.rs.binomial(n=1, p=1-self.dropout, size=X_train_batch.shape)
                intermediate_X = self.forward_helper(X_train_batch)

                # backward
                output = intermediate_X[-1]
                D = -(X_train_batch - output) * output * (1 - output)
                for i in range(len(self.dimensions) - 1)[::-1]:
                    dW = intermediate_X[i].T.dot(D) / training_size_batch
                    dB = np.sum(D, axis=0, keepdims=True) / training_size_batch
                    self.models['W'][i] += -self.learning_rate * dW
                    self.models['B'][i] += -self.learning_rate * dB
                    # calculate new D
                    D = D.dot(self.models['W'][i].T)
                    if self.active_function == sigmoid:
                        D = D * (intermediate_X[i] * (1 - intermediate_X[i]))
                    elif self.active_function == tanh:
                        D = D * (1 - intermediate_X[i] * intermediate_X[i])
                    elif self.active_function == relu:
                        tmp = np.maximum(intermediate_X[i], 0.0)
                        tmp[tmp > 0] = 1.0
                        D = D * tmp
                    else:
                        print 'some thing wrong'

                batch_start += batch

            print 'epoch:{0}, loss:{1}'.format(epoch, self.average_loss(X_train))
            if epoch % 50 == 0:
                name = 'lr_{0}_dropout_{1}_batch_{2}_epoch_{3}'.format(self.learning_rate, self.dropout, batch, epoch)
                if self.dimensions[1] != 100:
                    name = 'lr_{0}_hidden_{1}_dropout_{2}_batch_{3}_epoch_{4}'.format(
                        self.learning_rate, self.dimensions[1], self.dropout, batch, epoch)
                # self.plot_model(name)
                self.save_model(name)
                # self.plot_v_vhv_picture(X_train, self.forward_helper(X_train)[-1])
