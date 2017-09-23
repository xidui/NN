import matplotlib.pyplot as plt
import numpy as np
import json
import os


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def softmax(X):
    tmp = np.exp(X)
    return tmp / np.sum(tmp, axis=1, keepdims=True)


def tanh(X):
    return np.tanh(X)


def relu(X):
    return np.maximum(X, 0)


def plot(filename):
    n = NN()
    n.load_model(filename)
    plot_data = n.load_plot_data(filename)
    n.plot_data(filename, filename, plot_data)


class NN:
    def __init__(self, learning_rate=0.5, reg_lambda=0.0, momentum=0.0,
                 dimensions=(784, 100, 10), seed=2017, active_function=sigmoid):
        np.random.seed(seed)
        self.seed = seed
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.momentum = momentum
        self.dimensions = dimensions
        self.active_function = active_function
        self.models = {
            'W': [],
            'B': [],
            'dW': [],
            'dB': []
        }

        for i in range(len(self.dimensions) - 1):
            input_dim = self.dimensions[i]
            output_dim = self.dimensions[i + 1]
            b = np.sqrt(6. / (input_dim + output_dim))
            self.models['W'].append(np.random.uniform(-b, b, (input_dim, output_dim)))
            self.models['B'].append(np.zeros((1, output_dim)))
            self.models['dW'].append(np.zeros((input_dim, output_dim)))
            self.models['dB'].append(np.zeros((1, output_dim)))

        for dir in ['pictures', 'models', 'plot_data']:
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
            'dW': [dW.tolist() for dW in self.models['dW']],
            'dB': [dB.tolist() for dB in self.models['dB']]
        }
        f = open('models/{0}'.format(name), 'w')
        f.write(json.dumps(model))
        f.close()

    def load_model(self, name):
        with open('models/{0}'.format(name), 'r') as json_data:
            model = json.load(json_data)
            model['W'] = [np.array(W) for W in model['W']]
            model['B'] = [np.array(B) for B in model['B']]
            model['dW'] = [np.array(dW) for dW in model['dW']]
            model['dB'] = [np.array(dB) for dB in model['dB']]
            self.models = model

    def plot_model(self, title):
        fig, ax = plt.subplots(nrows=10, ncols=10)
        for r in range(10):
            for c in range(10):
                img = self.models['W'][0][:, r * 10 + c].reshape((28, 28))
                ax[r][c].axis('off')
                ax[r][c].imshow(img, cmap='gray')
        if title:
            fig.suptitle(title)
        fig.show()

    def plot_data(self, title, name, plot_data):
        fig, ax = plt.subplots(nrows=2, ncols=1)
        # plot loss
        ax[0].set_ylim([0, 1.0])
        ax[0].plot(plot_data['epochs'], plot_data['loss_train'], color='blue')
        ax[0].plot(plot_data['epochs'], plot_data['loss_valid'], color='green')
        ax[0].set_xlabel('epoches')
        ax[0].set_ylabel('average loss')
        ax[0].legend(['train', 'valid'])
        # plot incorrect
        ax[1].set_ylim([0, 1.0])
        ax[1].plot(plot_data['epochs'], plot_data['train_incorrect'], color='blue')
        ax[1].plot(plot_data['epochs'], plot_data['valid_incorrect'], color='green')
        ax[1].set_xlabel('epoches')
        ax[1].set_ylabel('incorrectness')
        ax[1].legend(['train', 'valid'])
        fig.suptitle(title)

        if plot_data['best_fit_epoch']:
            bfe = plot_data['best_fit_epoch']
            ax[0].vlines(bfe, 0,
                         max(max(plot_data['loss_train']), max(plot_data['loss_valid'])),
                         color='red',
                         linestyles='dashed',
                         label='overfit')
            ax[0].plot([bfe], [[plot_data['loss_train'][bfe/10-1]]], marker='o', markersize=3, color='red')
            ax[0].annotate(' ' + str(round(plot_data['loss_train'][bfe/10-1], 6)),
                           xy=(bfe, plot_data['loss_train'][bfe/10-1]))
            ax[0].plot([bfe], [plot_data['loss_valid'][bfe / 10 - 1]], marker='o', markersize=3, color='red')
            ax[0].annotate(' ' + str(round(plot_data['loss_valid'][bfe / 10 - 1], 6)),
                           xy=(bfe, plot_data['loss_valid'][bfe / 10 - 1]))
            ax[1].vlines(bfe, 0,
                         max(max(plot_data['loss_train']), max(plot_data['loss_valid'])),
                         color='red',
                         linestyles='dashed',
                         label='overfit')
            ax[1].plot([bfe], [plot_data['train_incorrect'][bfe / 10 - 1]], marker='o', markersize=3, color='red')
            ax[1].annotate(' ' + str(round(plot_data['train_incorrect'][bfe / 10 - 1], 6)),
                           xy=(bfe, plot_data['train_incorrect'][bfe / 10 - 1]))
            ax[1].plot([bfe], [plot_data['valid_incorrect'][bfe / 10 - 1]], marker='o', markersize=3, color='red')
            ax[1].annotate(' ' + str(round(plot_data['valid_incorrect'][bfe / 10 - 1], 6)),
                           xy=(bfe, plot_data['valid_incorrect'][bfe / 10 - 1]))

        fig.savefig('pictures/loss_and_correct_{0}.png'.format(name))

    def save_plot_data(self, data, name):
        f = open('plot_data/{0}'.format(name), 'w')
        f.write(json.dumps(data))
        f.close()

    def load_plot_data(self, name):
        with open('plot_data/{0}'.format(name), 'r') as json_data:
            plot_data = json.load(json_data)
            return plot_data

    def forward_helper(self, X):
        intermediate_X = [X]
        for i in range(len(self.dimensions) - 1):
            W = self.models['W'][i]
            B = self.models['B'][i]
            X = intermediate_X[-1]
            U = X.dot(W) + B
            if i < len(self.dimensions) - 2:
                intermediate_X.append(self.active_function(U))
            else:
                # last layer use softmax
                intermediate_X.append(softmax(U))
        return intermediate_X

    def average_loss(self, X, Y):
        size = Y.size
        output = self.forward_helper(X)[-1]
        data_loss = np.sum(-np.log(output[range(size), Y]))
        data_loss += self.reg_lambda / 2 * sum([np.sum(np.square(W)) for W in self.models['W']])
        return data_loss * (1. / size)

    def predict(self, X):
        output = self.forward_helper(X)[-1]
        return np.argmax(output, axis=1)

    def train(self, train_data_file, validate_data_file, epoches, start_from_latest=True):
        plot_data = {
            'loss_train': [],
            'loss_valid': [],
            'train_incorrect': [],
            'valid_incorrect': [],
            'epochs': [],
            'best_fit_epoch': None
        }

        start_from = 0
        if start_from_latest:
            name = 'seed_{0}_rate_{1}_momentum_{2}_dim_{3}_act_{4}_epoch_'.format(
                self.seed, self.learning_rate,
                self.momentum, '.'.join([str(d) for d in self.dimensions]),
                self.active_function.__name__)
            start_from = 0
            for file in os.listdir('models'):
                if name in file:
                    if file.split('_')[-1] == 'best':
                        continue
                    start_from = max(int(file.split('_')[-1]), start_from)
            name = name + str(start_from)
            if os.path.exists('models/{0}'.format(name)) and os.path.exists('plot_data/{0}'.format(name)):
                self.load_model(name)
                plot_data = self.load_plot_data(name)

        if start_from > 3000 and plot_data['best_fit_epoch'] and start_from > 2 * plot_data['best_fit_epoch']:
            return

        X_train, Y_train = NN.read_data(train_data_file)  # X1: (3000, 784), Y (3000, 1)
        X_validate, Y_validate = NN.read_data(validate_data_file)
        Y_train = Y_train.astype(np.int64)
        Y_validate = Y_validate.astype(np.int64)
        training_size = Y_train.size

        for epoch in range(start_from + 1, epoches + 1):
            if epoch > 3000 and plot_data['best_fit_epoch'] and epoch > 2 * plot_data['best_fit_epoch']:
                break

            # forward
            intermediate_X = self.forward_helper(X_train)

            # backward
            output = intermediate_X[-1]
            D = output
            D[range(training_size), Y_train] -= 1
            for i in range(len(self.dimensions) - 1)[::-1]:
                dW = intermediate_X[i].T.dot(D) / training_size
                dB = np.sum(D, axis=0, keepdims=True) / training_size
                # weight decay (L2)
                dW += self.reg_lambda * self.models['W'][i] / training_size
                # momentum
                dW = -self.learning_rate * dW + self.momentum * self.models['dW'][i]
                dB = -self.learning_rate * dB + self.momentum * self.models['dB'][i]
                self.models['W'][i] += dW
                self.models['B'][i] += dB
                # save dW for next iteration
                self.models['dW'][i] = dW
                self.models['dB'][i] = dB
                # calculate new D
                D = D.dot(self.models['W'][i].T)
                if self.active_function == sigmoid:
                    D = D * (intermediate_X[i] * (1 - intermediate_X[i]))
                elif self.active_function == tanh:
                    D = D * (1 - intermediate_X[i] * intermediate_X[i])
                elif self.active_function == relu:
                    tmp = np.maximum(intermediate_X[i], 0.0)
                    tmp[tmp>0] = 1.0
                    D = D * tmp
                else:
                    print 'some thing wrong'

            # process results
            title = 'seed_{0}_rate_{1}_momentum_{2}_dim_{3}_act_{4}'.format(
                self.seed, self.learning_rate,
                self.momentum, '.'.join([str(d) for d in self.dimensions]),
                self.active_function.__name__)
            if self.reg_lambda != 0:
                title = '{0}_reg_{1}'.format(title, self.reg_lambda)
            name = '{0}_epoch_{1}'.format(title, epoch)
            if epoch % 10 == 0:
                plot_data['epochs'].append(epoch)

                # loss of train data
                lt = self.average_loss(X_train, Y_train)
                plot_data['loss_train'].append(lt)

                lv = self.average_loss(X_validate, Y_validate)
                if len(plot_data['loss_valid']) and lv >= plot_data['loss_valid'][-1] and plot_data['best_fit_epoch'] is None and epoch > 200:
                    plot_data['best_fit_epoch'] = epoch
                    self.save_model(name + '_best')

                plot_data['loss_valid'].append(lv)
                # loss of validation data
                print("itr {0}: {1}, {2}".format(epoch, lt, lv))

                train_predict = self.predict(X_train)
                result_train = train_predict == Y_train
                y_validate_predict = self.predict(X_validate)
                result_validate = y_validate_predict == Y_validate

                plot_data['train_incorrect'].append(1 - np.sum(result_train) / (result_train.size + 0.0))
                plot_data['valid_incorrect'].append(1 - np.sum(result_validate) / (result_validate.size + 0.0))
                print("incorrectness: {0} {1}".format(plot_data['train_incorrect'][-1], plot_data['valid_incorrect'][-1]))

            if epoch % 1000 == 0:
                self.save_model(name)
                self.plot_data(title=title, plot_data=plot_data, name=name)
                # save plot data
                self.save_plot_data(plot_data, name)

if __name__ == '__main__':
    # problem a, b
    # for seed in range(2017, 2022):
    #     nn = NN(seed=seed, learning_rate=0.2, dimensions=(784, 100, 10), momentum=0.0, active_function=sigmoid)
    #     nn.train(train_data_file='digitstrain.txt', validate_data_file='digitsvalid.txt', epoches=10000)
    # problem c

    # problem d
    # for rate in [0.01, 0.2, 0.5]:
    #     for momentum in [0.0, 0.5, 0.9]:
    #         nn = NN(seed=2017, learning_rate=rate, dimensions=(784, 100, 10), active_function=sigmoid, momentum=momentum)
    #         nn.train(train_data_file='digitstrain.txt', validate_data_file='digitsvalid.txt', epoches=20000)
    #
    # # problem e
    # for hidden in [20, 100, 200, 500]:
    #     nn = NN(seed=2017, learning_rate=0.3, dimensions=(784, hidden, 10), active_function=sigmoid, momentum=0.5)
    #     nn.train(train_data_file='digitstrain.txt', validate_data_file='digitsvalid.txt', epoches=10000)

    # problem f
    for reg in [0.1, 0.5, 0.05, 0.01]:
        nn = NN(seed=2017, learning_rate=0.5, dimensions=(784, 100, 10), active_function=sigmoid, momentum=0.5, reg_lambda=reg)
        nn.train(train_data_file='digitstrain.txt', validate_data_file='digitsvalid.txt', epoches=10000, start_from_latest=False)
    # problem g
    # problem h
    # problem i
