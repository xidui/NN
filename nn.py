import matplotlib.pyplot as plt
import numpy as np
import json


learning_rate = 0.1
regularization_lambda = 0.01
enable_regular = False
momentum = 0


def read_data(filename):
    train_data = np.genfromtxt(filename, delimiter=',')
    return train_data[:, :-1], train_data[:, -1]


def average_loss(W1, B1, W2, B2, X1, Y):
    size = Y.size
    U1 = X1.dot(W1) + B1  # U1: (3000, 100)
    # sigmoid
    X2 = 1 / (1 + np.exp(-U1))  # X2: (3000, 100)
    U2 = X2.dot(W2) + B2  # U2: (3000, 10)
    # softmax
    tmp = np.exp(U2)  # (3000, 10)
    output = tmp / np.sum(tmp, axis=1, keepdims=True)  # (3000, 10)
    data_loss = np.sum(-np.log(output[range(size), Y]))
    if enable_regular:
        data_loss += regularization_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return data_loss * (1. / size)


def predict(W1, B1, W2, B2, X):
    U1 = X.dot(W1) + B1
    X2 = 1 / (1 + np.exp(-U1))
    U2 = X2.dot(W2) + B2
    tmp = np.exp(U2)
    output = tmp / np.sum(tmp, axis=1, keepdims=True)
    return np.argmax(output, axis=1)


def save_model(W1, B1, W2, B2, name):
    model = {
        'W1': W1.tolist(),
        'B1': B1.tolist(),
        'W2': W2.tolist(),
        'B2': B2.tolist()
    }
    f = open('models/{0}'.format(name), 'w')
    f.write(json.dumps(model))
    f.close()


def single_layer_nn(hidden_dim=100, epoch=2000, input_dim=784, output_dim=10, seed=2017, momentum=0):
    np.random.seed(seed)
    X1, Y = read_data('digitstrain.txt')  # X1: (3000, 784), Y (3000, 1)
    X_validate, Y_validate = read_data('digitstest.txt')
    Y = Y.astype(np.int64)
    Y_validate = Y_validate.astype(np.int64)

    b1 = np.sqrt(6. / (input_dim + hidden_dim))
    W1 = np.random.uniform(-b1, b1, (input_dim, hidden_dim))
    B1 = np.zeros((1, hidden_dim))
    b2 = np.sqrt(6. / (hidden_dim + output_dim))
    W2 = np.random.uniform(-b2, b2, (hidden_dim, output_dim))
    B2 = np.zeros((1, output_dim))

    training_size = Y.size

    loss_train = []
    loss_valid = []
    epochs = []

    dW1_pre, dW2_pre, dB1_pre, dB2_pre = None, None, None, None
    for i in range(epoch):
        # forward
        U1 = X1.dot(W1) + B1  # U1: (3000, 100)
        # sigmoid
        X2 = 1 / (1 + np.exp(-U1))  # X2: (3000, 100)
        U2 = X2.dot(W2) + B2  # U2: (3000, 10)
        # softmax
        tmp = np.exp(U2)  # (3000, 10)
        output = tmp / np.sum(tmp, axis=1, keepdims=True)  # (3000, 10)

        # backward
        D3 = output
        D3[range(training_size), Y] -= 1  # D3: (3000, 10)
        dW2 = X2.T.dot(D3) / training_size  # (100, 10)  Do we need to divide trainning_size here??
        dB2 = np.sum(D3, axis=0, keepdims=True) / training_size # (1, 10)
        D2 = D3.dot(W2.T) * (X2 * (1 - X2))  # (3000, 100)
        dW1 = X1.T.dot(D2) / training_size # (784, 100)  Do we need to divide trainning_size here??
        dB1 = np.sum(D2, axis=0, keepdims=True) / training_size # (1, 100)

        if enable_regular:
            dW1 += regularization_lambda * W1 / training_size  # Do we need to divide trainning_size here??
            dW2 += regularization_lambda * W2 / training_size  # Do we need to divide trainning_size here??

        if dW1_pre is None:
            dW1_pre = np.zeros(dW1.shape)
            dW2_pre = np.zeros(dW2.shape)
            dB1_pre = np.zeros(dB1.shape)
            dB2_pre = np.zeros(dB2.shape)

        dW1_pre = -learning_rate * dW1 + momentum * dW1_pre
        dW2_pre = -learning_rate * dW2 + momentum * dW2_pre
        dB1_pre = -learning_rate * dB1 + momentum * dB1_pre
        dB2_pre = -learning_rate * dB2 + momentum * dB2_pre

        # update
        W1 += dW1_pre
        B1 += dB1_pre
        W2 += dW2_pre
        B2 += dB2_pre

        if (i + 1) % 1000 == 0:
            name='seed_{0}_epoch_{1}_rate_{2}'.format(seed, i + 1, learning_rate)
            save_model(W1, B1, W2, B2, name)
            plt.plot(epochs, loss_train, color='blue')
            plt.plot(epochs, loss_valid, color='green')
            plt.xlabel('epoches')
            plt.ylabel('average loss')
            plt.legend(['train', 'valid'])
            plt.savefig('pictures/loss_{0}.png'.format(name))

        if (i + 1) % 10 == 0:
            epochs.append(i + 1)

            # loss of train data
            lt = average_loss(W1, B1, W2, B2, X1, Y)
            loss_train.append(lt)

            lv = average_loss(W1, B1, W2, B2, X_validate, Y_validate)
            loss_valid.append(lv)
            # loss of validation data
            print("itr {0}: {1}, {2}".format(i + 1, lt, lv))

            train_predict = predict(W1, B1, W2, B2, X1)
            result_train = train_predict == Y
            y_validate_predict = predict(W1, B1, W2, B2, X_validate)
            result_validate = y_validate_predict == Y_validate
            print("correctness: {0}/{1} {2}/{3}".format(
                np.sum(result_train), result_train.size,
                np.sum(result_validate), result_validate.size,
            ))


if __name__ == '__main__':
    single_layer_nn(hidden_dim=100, epoch=20000, seed=2017, momentum=momentum)
