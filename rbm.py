import numpy as np
import matplotlib.pyplot as plt
import os
import json


def sigmoid(X):
    return 1. / (1 + np.exp(-X))


class RBM:
    def __init__(self, hidden=100, visible=28*28, seed=2017):
        self.rs = np.random.RandomState(seed)
        self.dimension = (visible, hidden)
        self.W = self.rs.normal(loc=0, scale=0.1, size=self.dimension)
        self.hBias = np.zeros(hidden)
        self.vBias = np.zeros(visible)

        for dir in ['rbm_models', 'rbm_plot_models', 'rbm_pictures']:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def load_model(self, filename):
        with open('rbm_models/{0}'.format(filename), 'r') as json_data:
            data = json.load(json_data)
            self.W = np.array(data['W'])
            self.hBias = np.array(data['H'])
            self.vBias = np.array(data['V'])

    @staticmethod
    def read_data(filename):
        train_data = np.genfromtxt(filename, delimiter=',')
        return train_data[:, :-1], train_data[:, -1]

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
        fig.savefig('rbmpicture.png')
        plt.close()

    def plot_model(self, title):
        fig, ax = plt.subplots(nrows=10, ncols=10)
        for r in range(10):
            for c in range(10):
                img = self.W[:, r * 10 + c].reshape((28, 28))
                ax[r][c].axis('off')
                ax[r][c].imshow(img, cmap='gray')
        if title:
            fig.suptitle(title)
        fig.savefig('rbm_plot_models/{0}.png'.format(title))
        plt.close()

    def plot_error(self, title, train_error, valid_error):
        epoches = range(1, len(train_error) + 1)
        plt.figure(1)
        # plot loss
        # ax[0].set_ylim([0, 1.0])
        plt.plot(epoches, train_error, color='blue')
        plt.plot(epoches, valid_error, color='green')
        plt.ylabel('average loss')
        plt.legend(['train', 'valid'])
        plt.title(title)
        plt.savefig('rbm_pictures/loss_{0}.png'.format(title))
        plt.close()

    def save_model(self, name):
        model = {
            'W': self.W.tolist(),
            'H': self.hBias.tolist(),
            'V': self.vBias.tolist()
        }
        f = open('rbm_models/{0}'.format(name), 'w')
        f.write(json.dumps(model))
        f.close()

    def contrast_divergence(self, lr, k, Input):
        batch_size = Input.shape[0]
        V_sample = np.copy(Input)  # (3000 * 784)
        pH, H_sample = self.sample_h_given_v(V_sample)
        pH1 = np.copy(pH)
        for _ in range(k):
            pV, V_sample = self.sample_v_given_h(H_sample)
            pH, H_sample = self.sample_h_given_v(V_sample)

        self.W += lr * (Input.T.dot(pH1) - V_sample.T.dot(pH)) / batch_size # (784, 3000) * (3000, 100) = (784, 100)
        self.hBias += lr * (np.mean(pH1 - pH, axis=0))                      # (3000, 100) -> (100, )
        self.vBias += lr * (np.mean(Input - V_sample, axis=0))              # (3000, 784) -> (784, )

    def sample_h_given_v(self, V):
        '''
        V: (3000, 784)
        '''
        pH = sigmoid(V.dot(self.W) + self.hBias) # (3000, 784) * (784, 100) = (3000, 100)
        H_sample = self.rs.binomial(n=1, p=pH, size=pH.shape)
        return pH, H_sample

    def sample_v_given_h(self, H):
        '''
        H: (3000, 100)
        '''
        pV = sigmoid(H.dot(self.W.T) + self.vBias) # (30-00, 100) * (100, 784) = (3000, 784)
        V_sample = self.rs.binomial(n=1, p=pV, size=pV.shape)
        return pV, V_sample

    def reconstruction_cross_entropy(self, Input):
        pH = sigmoid(np.dot(Input, self.W) + self.hBias)
        pV = sigmoid(np.dot(pH, self.W.T) + self.vBias)
        cross_entropy = - np.mean(np.sum(Input * np.log(pV) + (1. - Input) * np.log(1. - pV), axis=1))
        return cross_entropy

    def train(self, lr, k, batch_size, epoches, train_data, valid_data):
        train_error = []
        valid_error = []
        for epoch in range(1, epoches + 1):
            l = range(train_data.shape[0])
            np.random.shuffle(l)
            start = 0
            while start < len(l):
                chosen = l[start:start + batch_size]
                chosen = train_data[chosen, :]
                self.contrast_divergence(lr, k, chosen)
                start += batch_size
            cross_entropy = self.reconstruction_cross_entropy(train_data)
            cross_entropy_2 = self.reconstruction_cross_entropy(valid_data)
            print '{0} {1} {2}'.format(epoch, cross_entropy, cross_entropy_2)
            train_error.append(cross_entropy)
            valid_error.append(cross_entropy_2)

            if epoch % 50 == 0:
                name = 'lr_{0}_k_{1}_batch_{2}_epoch_{3}'.format(lr, k, batch_size, epoch)
                if self.dimension[1] != 100:
                    name = 'lr_{0}_hidden_{1}_k_{2}_batch_{3}_epoch_{4}'.format(lr, self.dimension[1], k, batch_size, epoch)
                # self.plot_model(name)
                self.save_model(name)
                self.plot_error(name, train_error, valid_error)

        # pH, H_sample = self.sample_h_given_v(train_data)
        # pV, V_sample = self.sample_v_given_h(H_sample)
        # # self.plot_v_vhv_picture(train_data, V_sample)
        return train_error, valid_error


if __name__ == '__main__':
    rbm = RBM(hidden=50)
    X_train, Y_train = RBM.read_data('digitstrain.txt')
    X_valid, Y_valid = RBM.read_data('digitsvalid.txt')
    rbm.train(0.1, 1, 32, 300, X_train, X_valid)
