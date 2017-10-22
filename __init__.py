from ae import AE
from rbm import RBM
from nn import NN
import matplotlib.pyplot as plt
import json
import numpy as np
import utils


def load_model(path):
    with open(path, 'r') as json_data:
        data = json.load(json_data)
        if 'ae_models' in path:
            return data['W'][0], data['B'][0]
        else:
            return data['W'], data['H']


if __name__ == '__main__':
    seed = 2019
    # problem a
    # rbm = RBM()
    # X_train, Y_train = RBM.read_data('digitstrain.txt')
    # X_valid, Y_valid = RBM.read_data('digitstest.txt')
    # rbm.train(lr=0.01, k=1, batch_size=32, epoches=1000, train_data=X_train, valid_data=X_valid)

    # problem b
    # train_errors = {}
    # valid_errors = {}
    # for k in [1, 5, 10, 20]:
    #     rbm = RBM()
    #     X_train, Y_train = RBM.read_data('digitstrain.txt')
    #     X_valid, Y_valid = RBM.read_data('digitstest.txt')
    #     train_error, valid_error = rbm.train(
    #         lr=0.1, k=k, batch_size=32, epoches=1000, train_data=X_train, valid_data=X_valid)
    #     train_errors[k] = train_error
    #     valid_errors[k] = valid_error
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # legend = []
    # for key, train_error in train_errors.items():
    #     ax.plot(range(1, len(train_error) + 1), train_error)
    #     ax.set_ylabel('train_error')
    #     ax.set_xlabel('epoches')
    #     legend.append('k = {0}'.format(key))
    # ax.legend(legend)
    # plt.savefig('train_error_compare.png')
    # plt.close()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # legend = []
    # for key, valid_error in valid_errors.items():
    #     ax.plot(range(1, len(valid_error) + 1), valid_error)
    #     ax.set_ylabel('valid_error')
    #     ax.set_xlabel('epoches')
    #     legend.append('k = {0}'.format(key))
    # ax.legend(legend)
    # plt.savefig('valid_error_compare.png')
    # plt.close()

    # problem c
    # for k in [1,5,10,20]:
    #     rbm = RBM()
    #     rbm.load_model('lr_0.1_k_{0}_batch_32_epoch_1000'.format(k))
    #     V_sample = np.random.random((100, 784))
    #     for _ in range(1000):
    #         pH, H_sample = rbm.sample_h_given_v(V_sample)
    #         pV, V_sample = rbm.sample_v_given_h(H_sample)
    #     name = 'sample_figure_k_{0}'.format(k)
    #     utils.plot_model(pV.T, name, '{0}.png'.format(name))

    # problem d
    # ae = AE(dimensions=(784, 100, 784), learning_rate=0.01, seed=seed, dropout=0.0)
    # ae.train(train_data_file='digitstrain.txt', epoches=1000, batch=32)

    # problem e
    # ae = AE(dimensions=(784, 100, 784), learning_rate=0.01, seed=seed, dropout=0.1)
    # ae.train(train_data_file='digitstrain.txt', epoches=1000, batch=32)
    # ae = AE(dimensions=(784, 100, 784), learning_rate=0.01, seed=seed, dropout=0.3)
    # ae.train(train_data_file='digitstrain.txt', epoches=1000, batch=32)
    # ae = AE(dimensions=(784, 100, 784), learning_rate=0.01, seed=seed, dropout=0.5)
    # ae.train(train_data_file='digitstrain.txt', epoches=1000, batch=32)


    # problem f
    # models = {
    #     'baseline': (None, None, 'blue'),
    #     'rbm_1_500': load_model('rbm_models/lr_0.1_k_1_batch_32_epoch_500') + ('green',),
    #     # 'rbm_5_500': load_model('rbm_models/lr_0.1_k_5_batch_32_epoch_500') + ('yellow',),
    #     # 'rbm_10_500': load_model('rbm_models/lr_0.1_k_10_batch_32_epoch_500') + ('red',),
    #     # 'rbm_20_500': load_model('rbm_models/lr_0.1_k_20_batch_32_epoch_500') + ('pink',),
    #     # 'rbm': load_model('rbm_models/lr_0.01_k_1_batch_32_epoch_1000') + ('green',),
    #     'ae': load_model('ae_models/lr_0.01_dropout_0.0_batch_32_epoch_1000') + ('purple',),
    #     # 'ae_noise_0.1': load_model('ae_models/lr_0.01_dropout_0.1_batch_32_epoch_1000') + ('yellow',),
    #     # 'ae_noise_0.3': load_model('ae_models/lr_0.01_dropout_0.3_batch_32_epoch_1000') + ('pink',),
    #     'ae_noise_0.5': load_model('ae_models/lr_0.01_dropout_0.5_batch_32_epoch_1000') + ('yellow',),
    # }
    # for lr, mom, reg in [
    #     # [0.05, 0.5, 0.00001],
    #     # [0.1, 0.3, 0.00005],
    #     # [0.3, 0.5, 0.00005],
    #     # [0.2, 0.5, 0.00001],
    #     [0.2, 0.1, 0.00001],
    #     [0.5, 0, 0],
    #     [0.5, 0.2, 0],
    #     [0.5, 0, 0.00001]
    # ]:
    #     plot_data = []
    #     for key, value in models.items():
    #         nn = NN(seed=seed, learning_rate=lr, dimensions=(784, 100, 10), momentum=mom, reg_lambda=reg, W=value[0], B=value[1])
    #         plot_data.append((key, value[2], nn.train(train_data_file='digitstrain.txt', validate_data_file='digitstest.txt', epoches=300, batch=32, start_from_latest=False)))
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     legend = []
    #     for key, color, pd in plot_data:
    #         ax.plot(range(1, len(pd['valid_incorrect']) + 1), pd['valid_incorrect'], color=color)
    #         ax.set_ylabel('incorrect')
    #         ax.set_xlabel('epoches')
    #         legend.append(key)
    #     ax.legend(legend)
    #     name = 'lr_{0}_momentum_{1}_reg_{2}'.format(lr, mom, reg)
    #     ax.set_title(name)
    #     plt.savefig('compare_models_{0}.png'.format(name))
    #     plt.close()

    # problem g
    # for hidden in [50, 100, 200, 500]:
        # rbm = RBM(hidden=hidden)
        # X_train, Y_train = RBM.read_data('digitstrain.txt')
        # X_valid, Y_valid = RBM.read_data('digitstest.txt')
        # rbm.train(lr=0.1, k=1, batch_size=32, epoches=500, train_data=X_train, valid_data=X_valid)
        #
        # ae = AE(dimensions=(784, hidden, 784), learning_rate=0.01, seed=seed, dropout=0.0)
        # ae.train(train_data_file='digitstrain.txt', epoches=1000, batch=32)
        #
        # ae = AE(dimensions=(784, hidden, 784), learning_rate=0.01, seed=seed, dropout=0.2)
        # ae.train(train_data_file='digitstrain.txt', epoches=1000, batch=32)

        # if hidden == 100:
        #     models = {
        #         'baseline': (None, None, 'blue'),
        #         'rbm': load_model('rbm_models/lr_0.1_k_1_batch_32_epoch_500') + ('green',),
        #         'ae': load_model('ae_models/lr_0.01_dropout_0.0_batch_32_epoch_1000') + (
        #         'purple',),
        #         'ae_noise_0.2': load_model(
        #             'ae_models/lr_0.01_dropout_0.2_batch_32_epoch_1000') + ('yellow',),
        #     }
        # else:
        #     models = {
        #         'baseline': (None, None, 'blue'),
        #         'rbm': load_model('rbm_models/lr_0.1_hidden_{0}_k_1_batch_32_epoch_500'.format(hidden)) + ('green',),
        #         'ae': load_model('ae_models/lr_0.01_hidden_{0}_dropout_0.0_batch_32_epoch_1000'.format(hidden)) + (
        #         'purple',),
        #         'ae_noise_0.2': load_model(
        #             'ae_models/lr_0.01_hidden_{0}_dropout_0.2_batch_32_epoch_1000'.format(hidden)) + ('yellow',),
        #     }
        #
        # plot_data = []
        # for key, value in models.items():
        #     nn = NN(seed=seed, learning_rate=0.01, dimensions=(784, hidden, 10), momentum=0.5, reg_lambda=0.00001, W=value[0],
        #             B=value[1])
        #     plot_data.append((key, value[2],
        #                       nn.train(train_data_file='digitstrain.txt', validate_data_file='digitstest.txt',
        #                                epoches=300, batch=32, start_from_latest=False)))
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # legend = []
        # for key, color, pd in plot_data:
        #     ax.plot(range(1, len(pd['valid_incorrect']) + 1), pd['valid_incorrect'], color=color)
        #     ax.set_ylabel('incorrect')
        #     ax.set_xlabel('epoches')
        #     legend.append(key)
        # ax.legend(legend)
        # name = 'compare_models_{0}.png'.format(hidden)
        # ax.set_title(name)
        # plt.savefig(name)
        # plt.close()

    pass
