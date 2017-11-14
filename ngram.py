import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import json
from collections import defaultdict


N = 4
EMBEDDING_DIMENSION = 16
VOCABULARY = 8000
HIDDEN = 128
BATCH_SIZE = 256  # todo
lr = 0.05
mom = 0.2


class NGram:
    def __init__(self, n, embedding_dimension, vocabulary_size, hidden, activation=False):
        self.n = n
        self.hidden = hidden
        self.embedding_dimension = embedding_dimension
        self.vocabulary_size = vocabulary_size
        self.activation = activation
        self.vocabulary = None

        self.rs = np.random.RandomState(2017)
        self.model = {
            'word_encoding': self.rs.normal(loc=0, scale=0.1, size=(vocabulary_size, embedding_dimension)),
            'embed_to_hid_weights': self.rs.normal(loc=0, scale=0.1, size=((n - 1) * embedding_dimension, hidden)),
            'embed_to_hid_bias': np.zeros(hidden),
            'hid_to_output_weights': self.rs.normal(loc=0, scale=1, size=(hidden, vocabulary_size)),
            'hid_to_output_bias': np.zeros(vocabulary_size)
        }
        self.momentum = {
            'd_word_encoding': np.zeros((vocabulary_size, embedding_dimension)),
            'd_embed_to_hid_weights': np.zeros(((n - 1) * embedding_dimension, hidden)),
            'd_embed_to_hid_bias': np.zeros(hidden),
            'd_hid_to_output_weights': np.zeros((hidden, vocabulary_size)),
            'd_hid_to_output_bias': np.zeros(vocabulary_size)
        }

        for dir in ['n_gram_models', 'n_gram_pictures']:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def create_vocabulary(self, train_file):
        dic = defaultdict(int)
        for l in utils.get_lines(train_file):
            for word in l.split():
                dic[word.lower()] += 1
        vocabulary = dict(sorted(dic.items(), key=lambda x: x[1], reverse=True)[:self.vocabulary_size - 3])
        vocabulary['UNK'] = 1
        vocabulary['START'] = 1
        vocabulary['END'] = 1
        tmp = 0
        for key in vocabulary.keys():
            vocabulary[key] = tmp
            tmp += 1
        return vocabulary

    def get_word_count(self, train_file):
        dic = defaultdict(int)
        for l in utils.get_lines(train_file):
            for word in l.split():
                dic[word.lower()] += 1
        word_count = dict(sorted(dic.items(), key=lambda x: x[1], reverse=True)[:self.vocabulary_size - 3])
        return word_count

    def get_label_from_words(self, lines):
        V = self.vocabulary
        data_set = []
        for l in lines:
            words = l.split()
            words = [word.lower() for word in words]
            words = ['START'] + words + ['END']
            for i in range(len(words) - self.n + 1):
                gram = [V['UNK'] if word not in V else V[word] for word in words[i: i + self.n]]
                data_set.append(gram)

        np.random.shuffle(data_set)  # todo shuffle will increase the perplexity
        label_x = []
        label_y = []
        for l in data_set:
            label_x.append(l[:-1])
            label_y.append(l[-1])
        return np.array(label_x), np.array(label_y)

    def get_input_vector_from_label(self, input_labels):
        input_vector = []
        for l in input_labels.tolist():
            tmp = []
            for index in l:
                tmp += self.model['word_encoding'][index, :].tolist()
            input_vector.append(tmp)
        return np.array(input_vector)

    def calculate_loss(self, x, y):
        tmp = x.dot(self.model['embed_to_hid_weights']) + self.model['embed_to_hid_bias']
        tmp = tmp.dot(self.model['hid_to_output_weights']) + self.model['hid_to_output_bias']
        tmp = utils.softmax(tmp)
        loss = np.sum(-np.log2(tmp[:, y]))
        loss /= x.shape[0]
        return loss

    def calculate_perplexity(self, val_file):
        total_perplexity = 0.0
        total_sentence = 0
        valid_size = 0
        val_loss = 0.0
        for line in utils.get_lines(val_file):
            valid_label_x, valid_y = self.get_label_from_words([line])
            valid_vector_x = self.get_input_vector_from_label(valid_label_x)
            total_sentence += 1
            valid_size += len(valid_y)
            # predict
            hidden_input_batch = valid_vector_x.dot(
                self.model['embed_to_hid_weights']) + self.model['embed_to_hid_bias']  # (batch, 128)
            if self.activation:
                hidden_input_batch = np.tanh(hidden_input_batch)
            output_batch = hidden_input_batch.dot(
                self.model['hid_to_output_weights']) + self.model['hid_to_output_bias']
            output_batch = utils.softmax(output_batch)
            val_loss += np.sum(-np.log2(output_batch[range(len(valid_y)), valid_y]))
            probs = output_batch[range(len(valid_y)), valid_y]
            # print probs
            probs = -np.log2(probs)
            perplexity = 2 ** np.mean(probs)
            total_perplexity += perplexity
        return val_loss / valid_size, total_perplexity / total_sentence

    def save_model(self, name):
        model = {
            'vocabulary': self.vocabulary,
            'word_encoding': self.model['word_encoding'].tolist(),
            'embed_to_hid_weights': self.model['embed_to_hid_weights'].tolist(),
            'embed_to_hid_bias': self.model['embed_to_hid_bias'].tolist(),
            'hid_to_output_weights': self.model['hid_to_output_weights'].tolist(),
            'hid_to_output_bias': self.model['hid_to_output_bias'].tolist()
        }
        f = open('n_gram_models/{0}'.format(name), 'w')
        f.write(json.dumps(model))
        f.close()

    def load_model(self, name):
        with open('n_gram_models/{0}'.format(name), 'r') as json_data:
            data = json.load(json_data)
            if 'vocabulary' in data:
                self.vocabulary = data['vocabulary']
            else:
                self.vocabulary = self.create_vocabulary('train.txt')
            self.model = {
                'word_encoding': np.array(data['word_encoding']),
                'embed_to_hid_weights': np.array(data['embed_to_hid_weights']),
                'embed_to_hid_bias': np.array(data['embed_to_hid_bias']),
                'hid_to_output_weights': np.array(data['hid_to_output_weights']),
                'hid_to_output_bias': np.array(data['hid_to_output_bias']),
            }

    def predict(self, words, times):
        V = self.vocabulary
        reverse_vocabulary = dict([(item[1], item[0]) for item in V.items()])
        for _ in range(times):
            labels = np.array([[V['UNK'] if word not in V else V[word] for word in words[-3:]]])
            vector_x = self.get_input_vector_from_label(labels)
            # predict
            hidden_input_batch = vector_x.dot(
                self.model['embed_to_hid_weights']) + self.model['embed_to_hid_bias']  # (batch, 128)
            if self.activation:
                hidden_input_batch = np.tanh(hidden_input_batch)
            output_batch = hidden_input_batch.dot(
                self.model['hid_to_output_weights']) + self.model['hid_to_output_bias']
            output_batch = utils.softmax(output_batch)
            chosen = output_batch.argmax(axis=1)
            new_word = reverse_vocabulary[chosen[0]]
            words.append(new_word)
            if new_word == 'END':
                break
        return words

    def plot_result(self, name, perplexity, train_loss, val_loss):
        epoches = range(1, len(perplexity) + 1)
        plt.figure(1)
        plt.plot(epoches, perplexity)
        plt.ylabel('perplexity')
        plt.xlabel('epoches')
        plt.title(name)
        plt.savefig('n_gram_pictures/perplexity_{0}'.format(name))
        plt.close()

        plt.figure(2)
        plt.plot(epoches, train_loss, color='blue')
        plt.plot(epoches, val_loss, color='green')
        plt.ylabel('average loss')
        plt.legend(['train', 'valid'])
        plt.title(name)
        plt.savefig('n_gram_pictures/loss_{0}.png'.format(name))
        plt.close()

    def get_nearest(self, word, count):
        reverse_vocabulary = dict([(item[1], item[0]) for item in self.vocabulary.items()])
        label = self.vocabulary[word] if word in self.vocabulary else self.vocabulary['UNK']
        vector = self.model['word_encoding'][label, :]
        result = []
        for i in range(self.vocabulary_size):
            tmp = self.model['word_encoding'][i, :]
            distance = np.sqrt(np.sum((vector - tmp) ** 2))
            result.append((distance, reverse_vocabulary[i]))
        tmp = sorted(result, key=lambda x:x[0])[0:count]
        return [(round(x[0], 3), x[1]) for x in tmp]

    def train(self, train_file, val_file, lr, epochs, batch, mom=0.5):
        # handle input
        self.vocabulary = self.create_vocabulary(train_file)
        lines = utils.get_lines('train.txt')
        train_label_x, train_y = self.get_label_from_words(lines)
        train_size = train_label_x.shape[0]

        perplexity_list = []
        train_loss_list = []
        val_loss_list = []

        # train
        for epoch in range(epochs):
            start = 0
            train_loss = 0.0
            while start < train_size:
                # print start, train_size
                end = min(start + batch, train_size)
                batch_size = end - start
                train_label_x_batch = train_label_x[start:start+batch_size, :]
                train_vector_x_batch = self.get_input_vector_from_label(train_label_x_batch)
                y_batch = train_y[start: start+batch_size]
                # forward
                hidden_input_batch = train_vector_x_batch.dot(
                    self.model['embed_to_hid_weights']) + self.model['embed_to_hid_bias']  # (batch, 128)
                if self.activation:
                    hidden_input_batch = np.tanh(hidden_input_batch)
                output_batch = hidden_input_batch.dot(
                    self.model['hid_to_output_weights']) + self.model['hid_to_output_bias']
                output_batch = utils.softmax(output_batch)

                x = output_batch[range(batch_size), y_batch]
                y = np.max(output_batch, axis=1)
                # loss
                train_loss += np.sum(-np.log2(output_batch[range(batch_size), y_batch]))

                # backward
                D = output_batch  # (batch, 8000)
                D[range(batch_size), y_batch] -= 1  # (batch, 8000)
                d_hid_to_output_weights = hidden_input_batch.T.dot(
                    D) / batch_size  # (128, batch) * (batch, 8000) = (128, 8000)
                d_hid_to_output_bias = np.sum(D, axis=0) / batch_size  # (8000)

                D = D.dot(self.model['hid_to_output_weights'].T)  # (batch, 8000) * (8000, 128) = (batch, 128)
                if self.activation:
                    D = D * (1 - hidden_input_batch * hidden_input_batch)
                d_embed_to_hid_weights = train_vector_x_batch.T.dot(D) / batch_size  # (48, batch) * (batch, 128) = (48, 128)
                d_embed_to_hid_bias = np.sum(D, axis=0) / batch_size  # (128)

                D = D.dot(self.model['embed_to_hid_weights'].T)  # (batch, 128) * (128, 48) = (batch, 48)
                d_word_encoding = np.zeros((self.vocabulary_size, self.embedding_dimension))
                for i in range(batch_size):
                    for j in range(3):
                        tmp = D[i, self.embedding_dimension * j:self.embedding_dimension * (j + 1)]
                        input_index = train_label_x_batch[i, j]
                        d_word_encoding[input_index] += tmp
                d_word_encoding /= batch_size  # todo

                # update weight
                self.momentum['d_embed_to_hid_weights'] = -lr * d_embed_to_hid_weights + self.momentum['d_embed_to_hid_weights'] * mom
                self.momentum['d_embed_to_hid_bias'] = -lr * d_embed_to_hid_bias + self.momentum['d_embed_to_hid_bias'] * mom
                self.momentum['d_hid_to_output_weights'] = -lr * d_hid_to_output_weights + self.momentum['d_hid_to_output_weights'] * mom
                self.momentum['d_hid_to_output_bias'] = -lr * d_hid_to_output_bias + self.momentum['d_hid_to_output_bias'] * mom
                self.momentum['d_word_encoding'] = -lr * d_word_encoding + self.momentum['d_word_encoding'] * mom

                self.model['embed_to_hid_weights'] += self.momentum['d_embed_to_hid_weights']
                self.model['embed_to_hid_bias'] += self.momentum['d_embed_to_hid_bias']
                self.model['hid_to_output_weights'] += self.momentum['d_hid_to_output_weights']
                self.model['hid_to_output_bias'] += self.momentum['d_hid_to_output_bias']
                self.model['word_encoding'] += self.momentum['d_word_encoding']

                # update start position
                start += batch_size

            # calculate perplexity
            val_loss, perplexity = self.calculate_perplexity(val_file=val_file)

            perplexity_list.append(perplexity)
            val_loss_list.append(val_loss)
            train_loss_list.append(train_loss / train_size)

            print 'epoch: {0}, train loss: {1}, valid loss: {2}, perplexity: {3}'.format(
                epoch,
                train_loss / train_size,
                val_loss,
                perplexity
            )

            if (epoch + 1) % 10 == 0:
                name = 'epoch_{0}_n_{1}_hidden_{2}_activation_{3}'.format(
                    epoch + 1,
                    self.n,
                    self.hidden,
                    self.activation
                )
                self.save_model(name)
                self.plot_result(name, perplexity_list, train_loss_list, val_loss_list)
        return val_loss_list


if __name__ == '__main__':
    # problem 3.1
    # ngram = NGram(n=N,
    #               embedding_dimension=EMBEDDING_DIMENSION,
    #               vocabulary_size=VOCABULARY,
    #               hidden=HIDDEN)
    # vocabulary = ngram.create_vocabulary('train.txt')
    # reverse_vocabulary = dict([(item[1], item[0])for item in vocabulary.items()])
    # lines = utils.get_lines('train.txt')
    # gram_count = defaultdict(int)
    # for l in lines:
    #     words = l.split()
    #     words = [word.lower() for word in words]
    #     words = ['START'] + words + ['END']
    #     for i in range(len(words) - 3):
    #         gram = [vocabulary['UNK'] if word not in vocabulary else vocabulary[word] for word in words[i: i+N]]
    #         gram_count[tuple(gram)] += 1
    #
    # y = sorted(gram_count.values(), reverse=True)
    # x = range(1, len(y) + 1)
    # plt.plot(x[:1000], y[:1000], color='blue')
    # plt.title("Gram Distribution")
    # plt.xlabel("word")
    # plt.ylabel("Frequency")
    # plt.savefig('ngram_distribution.png')
    # plt.close()
    # items = sorted(gram_count.items(), key=lambda x:x[1], reverse=True)[:50]
    # index = 1
    # for grams, count in items:
    #     words = [reverse_vocabulary[g] for g in grams]
    #     print '{2: <2} |{0: <30} | count:{1}'.format(' '.join(words), count, index)
    #     index += 1

    # problem 3.2
    # ngram = NGram(n=N,
    #               embedding_dimension=EMBEDDING_DIMENSION,
    #               vocabulary_size=VOCABULARY,
    #               hidden=HIDDEN)
    # val_loss_1 = ngram.train(train_file='train.txt', val_file='val.txt', lr=lr, epochs=100, batch=BATCH_SIZE, mom=mom)
    # ngram = NGram(n=N,
    #               embedding_dimension=EMBEDDING_DIMENSION,
    #               vocabulary_size=VOCABULARY,
    #               hidden=256)
    # ngram.train(train_file='train.txt', val_file='val.txt', lr=lr, epochs=100, batch=BATCH_SIZE, mom=mom)
    # ngram = NGram(n=N,
    #               embedding_dimension=EMBEDDING_DIMENSION,
    #               vocabulary_size=VOCABULARY,
    #               hidden=512)
    # ngram.train(train_file='train.txt', val_file='val.txt', lr=lr, epochs=100, batch=BATCH_SIZE, mom=mom)

    # problem 3.3
    # ngram = NGram(n=N,
    #               embedding_dimension=EMBEDDING_DIMENSION,
    #               vocabulary_size=VOCABULARY,
    #               hidden=HIDDEN,
    #               activation=True)
    # val_loss_2 = ngram.train(train_file='train.txt', val_file='val.txt', lr=lr, epochs=100, batch=BATCH_SIZE, mom=mom)
    # epoches = range(1, len(val_loss_1) + 1)
    # plt.figure(3)
    # plt.plot(epoches, val_loss_1, color='blue')
    # plt.plot(epoches, val_loss_2, color='green')
    # plt.ylabel('validation loss')
    # plt.legend(['linear', 'tanh'])
    # plt.title('linear vs tanh')
    # plt.savefig('n_gram_pictures/loss_compare.png')
    # plt.close()

    # problem 3.4
    # ngram = NGram(n=N,
    #               embedding_dimension=EMBEDDING_DIMENSION,
    #               vocabulary_size=VOCABULARY,
    #               hidden=HIDDEN,
    #               activation=False)
    # ngram.load_model('epoch_60_n_4_hidden_128_activation_True')
    # words_list = [
    #     ['said', '0', '*t*-1'],
    #     ['million', '*u*', ','],
    #     ['the', 'company', 'said'],
    #     ['new', 'york', 'stock'],
    #     ['york', 'stock', 'exchange'],
    #     ['and', 'chief', 'executive'],
    #     ['says', '0', '*t*-1'],
    #     [',', 'for', 'example'],
    #     ['president', 'and', 'chief'],
    #     ['START', 'the', 'company']
    # ]
    # for words in words_list:
    #     sentence = ngram.predict(words, 10)
    #     for word in sentence:
    #         print word,
    #     print ''

    # print ngram.get_nearest('for', 11)
    #
    # # problem 3.5
    ngram = NGram(n=N,
                  embedding_dimension=2,
                  vocabulary_size=VOCABULARY,
                  hidden=HIDDEN,
                  activation=False)
    # ngram.train(train_file='train.txt', val_file='val.txt', lr=lr, epochs=100, batch=BATCH_SIZE, mom=mom)
    vocabulary = ngram.create_vocabulary('train.txt')
    reverse_vocabulary = dict([(item[1], item[0]) for item in vocabulary.items()])
    word_count = ngram.get_word_count('train.txt')
    ngram.load_model('epoch_100_n_4_hidden_128_activation_False')
    tmp = list(enumerate(ngram.model['word_encoding'].tolist()))

    chosen_i = None
    chosen_v = None
    flag = False
    while not flag:
        np.random.shuffle(tmp)
        for i, v in tmp[:500]:
            if reverse_vocabulary[i] == 'discuss':
                chosen_i = i
                chosen_v = v
                flag = True
                temp = tmp[:500]
                break
    print reverse_vocabulary[chosen_i]

    result = []
    for index, vector in temp:
        distance = np.sqrt(np.sum((np.array(vector) - np.array(chosen_v)) ** 2))
        result.append((index, reverse_vocabulary[index], distance,  vector))
    result = sorted(result, key=lambda x: x[2])[0:20]
    print [x[1] for x in result]
    plt.scatter([x[0] for _, x in temp], [x[1] for _, x in temp])
    plt.scatter([x[3][0] for x in result], [x[3][1] for x in result], color='green')
    plt.scatter([chosen_v[0]], chosen_v[1], color='red')
    plt.title('500 words')
    plt.show()
