import numpy as np
import utils
from collections import defaultdict


N = 4
EMBEDDING_DIMENSION = 16
VOCABULARY = 8000
HIDDEN = 128
BATCH_SIZE = 128  # todo
lr = 0.1


class NGram:
    def __init__(self, n, embedding_dimension, vocabulary_size, hidden):
        self.n = n
        self.hidden = hidden
        self.embedding_dimension = embedding_dimension
        self.vocabulary_size = vocabulary_size
        self.vocabulary = None
        self.model = {
            'word_encoding': np.random.randn(vocabulary_size, embedding_dimension),
            'embed_to_hid_weights': np.random.randn((n - 1) * embedding_dimension, hidden),
            'embed_to_hid_bias': np.zeros(hidden),
            'hid_to_output_weights': np.random.randn(hidden, vocabulary_size),
            'hid_to_output_bias': np.zeros(vocabulary_size)
        }

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

        # np.random.shuffle(data_set)  # todo shuffle will increase the perplexity
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

    def train(self, train_file, val_file, lr, epoches, batch):
        # handle input
        self.vocabulary = self.create_vocabulary(train_file)
        lines = utils.get_lines('train.txt')
        train_label_x, train_y = self.get_label_from_words(lines)
        train_size = train_label_x.shape[0]

        # train
        for epoch in range(epoches):
            start = 0
            train_loss = 0.0
            while start < train_size:
                # print start, train_size
                end = min(start + batch, train_size)
                batch_size = end - start
                train_label_x_batch = train_label_x[start:end, :]
                train_vector_x_batch = self.get_input_vector_from_label(train_label_x_batch)
                y_batch = train_y[start: end]
                # forward
                hidden_input_batch = train_vector_x_batch.dot(
                    self.model['embed_to_hid_weights']) + self.model['embed_to_hid_bias']  # (batch, 128)
                output_batch = hidden_input_batch.dot(
                    self.model['hid_to_output_weights']) + self.model['hid_to_output_bias']
                output_batch = utils.softmax(output_batch)

                # loss
                train_loss += np.sum(-np.log2(output_batch[range(batch_size), y_batch]))

                # backward
                D = output_batch  # (batch, 8000)
                D[range(batch_size), y_batch] -= 1  # (batch, 8000)
                d_hid_to_output_weights = hidden_input_batch.T.dot(
                    D) / batch_size  # (128, batch) * (batch, 8000) = (128, 8000)
                d_hid_to_output_bias = np.sum(D, axis=0) / batch_size  # (8000)

                D = D.dot(self.model['hid_to_output_weights'].T)  # (batch, 8000) * (8000, 128) = (batch, 128)
                d_embed_to_hid_weights = train_vector_x_batch.T.dot(
                    D) / batch_size  # (48, batch) * (batch, 128) = (48, 128)
                d_embed_to_hid_bias = np.sum(D, axis=0) / batch_size  # (128)

                D = D.dot(self.model['embed_to_hid_weights'].T)  # (batch, 128) * (128, 48) = (batch, 48)
                d_word_encoding = np.zeros((self.vocabulary_size, self.embedding_dimension))
                for i in range(batch_size):
                    for j in range(3):
                        tmp = D[i, 16 * j:16 * (j + 1)]
                        input_index = train_label_x_batch[i, j]
                        d_word_encoding[input_index] += tmp
                d_word_encoding /= batch_size  # todo
                # update weight
                self.model['embed_to_hid_weights'] -= lr * d_embed_to_hid_weights
                self.model['embed_to_hid_bias'] -= lr * d_embed_to_hid_bias
                self.model['hid_to_output_weights'] -= lr * d_hid_to_output_weights
                self.model['hid_to_output_bias'] -= lr * d_hid_to_output_bias
                self.model['word_encoding'] -= lr * d_word_encoding
                # update start position
                start += batch_size

            # calculate perplexity
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
                output_batch = hidden_input_batch.dot(
                    self.model['hid_to_output_weights']) + self.model['hid_to_output_bias']
                output_batch = utils.softmax(output_batch)
                val_loss += np.sum(-np.log2(output_batch[range(len(valid_y)), valid_y]))
                probs = output_batch[range(len(valid_y)), valid_y]
                # print probs
                probs = -np.log2(probs)
                perplexity = 2 ** np.mean(probs)
                total_perplexity += perplexity

            print 'epoch: {0}, train loss: {1}, valid loss: {2}, perplexity: {3}'.format(
                epoch,
                train_loss / train_size,
                val_loss / valid_size,
                total_perplexity / total_sentence
            )


if __name__ == '__main__':
    np.random.seed(2017)

    # problem 3.1
    # lines = get_lines('train.txt')
    # gram_count = defaultdict(int)
    # for l in lines:
    #     words = l.split()
    #     words = [word.lower() for word in words]
    #     words = ['START'] + words + ['END']
    #     for i in range(len(words) - 3):
    #         gram = [vocabulary['UNK'] if word not in vocabulary else vocabulary[word] for word in words[i: i+N]]
    #         gram_count[tuple(gram)] += 1
    # print sorted(gram_count.values(), reverse=True)[:50]

    # problem 3.2
    ngram = NGram(n=N,
                  embedding_dimension=EMBEDDING_DIMENSION,
                  vocabulary_size=VOCABULARY,
                  hidden=HIDDEN)
    ngram.train(train_file='train.txt', val_file='val.txt', lr=lr, epoches=500, batch=BATCH_SIZE)
