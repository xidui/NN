import numpy as np
import utils
from collections import defaultdict


def get_lines(file):
    f = open(file, 'r')
    lines = f.readlines()
    f.close()
    return lines


def create_vocabulary(train_file, size=8000):
    dic = defaultdict(int)
    for l in get_lines(train_file):
        for word in l.split():
            dic[word.lower()] += 1
    vocabulary = dict(sorted(dic.items(), key=lambda x: x[1], reverse=True)[:size - 3])
    vocabulary['UNK'] = 1
    vocabulary['START'] = 1
    vocabulary['END'] = 1
    tmp = 0
    for key in vocabulary.keys():
        vocabulary[key] = tmp
        tmp += 1
    return vocabulary


def convert_data(lines, vocabulary, N=4):
    gram_count = defaultdict(int)
    for l in lines:
        words = l.split()
        words = [word.lower() for word in words]
        words = ['START'] + words + ['END']
        for i in range(len(words) - N + 1):
            gram = [len(vocabulary) - 3 if word not in vocabulary else vocabulary[word] for word in words[i: i + N]]
            gram_count[tuple(gram)] += 1

    data_set = []
    for k, v in gram_count.items():
        data_set += [k] * v
    np.random.shuffle(data_set)
    train_vector_x = []
    train_label_x = []
    train_y = []
    for l in data_set:
        tmp = []
        for index in list(l)[:-1]:
            tmp += word_encoding[index, :].tolist()
        train_vector_x.append(tmp)
        train_label_x.append(list(l)[:-1])
        train_y.append(l[-1])
    train_vector_x = np.array(train_vector_x)
    train_label_x = np.array(train_label_x)
    return train_label_x, train_vector_x, train_y


def convert_to_label(vocabulary, words):
    return [len(vocabulary) - 3 if word not in vocabulary else vocabulary[word] for word in words]


N = 4
EMBEDDING_DIMENSION = 16
VACABULARY = 8000
HIDDEN = 128
word_encoding = None
embed_to_hid_weights = None
embed_to_hid_bias = None
hid_to_output_weights = None
hid_to_output_bias = None
BATCH_SIZE = 128
lr = 0.01


if __name__ == '__main__':
    vocabulary = create_vocabulary(train_file='train.txt', size=VACABULARY)

    # problem 3.1
    lines = get_lines('train.txt')
    gram_count = defaultdict(int)
    for l in lines:
        words = l.split()
        words = [word.lower() for word in words]
        words = ['START'] + words + ['END']
        for i in range(len(words) - 3):
            gram = [7997 if word not in vocabulary else vocabulary[word] for word in words[i: i+4]]
            gram_count[tuple(gram)] += 1
    # print sorted(gram_count.values(), reverse=True)[:50]

    # problem 3.2
    # initialize
    word_encoding = np.random.randn(VACABULARY, EMBEDDING_DIMENSION)
    embed_to_hid_weights = np.random.randn((N - 1) * EMBEDDING_DIMENSION, HIDDEN)
    embed_to_hid_bias = np.zeros(HIDDEN)
    hid_to_output_weights = np.random.randn(HIDDEN, VACABULARY)  # (128, 8000)
    hid_to_output_bias = np.zeros(VACABULARY)

    # handle input
    lines = get_lines('train.txt')
    train_label_x, train_vector_x, train_y = convert_data(lines, vocabulary, N)
    train_size = train_vector_x.shape[0]

    # train
    for epoch in range(200):
        start = 0
        while start < train_size:
            # print start, train_size
            end = min(start + BATCH_SIZE, train_size)
            batch_size = end - start
            train_vector_x_batch = train_vector_x[start:end, :]
            train_label_x_batch = train_label_x[start:end, :]
            y_batch = train_y[start: end]
            # forward
            hidden_input_batch = train_vector_x_batch.dot(embed_to_hid_weights) + embed_to_hid_bias  # (batch, 128)
            output_batch = hidden_input_batch.dot(hid_to_output_weights) + hid_to_output_bias
            output_batch = utils.softmax(output_batch)

            # backward
            D = output_batch  # (batch, 8000)
            D[range(batch_size), y_batch] -= 1  # (batch, 8000)
            d_hid_to_output_weights = hidden_input_batch.T.dot(D) / batch_size  # (128, batch) * (batch, 8000) = (128, 8000)
            d_hid_to_output_bias = np.sum(D, axis=0) / batch_size  # (8000)

            D = D.dot(hid_to_output_weights.T)  # (batch, 8000) * (8000, 128) = (batch, 128)
            d_embed_to_hid_weights = train_vector_x_batch.T.dot(D) / batch_size  # (48, batch) * (batch, 128) = (48, 128)
            d_embed_to_hid_bias = np.sum(D, axis=0) / batch_size  # (128)

            D = D.dot(embed_to_hid_weights.T)  # (batch, 128) * (128, 48) = (batch, 48)
            d_word_encoding = np.zeros((VACABULARY, EMBEDDING_DIMENSION))
            for i in range(batch_size):
                for j in range(3):
                    tmp = D[i, 16*j: 16*(j+1)]
                    input_index = train_label_x_batch[i, j]
                    d_word_encoding[input_index] += tmp
            d_word_encoding /= batch_size
            # update weight
            embed_to_hid_weights -= lr * d_embed_to_hid_weights
            embed_to_hid_bias -= lr * d_embed_to_hid_bias
            hid_to_output_weights -= lr * d_hid_to_output_weights
            hid_to_output_bias -= lr * d_hid_to_output_bias
            word_encoding -= lr * d_word_encoding
            # update start position
            start += batch_size

        # calculate perplexity
        total_perplexity = 0.0
        total_sentence = 0
        for line in get_lines('val.txt'):
            total_sentence += 1
            valid_label_x, valid_vector_x, valid_y = convert_data([line], vocabulary, N)
            # predict
            hidden_input_batch = valid_vector_x.dot(embed_to_hid_weights) + embed_to_hid_bias  # (batch, 128)
            output_batch = hidden_input_batch.dot(hid_to_output_weights) + hid_to_output_bias
            output_batch = utils.softmax(output_batch)
            probs = output_batch[range(len(valid_y)), valid_y]
            probs = -np.log(probs)
            perplexity = 2.718281828 ** np.mean(probs)
            total_perplexity += perplexity

        print 'epoch: {0}, perplexity: {1}'.format(epoch, total_perplexity / total_sentence)
