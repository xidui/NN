import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import ngram
import utils
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


lr = 0.05
embedding_dim = 16
HIDDEN = 128
batch = 16


class GRU_NGRAM(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, hidden_size):
        super(GRU_NGRAM, self).__init__()
        self.encoder = nn.Embedding(vocabulary_size, embedding_dim)
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1)
        self.decoder = nn.Linear(hidden_size, vocabulary_size)
        self.init_weights()
        self.nhid = hidden_size
        self.nlayers = 1

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        embeds = self.encoder(input)
        embeds = embeds.permute(1, 0, 2)
        rnn_output, hidden = self.rnn(embeds, hidden)
        output = self.decoder(rnn_output[-1].view(-1, self.nhid))
        output = F.log_softmax(output)
        return output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


tmp = ngram.NGram(n=4, embedding_dimension=embedding_dim, vocabulary_size=8000, hidden=128)
tmp.vocabulary = tmp.create_vocabulary('train.txt')
l = utils.get_lines('train.txt')
train_label_x, train_y = tmp.get_label_from_words(l)


def train(epochs=15, truncate=False):
    model = GRU_NGRAM(8000, embedding_dim, 128)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5, weight_decay=0)

    train_loss = [0.0] * epochs
    valid_loss = [0.0] * epochs
    perplexity_list = [0.0] * epochs

    for epoch in range(epochs):
        model.train()
        start = 0
        while start < len(train_label_x):
            optimizer.zero_grad()
            train_label_x_batch = train_label_x[start:start + batch]
            train_label_y_batch = train_y[start:start + batch]

            input = Variable(torch.LongTensor(train_label_x_batch.tolist()))
            target = Variable(torch.LongTensor(train_label_y_batch.tolist()))

            hidden = model.init_hidden(min(batch, len(train_label_x_batch)))
            if truncate:
                input = input[:,2:]
            output, hidden = model(input, hidden)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss[epoch] += loss.data[0] * len(train_label_y_batch)

            start += batch

        model.eval()
        total_sentence = 0
        valid_size = 0
        for line in utils.get_lines('val.txt'):
            valid_label_x, valid_y = tmp.get_label_from_words([line])
            total_sentence += 1
            valid_size += len(valid_y)

            start = 0
            sentense_probs = 0.0
            while start < len(valid_label_x):
                valid_label_x_batch = valid_label_x[start:start + batch]
                valid_label_y_batch = valid_y[start:start + batch]

                input = Variable(torch.LongTensor(valid_label_x_batch.tolist()))
                target = Variable(torch.LongTensor(valid_label_y_batch.tolist()))

                hidden = model.init_hidden(min(batch, len(valid_label_x_batch)))
                output, hidden = model(input, hidden)
                loss = criterion(output, target)
                valid_loss[epoch] += loss.data[0] * len(valid_label_y_batch)

                start += batch

                output = output.data.numpy()
                target = target.data.numpy()
                probs = output[range(len(target)), target]
                sentense_probs -= np.sum(probs)

            perplexity = 2 ** (sentense_probs/len(valid_y))
            perplexity_list[epoch] += perplexity

        train_loss[epoch] /= len(train_label_x)
        valid_loss[epoch] /= valid_size
        perplexity_list[epoch] /= total_sentence

        print 'epoch:{0}, train loss:{1}, valid loss:{2}, perpleixty:{3}'.format(
            epoch,
            train_loss[epoch],
            valid_loss[epoch],
            perplexity_list[epoch]
        )

    plt.figure(1)
    plt.plot(range(1, len(perplexity_list) + 1), perplexity_list)
    plt.ylabel('perplexity')
    plt.xlabel('epoches')
    plt.title('GRU / embedding:{0}'.format(embedding_dim))
    plt.savefig('n_gram_pictures/gru_perplexity_embedding_{0}.png'.format(embedding_dim))
    plt.close()

    plt.figure(2)
    plt.plot(range(1, len(perplexity_list) + 1), train_loss, color='blue')
    plt.plot(range(1, len(perplexity_list) + 1), valid_loss, color='green')
    plt.ylabel('average loss')
    plt.legend(['train', 'valid'])
    plt.title('GRU loss / embedding:{0}'.format(embedding_dim))
    plt.savefig('n_gram_pictures/gru_loss_embedding_{0}.png'.format(embedding_dim))
    plt.close()

# problem 3.6.1, 3.6.2
# for embedding_dim in [16, 32, 64, 128]:
#     train()

# problem 3.6.3
embedding_dim = 16
train_label_x = train_label_x[:len(train_label_x) // 10]
train_y = train_y[:len(train_y) // 10]
train(truncate=True, epochs=12)
