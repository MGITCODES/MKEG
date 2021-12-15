import torch
import torch.nn as nn
import numpy as np
import random as rd


class WordLstm(nn.Module):
    def __init__(self,vocab_size,embed_size=512,hidden_size=512):
        super(WordLstm, self).__init__()

        self.embed = nn.Embedding(vocab_size,embed_size)

        self.lstm = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.lin = nn.Linear(hidden_size,vocab_size)

    def forward(self,topic,word):
        embedding = self.embed(word)
        embedding = torch.cat((topic, embedding), 1)
        hidden,_ = self.lstm(embedding)
        output = self.lin(hidden[:,-1,:])

        return output


    def sample(self,features, start_tokens):
        sampled_ids = np.zeros((np.shape(features)[0], 50))
        sampled_ids[:, 0] = start_tokens.view(-1, )
        inputword1 = start_tokens
        embeddings = features
        embeddings = embeddings

        for i in range(1, 20):
            predicted = self.embed(inputword1)
            iii = torch.cat((embeddings, predicted), dim=1)
            hidden_states, _ = self.lstm(iii)
            hidden_states = hidden_states[:, -1, :]
            outputs = self.lin(hidden_states)

            # predicted = torch.max(outputs, 1)[1]
            # sampled_ids[:, i] = predicted
            # predicted = predicted.unsqueeze(1)

            _,a = torch.topk(outputs,k=10,dim=1)
            predicted = torch.zeros(a.size(0),dtype=int).cuda()
            for j in range(0,a.size(0)):
                predicted[j] = a[j][rd.randint(0,0)]
            sampled_ids[:, i] = predicted
            predicted = predicted.unsqueeze(1)
            # inputword1 = predicted
            inputword1 = torch.cat([inputword1, predicted], dim=1)
        return sampled_ids

class WordLSTM(nn.Module):
    def __init__(self,
                 embed_size,
                 hidden_size,
                 vocab_size,
                 num_layers,
                 n_max=50):
        super(WordLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.__init_weights()
        self.n_max = n_max
        self.vocab_size = vocab_size

    def __init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, topic_vec, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((topic_vec, embeddings), 1)
        hidden, _ = self.lstm(embeddings)
        outputs = self.linear(hidden[:, -1, :])
        return outputs



    def sample(self, features, start_tokens):
        sampled_ids = np.zeros((np.shape(features)[0], self.n_max))
        sampled_ids[:, 0] = start_tokens.view(-1, )
        predicted = start_tokens
        embeddings = features
        embeddings = embeddings

        for i in range(1, self.n_max):
            predicted = self.embed(predicted)
            embeddings = torch.cat([embeddings, predicted], dim=1)
            hidden_states, _ = self.lstm(embeddings)
            hidden_states = hidden_states[:, -1, :]
            outputs = self.linear(hidden_states)

            # predicted = torch.max(outputs, 1)[1]
            # sampled_ids[:, i] = predicted
            # predicted = predicted.unsqueeze(1)

            _,a = torch.topk(outputs,k=10,dim=1)
            predicted = torch.zeros(a.size(0),dtype=int).cuda()
            for j in range(0,a.size(0)-1):
                predicted[j] = a[j][rd.randint(0,0)]
            sampled_ids[:, i] = predicted
            predicted = predicted.unsqueeze(1)

        return sampled_ids


