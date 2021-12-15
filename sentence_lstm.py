import torch
import torch.nn as nn

class SentenceLstm(nn.Module):
    def __init__(self, visual_size=512, hidden_size=512,embed_size=512,layer_num=1):
        super(SentenceLstm, self).__init__()
        self.W_v = nn.Linear(in_features=visual_size,out_features=visual_size)
        self.W_h = nn.Linear(in_features=hidden_size,out_features=hidden_size)
        self.W_a = nn.Linear(in_features=visual_size,out_features=visual_size)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.lstm = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=layer_num,batch_first=True, dropout=0.2)

        self.W_S_s_1 = nn.Linear(in_features=hidden_size,out_features=hidden_size)
        self.W_S_s = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.W_S = nn.Linear(in_features=hidden_size,out_features=2)
        self.sigmoid = nn.Sigmoid()

        # 用于线性组合vt和st
        self.W_v = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.W_s = nn.Linear(in_features=visual_size, out_features=visual_size)

    def forward(self,visual,hidden_state,sentence_state):

        # 注意力
        a = self.softmax(self.W_a(self.tanh(torch.add(self.W_v(visual),self.W_h(hidden_state)))))
        vt = torch.mul(a,visual).sum(1).unsqueeze(1)

        output, hidden = self.lstm(vt,sentence_state)

        stop = self.sigmoid(self.W_S(self.tanh(self.W_S_s_1(hidden_state)+self.W_S_s(output))))

        word_input = self.W_v(vt) + self.W_s(output)





        # word_input = torch.add(self.W_v(visual),self.W_h(hidden_state))

        # hidden即st，同时用于更新下个h
        return vt, output, hidden, stop, word_input

