
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NaiveLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(NaiveLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size, 1))

        self.w_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_if = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.b_hf = nn.Parameter(torch.Tensor(hidden_size, 1))

        self.w_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_io = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size, 1))

        self.w_ig = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.reset_weights()

    def reset_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, stdv, stdv)

    def forward(self, inputs, state):

        (i, f, g, o, h, c) = state
        h_t = h.t()
        c_t = c.t()

        seq_size = 1
        for t in range(seq_size):
            x = inputs.t()
            # print(x.shape, self.w_ii.shape, h_t.shape)
            i = torch.sigmoid(self.w_ii @ x + self.b_ii + self.w_hi @ h_t + self.b_hi)
            f = torch.sigmoid(self.w_if @ x + self.b_if + self.w_hf @ h_t + self.b_hf)
            g = torch.tanh(self.w_ig @ x + self.b_ig + self.w_hg @ h_t + self.b_hg)
            # g = torch.tanh(self.gatei(x.t()) + self.gateh(h_t.t())).t()
            o = torch.sigmoid(self.w_io @ x + self.b_io + self.w_ho @ h_t + self.b_ho)
            c_next_t = f * c_t + i * g
            h_next_t = o * torch.tanh(c_next_t)
            c_next = c_next_t.t()
            h_next = h_next_t.t()
            # print(h_next, c_next)
        return (h_next, c_next)


class DeepLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, lstm_size: int):
        super(DeepLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.LSTM = NaiveLSTM(4, lstm_size)
        self.Out = nn.Sequential(
            nn.Linear(lstm_size, 1, bias=True),
            nn.ReLU()
        )
        self.reset_weights()

    def reset_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs, seq_size, state):
        (i, f, g, o, x, c, h) = state
        for t in range(seq_size):
            x[t] = inputs[t].unsqueeze(0)
            h[t + 1], c[t + 1] = self.LSTM.forward(x[t], (i[t], f[t], g[t], o[t], h[t], c[t]))
        return self.Out(h[seq_size])


import csv
import numpy as np

with open('EURUSDD1.csv', 'r') as f:
    reader = csv.reader(f)
    data = []
    last_close = 0
    div = 0
    for row in reader:
        for i in range(4):
            row[i] = float(row[i]) - 0.7
        data.append(row)

target = []
train = []
for i in range(5, len(data)):
    tmp = []
    target.append(data[i][3])
    for j in range(5):
        tmp.append(data[i + j - 5])
    train.append(tmp)

train = torch.tensor(np.array(train), dtype=torch.float32)
target = torch.tensor(np.array(target), dtype=torch.float32)
print(train.shape)


hidden_size = 200
i = [0] * 20
f = [0] * 20
g = [0] * 20
o = [0] * 20
x = [0] * 20
c = [0] * 20
h = [0] * 20
c[0] = torch.zeros(1, hidden_size, requires_grad=True)
h[0] = torch.zeros(1, hidden_size, requires_grad=True)
state = (i, f, g, o, x, c, h)

Model = DeepLSTM(4, hidden_size, hidden_size)

#optimizer.load_state_dict(checkpoint['optimizer'])

optimizer = torch.optim.Adam(Model.parameters(), lr=0.001)
for epoch in range(1500):
  i = 0
  cnt = 0
  sum_loss = 0
  while(1):
    i = i + 10
    if(i >= 2000):
      break
    cnt = cnt + 1
    out = Model.forward(train[i], 5, state)
    mse = (out-target[i])**2
    loss = mse
    loss.backward()
    sum_loss = sum_loss + mse.detach()
    #print(Model.LSTM.w_ig.grad)
    #print(Model.LSTM.w_ii.grad)
  print(sum_loss/cnt)
  optimizer.step()