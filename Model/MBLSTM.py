import torch
import torch.nn as nn
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
            nn.init.uniform_(weight, 0.2 - stdv, 0.2 + stdv)

    def forward(self, inputs, state):

        if state is None:
            h_t = torch.zeros(1, self.hidden_size).t()
            c_t = torch.zeros(1, self.hidden_size).t()
        else:
            (h, c) = state
            h_t = h.t()
            c_t = c.t()

        seq_size = 1
        for t in range(seq_size):
            x = inputs

            i = torch.sigmoid(self.w_ii @ x + self.b_ii + self.w_hi @ h_t + self.b_hi)
            f = torch.sigmoid(self.w_if @ x + self.b_if + self.w_hf @ h_t + self.b_hf)
            g = torch.tanh(self.w_ig @ x + self.b_ig + self.w_hg @ h_t + self.b_hg)
            o = torch.sigmoid(self.w_io @ x + self.b_io + self.w_ho @ h_t + self.b_ho)

            c_next_t = f * c_t + i * g
            h_next_t = o * torch.tanh(c_next_t)
            c_next = c_next_t.t()
            h_next = h_next_t.t()

        return (h_next, c_next)


def lstm_step(x, h, c, w_ii, b_ii, w_hi, b_hi,
              w_if, b_if, w_hf, b_hf,
              w_ig, b_ig, w_hg, b_hg,
              w_io, b_io, w_ho, b_ho):
    x_t = x.t()
    h_t = h.t()
    c_t = c.t()
    i = torch.sigmoid(w_ii @ x_t + b_ii + w_hi @ h_t + b_hi)
    o = torch.sigmoid(w_io @ x_t + b_io + w_ho @ h_t + b_ho)
    g = torch.tanh(w_ig @ x_t + b_ig + w_hg @ h_t + b_hg)
    f = torch.sigmoid(w_if @ x_t + b_if + w_hf @ h_t + b_hf)

    c_next = f * c_t + i * g

    h_next = o * torch.tanh(c_next)
    c_next_t = c_next.t()
    h_next_t = h_next.t()

    return h_next_t, c_next_t


import csv
import numpy as np

with open('EURUSDD1.csv', 'r') as f:
    reader = csv.reader(f)
    data = []
    for row in reader:
        for i in range(5):
            row[i] = float(row[i]) / 10
            if i == 4:
                row[i] = row[i] / 10000
        data.append(row)
target = []
train = []
for i in range(5, len(data)):
    tmp = []
    target.append(data[i][1])
    for j in range(5):
        tmp.append(data[i + j - 5])
    train.append(tmp)

train = torch.tensor(np.array(train), dtype=torch.float32)
target = torch.tensor(np.array(target), dtype=torch.float32)
print(train.shape)
print(target.shape)


class MultiBiLSTM(nn.Module):
    def __init__(self):
        super(MultiBiLSTM, self).__init__()
        self.LSTMf1 = NaiveLSTM(5, 20)
        self.LSTMb1 = NaiveLSTM(5, 20)
        self.LSTMf2 = NaiveLSTM(20, 20)
        self.LSTMb2 = NaiveLSTM(20, 20)
        self.LSTMf3 = NaiveLSTM(20, 10)
        self.LSTMb3 = NaiveLSTM(20, 10)
        self.LSTMl = NaiveLSTM(10, 1)

    def forward(self, inputs, state):
        Hf1 = []
        Hf2 = []
        Hf3 = []
        Hb1 = []
        Hb2 = []
        Hb3 = []
        Hl = 0
        (hf1, cf1, hb1, cb1, hf2, cf2, hb2, cb2, hf3, cf3, hb3, cb3, hl, cl) = state
        seq_size = 5
        for t in range(seq_size):
            xf = inputs[:, t, :].t()
            xb = inputs[:, seq_size - t - 1, :].t()
            (hf1, cf1) = self.LSTMf1.forward(xf, (hf1, cf1))
            Hf1.append(hf1)
            (hb1, cb1) = self.LSTMb1.forward(xb, (hb1, cb1))
            Hb1.append(hb1)
        for t in range(seq_size):
            xf = Hf1[t].t()
            xb = Hb1[t].t()
            (hf2, cf2) = self.LSTMf2.forward(xf, (hf2, cf2))
            Hf2.append(hf2)
            (hb2, cb2) = self.LSTMb2.forward(xb, (hb2, cb2))
            Hb2.append(hb2)
        for t in range(seq_size):
            xf = Hf2[t].t()
            xb = Hb2[t].t()
            (hf3, cf3) = self.LSTMf3.forward(xf, (hf3, cf3))
            Hf3.append(hf3)
            (hb3, cb3) = self.LSTMb3.forward(xb, (hb3, cb3))
            Hb3.append(hb3)
        for t in range(seq_size):
            xl = (Hf3[t] + Hb3[seq_size - t - 1]).t()
            (hl, cl) = self.LSTMl.forward(xl, (hl, cl))
            Hl = hl
        return Hl


def MultiBiLSTM_step(sequence_len, test_embeddings,
                     LSTMf1, LSTMb1, LSTMf2, LSTMb2, LSTMf3, LSTMb3, LSTMl,
                     hf1, cf1, hb1, cb1,
                     hf2, cf2, hb2, cb2,
                     hf3, cf3, hb3, cb3,
                     hl, cl):
    for t in range(sequence_len):
        hf1[t + 1], cf1[t + 1] = lstm_step(test_embeddings[:, t, :], hf1[t], cf1[t],
                                           LSTMf1.w_ii, LSTMf1.b_ii, LSTMf1.w_hi, LSTMf1.b_hi,
                                           LSTMf1.w_if, LSTMf1.b_if, LSTMf1.w_hf, LSTMf1.b_hf,
                                           LSTMf1.w_ig, LSTMf1.b_ig, LSTMf1.w_hg, LSTMf1.b_hg,
                                           LSTMf1.w_io, LSTMf1.b_io, LSTMf1.w_ho, LSTMf1.b_ho)
        hb1[t + 1], cb1[t + 1] = lstm_step(test_embeddings[:, sequence_len - t - 1, :], hb1[t], cb1[t],
                                           LSTMb1.w_ii, LSTMb1.b_ii, LSTMb1.w_hi, LSTMb1.b_hi,
                                           LSTMb1.w_if, LSTMb1.b_if, LSTMb1.w_hf, LSTMb1.b_hf,
                                           LSTMb1.w_ig, LSTMb1.b_ig, LSTMb1.w_hg, LSTMb1.b_hg,
                                           LSTMb1.w_io, LSTMb1.b_io, LSTMb1.w_ho, LSTMb1.b_ho)
    for t in range(sequence_len):
        hf2[t + 1], cf2[t + 1] = lstm_step(hf1[t + 1], hf2[t], cf2[t],
                                           LSTMf2.w_ii, LSTMf2.b_ii, LSTMf2.w_hi, LSTMf2.b_hi,
                                           LSTMf2.w_if, LSTMf2.b_if, LSTMf2.w_hf, LSTMf2.b_hf,
                                           LSTMf2.w_ig, LSTMf2.b_ig, LSTMf2.w_hg, LSTMf2.b_hg,
                                           LSTMf2.w_io, LSTMf2.b_io, LSTMf2.w_ho, LSTMf2.b_ho)

        hb2[t + 1], cb2[t + 1] = lstm_step(hb1[t + 1], hb2[t], cb2[t],
                                           LSTMb2.w_ii, LSTMb2.b_ii, LSTMb2.w_hi, LSTMb2.b_hi,
                                           LSTMb2.w_if, LSTMb2.b_if, LSTMb2.w_hf, LSTMb2.b_hf,
                                           LSTMb2.w_ig, LSTMb2.b_ig, LSTMb2.w_hg, LSTMb2.b_hg,
                                           LSTMb2.w_io, LSTMb2.b_io, LSTMb2.w_ho, LSTMb2.b_ho)

    for t in range(sequence_len):
        hf3[t + 1], cf3[t + 1] = lstm_step(hf2[t + 1], hf3[t], cf3[t],
                                           LSTMf3.w_ii, LSTMf3.b_ii, LSTMf3.w_hi, LSTMf3.b_hi,
                                           LSTMf3.w_if, LSTMf3.b_if, LSTMf3.w_hf, LSTMf3.b_hf,
                                           LSTMf3.w_ig, LSTMf3.b_ig, LSTMf3.w_hg, LSTMf3.b_hg,
                                           LSTMf3.w_io, LSTMf3.b_io, LSTMf3.w_ho, LSTMf3.b_ho)

        hb3[t + 1], cb3[t + 1] = lstm_step(hb2[t + 1], hb3[t], cb3[t],
                                           LSTMb3.w_ii, LSTMb3.b_ii, LSTMb3.w_hi, LSTMb3.b_hi,
                                           LSTMb3.w_if, LSTMb3.b_if, LSTMb3.w_hf, LSTMb3.b_hf,
                                           LSTMb3.w_ig, LSTMb3.b_ig, LSTMb3.w_hg, LSTMb3.b_hg,
                                           LSTMb3.w_io, LSTMb3.b_io, LSTMb3.w_ho, LSTMb3.b_ho)

    for t in range(sequence_len):
        hl[t + 1], cl[t + 1] = lstm_step(hf3[t + 1] + hb3[sequence_len - t], hl[t], cl[t],
                                         LSTMl.w_ii, LSTMl.b_ii, LSTMl.w_hi, LSTMl.b_hi,
                                         LSTMl.w_if, LSTMl.b_if, LSTMl.w_hf, LSTMl.b_hf,
                                         LSTMl.w_ig, LSTMl.b_ig, LSTMl.w_hg, LSTMl.b_hg,
                                         LSTMl.w_io, LSTMl.b_io, LSTMl.w_ho, LSTMl.b_ho)
    return hl[sequence_len]


hf1 = [0] * 10
hf2 = [0] * 10
hf3 = [0] * 10
hb1 = [0] * 10
hb2 = [0] * 10
hb3 = [0] * 10
hl = [0] * 10
hf1[0] = torch.zeros(1, 20, requires_grad=True)
hb1[0] = torch.zeros(1, 20, requires_grad=True)
hf2[0] = torch.zeros(1, 20, requires_grad=True)
hb2[0] = torch.zeros(1, 20, requires_grad=True)
hf3[0] = torch.zeros(1, 10, requires_grad=True)
hb3[0] = torch.zeros(1, 10, requires_grad=True)
hl[0] = torch.zeros(1, 1, requires_grad=True)
cf1 = [0] * 10
cf2 = [0] * 10
cf3 = [0] * 10
cb1 = [0] * 10
cb2 = [0] * 10
cb3 = [0] * 10
cl = [0] * 10
cf1[0] = torch.zeros(1, 20, requires_grad=True)
cb1[0] = torch.zeros(1, 20, requires_grad=True)
cf2[0] = torch.zeros(1, 20, requires_grad=True)
cb2[0] = torch.zeros(1, 20, requires_grad=True)
cf3[0] = torch.zeros(1, 10, requires_grad=True)
cb3[0] = torch.zeros(1, 10, requires_grad=True)
cl[0] = torch.zeros(1, 1, requires_grad=True)

import matplotlib.pyplot as plt


def show(predict, target, close):
    plt.plot(predict, "r", label="predict")
    plt.plot(target, "b", label="target")
    plt.plot(close, "g", label="close")
    plt.xlabel('t', color='#1C2833')
    plt.ylabel('price', color='#1C2833')
    plt.ylabel('close', color='#1C2833')
    plt.legend(loc='best')
    plt.grid()
    plt.show()


Model = MultiBiLSTM()

import random
from math import e

print(e)

optimizer = torch.optim.Adam(Model.parameters(), lr=0.00001)

for epoch in range(10000000):
    predict = []
    plot_target = []
    close = []
    if epoch % 100 == 0:
        j = 0
        i = 0
        while (1):
            i = i + 10
            if (i == 2000):
                break
            predict.append(Model.forward(train[i + j].unsqueeze(0), (hf1[0], cf1[0], hb1[0], cb1[0],
                                                                     hf2[0], cf2[0], hb2[0], cb2[0],
                                                                     hf3[0], cf3[0], hb3[0], cb3[0],
                                                                     hl[0], cl[0])))
            t = target[i + j].detach()
            plot_target.append(target[i + j])
            close.append(train[i + j][4][3])
        show(predict, plot_target, close)
    j = 0
    optimizer.zero_grad()
    sum_loss = 0
    i = 0
    while (1):
        i = i + 1
        if (i >= 2000):
            break
        res = MultiBiLSTM_step(5, train[i + j].unsqueeze(0),
                               Model.LSTMf1, Model.LSTMb1,
                               Model.LSTMf2, Model.LSTMb2,
                               Model.LSTMf3, Model.LSTMb3,
                               Model.LSTMl,
                               hf1, cf1, hb1, cb1,
                               hf2, cf2, hb2, cb2,
                               hf3, cf3, hb3, cb3,
                               hl, cl)
        loss = 1000000000 * (
                    res - target[i + j]) ** 2  # +e**(-20000000*(res-train[i+j][4][3])*(target[i+j]-train[i+j][4][3]))
        loss.backward()
        sum_loss = sum_loss + loss.detach()
        # print(Model.LSTMl.w_hi.grad)
    optimizer.step()
    print(sum_loss)


# ----------------------------------save----------------------------------------------------------------------
def convert(param):
    result = []
    for vec in param:
        tmp = []
        for num in vec:
            tmp.append(str(num))
        result.append(','.join(tmp))
    return result


def save_param(f, LSTM):
    param = convert(np.array(LSTM.w_ii.detach()).tolist())
    f.writelines(';'.join(param) + '\n')
    param = convert(np.array(LSTM.w_hi.detach()).tolist())
    f.writelines(';'.join(param) + '\n')
    param = convert(np.array(LSTM.b_ii.detach()).tolist())
    f.writelines(';'.join(param) + '\n')
    param = convert(np.array(LSTM.b_hi.detach()).tolist())
    f.writelines(';'.join(param) + '\n')

    param = convert(np.array(LSTM.w_if.detach()).tolist())
    f.writelines(';'.join(param) + '\n')
    param = convert(np.array(LSTM.w_hf.detach()).tolist())
    f.writelines(';'.join(param) + '\n')
    param = convert(np.array(LSTM.b_if.detach()).tolist())
    f.writelines(';'.join(param) + '\n')
    param = convert(np.array(LSTM.b_hf.detach()).tolist())
    f.writelines(';'.join(param) + '\n')

    param = convert(np.array(LSTM.w_io.detach()).tolist())
    f.writelines(';'.join(param) + '\n')
    param = convert(np.array(LSTM.w_ho.detach()).tolist())
    f.writelines(';'.join(param) + '\n')
    param = convert(np.array(LSTM.b_io.detach()).tolist())
    f.writelines(';'.join(param) + '\n')
    param = convert(np.array(LSTM.b_ho.detach()).tolist())
    f.writelines(';'.join(param) + '\n')

    param = convert(np.array(LSTM.w_ig.detach()).tolist())
    f.writelines(';'.join(param) + '\n')
    param = convert(np.array(LSTM.w_hg.detach()).tolist())
    f.writelines(';'.join(param) + '\n')
    param = convert(np.array(LSTM.b_ig.detach()).tolist())
    f.writelines(';'.join(param) + '\n')
    param = convert(np.array(LSTM.b_hg.detach()).tolist())
    f.writelines(';'.join(param) + '\n')


f = open('param7.1.txt', 'a')
save_param(f, Model.LSTMf1)
save_param(f, Model.LSTMb1)
save_param(f, Model.LSTMf2)
save_param(f, Model.LSTMb2)
save_param(f, Model.LSTMf3)
save_param(f, Model.LSTMb3)
save_param(f, Model.LSTMl)
# save_param(f, Model.LSTMl2)
# save_param(f, Model.LSTMl3)
f.close()


# --------------------------------------------------read-----------------------------------------------------------
def convert2(string):
    param = string.split(';')
    result = []
    for line in param:
        tmp = []
        vec = line.split(',')
        for value in vec:
            tmp.append(float(value))
        result.append(tmp)
    return torch.tensor(np.array(result), dtype=torch.float32)


f = open('/content/param7.1.txt', 'r')
Model.LSTMf1.w_ii.data = convert2(f.readline())
Model.LSTMf1.w_hi.data = convert2(f.readline())
Model.LSTMf1.b_ii.data = convert2(f.readline())
Model.LSTMf1.b_hi.data = convert2(f.readline())

Model.LSTMf1.w_if.data = convert2(f.readline())
Model.LSTMf1.w_hf.data = convert2(f.readline())
Model.LSTMf1.b_if.data = convert2(f.readline())
Model.LSTMf1.b_hf.data = convert2(f.readline())

Model.LSTMf1.w_io.data = convert2(f.readline())
Model.LSTMf1.w_ho.data = convert2(f.readline())
Model.LSTMf1.b_io.data = convert2(f.readline())
Model.LSTMf1.b_ho.data = convert2(f.readline())

Model.LSTMf1.w_ig.data = convert2(f.readline())
Model.LSTMf1.w_hg.data = convert2(f.readline())
Model.LSTMf1.b_ig.data = convert2(f.readline())
Model.LSTMf1.b_hg.data = convert2(f.readline())
# -------------------------------------------
Model.LSTMb1.w_ii.data = convert2(f.readline())
Model.LSTMb1.w_hi.data = convert2(f.readline())
Model.LSTMb1.b_ii.data = convert2(f.readline())
Model.LSTMb1.b_hi.data = convert2(f.readline())

Model.LSTMb1.w_if.data = convert2(f.readline())
Model.LSTMb1.w_hf.data = convert2(f.readline())
Model.LSTMb1.b_if.data = convert2(f.readline())
Model.LSTMb1.b_hf.data = convert2(f.readline())

Model.LSTMb1.w_io.data = convert2(f.readline())
Model.LSTMb1.w_ho.data = convert2(f.readline())
Model.LSTMb1.b_io.data = convert2(f.readline())
Model.LSTMb1.b_ho.data = convert2(f.readline())

Model.LSTMb1.w_ig.data = convert2(f.readline())
Model.LSTMb1.w_hg.data = convert2(f.readline())
Model.LSTMb1.b_ig.data = convert2(f.readline())
Model.LSTMb1.b_hg.data = convert2(f.readline())
# -------------------------------------------
Model.LSTMf2.w_ii.data = convert2(f.readline())
Model.LSTMf2.w_hi.data = convert2(f.readline())
Model.LSTMf2.b_ii.data = convert2(f.readline())
Model.LSTMf2.b_hi.data = convert2(f.readline())

Model.LSTMf2.w_if.data = convert2(f.readline())
Model.LSTMf2.w_hf.data = convert2(f.readline())
Model.LSTMf2.b_if.data = convert2(f.readline())
Model.LSTMf2.b_hf.data = convert2(f.readline())

Model.LSTMf2.w_io.data = convert2(f.readline())
Model.LSTMf2.w_ho.data = convert2(f.readline())
Model.LSTMf2.b_io.data = convert2(f.readline())
Model.LSTMf2.b_ho.data = convert2(f.readline())

Model.LSTMf2.w_ig.data = convert2(f.readline())
Model.LSTMf2.w_hg.data = convert2(f.readline())
Model.LSTMf2.b_ig.data = convert2(f.readline())
Model.LSTMf2.b_hg.data = convert2(f.readline())
# --------------------------------------------
Model.LSTMb2.w_ii.data = convert2(f.readline())
Model.LSTMb2.w_hi.data = convert2(f.readline())
Model.LSTMb2.b_ii.data = convert2(f.readline())
Model.LSTMb2.b_hi.data = convert2(f.readline())

Model.LSTMb2.w_if.data = convert2(f.readline())
Model.LSTMb2.w_hf.data = convert2(f.readline())
Model.LSTMb2.b_if.data = convert2(f.readline())
Model.LSTMb2.b_hf.data = convert2(f.readline())

Model.LSTMb2.w_io.data = convert2(f.readline())
Model.LSTMb2.w_ho.data = convert2(f.readline())
Model.LSTMb2.b_io.data = convert2(f.readline())
Model.LSTMb2.b_ho.data = convert2(f.readline())

Model.LSTMb2.w_ig.data = convert2(f.readline())
Model.LSTMb2.w_hg.data = convert2(f.readline())
Model.LSTMb2.b_ig.data = convert2(f.readline())
Model.LSTMb2.b_hg.data = convert2(f.readline())
# -------------------------------------------
Model.LSTMf3.w_ii.data = convert2(f.readline())
Model.LSTMf3.w_hi.data = convert2(f.readline())
Model.LSTMf3.b_ii.data = convert2(f.readline())
Model.LSTMf3.b_hi.data = convert2(f.readline())

Model.LSTMf3.w_if.data = convert2(f.readline())
Model.LSTMf3.w_hf.data = convert2(f.readline())
Model.LSTMf3.b_if.data = convert2(f.readline())
Model.LSTMf3.b_hf.data = convert2(f.readline())

Model.LSTMf3.w_io.data = convert2(f.readline())
Model.LSTMf3.w_ho.data = convert2(f.readline())
Model.LSTMf3.b_io.data = convert2(f.readline())
Model.LSTMf3.b_ho.data = convert2(f.readline())

Model.LSTMf3.w_ig.data = convert2(f.readline())
Model.LSTMf3.w_hg.data = convert2(f.readline())
Model.LSTMf3.b_ig.data = convert2(f.readline())
Model.LSTMf3.b_hg.data = convert2(f.readline())
# -------------------------------------------
Model.LSTMb3.w_ii.data = convert2(f.readline())
Model.LSTMb3.w_hi.data = convert2(f.readline())
Model.LSTMb3.b_ii.data = convert2(f.readline())
Model.LSTMb3.b_hi.data = convert2(f.readline())

Model.LSTMb3.w_if.data = convert2(f.readline())
Model.LSTMb3.w_hf.data = convert2(f.readline())
Model.LSTMb3.b_if.data = convert2(f.readline())
Model.LSTMb3.b_hf.data = convert2(f.readline())

Model.LSTMb3.w_io.data = convert2(f.readline())
Model.LSTMb3.w_ho.data = convert2(f.readline())
Model.LSTMb3.b_io.data = convert2(f.readline())
Model.LSTMb3.b_ho.data = convert2(f.readline())

Model.LSTMb3.w_ig.data = convert2(f.readline())
Model.LSTMb3.w_hg.data = convert2(f.readline())
Model.LSTMb3.b_ig.data = convert2(f.readline())
Model.LSTMb3.b_hg.data = convert2(f.readline())
# -------------------------------------------
Model.LSTMl.w_ii.data = convert2(f.readline())
Model.LSTMl.w_hi.data = convert2(f.readline())
Model.LSTMl.b_ii.data = convert2(f.readline())
Model.LSTMl.b_hi.data = convert2(f.readline())

Model.LSTMl.w_if.data = convert2(f.readline())
Model.LSTMl.w_hf.data = convert2(f.readline())
Model.LSTMl.b_if.data = convert2(f.readline())
Model.LSTMl.b_hf.data = convert2(f.readline())

Model.LSTMl.w_io.data = convert2(f.readline())
Model.LSTMl.w_ho.data = convert2(f.readline())
Model.LSTMl.b_io.data = convert2(f.readline())
Model.LSTMl.b_ho.data = convert2(f.readline())

Model.LSTMl.w_ig.data = convert2(f.readline())
Model.LSTMl.w_hg.data = convert2(f.readline())
Model.LSTMl.b_ig.data = convert2(f.readline())
Model.LSTMl.b_hg.data = convert2(f.readline())
# #-------------------------------------------
# Model.LSTMl2.w_ii.data = convert2(f.readline())
# Model.LSTMl2.w_hi.data = convert2(f.readline())
# Model.LSTMl2.b_ii.data = convert2(f.readline())
# Model.LSTMl2.b_hi.data = convert2(f.readline())

# Model.LSTMl2.w_if.data = convert2(f.readline())
# Model.LSTMl2.w_hf.data = convert2(f.readline())
# Model.LSTMl2.b_if.data = convert2(f.readline())
# Model.LSTMl2.b_hf.data = convert2(f.readline())

# Model.LSTMl2.w_io.data = convert2(f.readline())
# Model.LSTMl2.w_ho.data = convert2(f.readline())
# Model.LSTMl2.b_io.data = convert2(f.readline())
# Model.LSTMl2.b_ho.data = convert2(f.readline())

# Model.LSTMl2.w_ig.data = convert2(f.readline())
# Model.LSTMl2.w_hg.data = convert2(f.readline())
# Model.LSTMl2.b_ig.data = convert2(f.readline())
# Model.LSTMl2.b_hg.data = convert2(f.readline())
# #-------------------------------------------
# Model.LSTMl3.w_ii.data = convert2(f.readline())
# Model.LSTMl3.w_hi.data = convert2(f.readline())
# Model.LSTMl3.b_ii.data = convert2(f.readline())
# Model.LSTMl3.b_hi.data = convert2(f.readline())

# Model.LSTMl3.w_if.data = convert2(f.readline())
# Model.LSTMl3.w_hf.data = convert2(f.readline())
# Model.LSTMl3.b_if.data = convert2(f.readline())
# Model.LSTMl3.b_hf.data = convert2(f.readline())

# Model.LSTMl3.w_io.data = convert2(f.readline())
# Model.LSTMl3.w_ho.data = convert2(f.readline())
# Model.LSTMl3.b_io.data = convert2(f.readline())
# Model.LSTMl3.b_ho.data = convert2(f.readline())

# Model.LSTMl3.w_ig.data = convert2(f.readline())
# Model.LSTMl3.w_hg.data = convert2(f.readline())
# Model.LSTMl3.b_ig.data = convert2(f.readline())
# Model.LSTMl3.b_hg.data = convert2(f.readline())
f.close()
