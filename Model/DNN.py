import torch
import torch.nn as nn
import math

import csv
import numpy as np
with open('EURUSDD1.csv', 'r') as f:
  reader = csv.reader(f)
  data = []
  last_close = 0
  div = 0
  for row in reader:
    for i in range(4):
      row[i] = float(row[i])-0.7
    data.append(row)

target = []
train = []
for i in range(5, len(data)):
  tmp = []
  target.append((data[i][3]))
  for j in range(5):
    for k in range(4):
      tmp.append(data[i+j-5][k])
  train.append(tmp)

train = torch.tensor(np.array(train),dtype=torch.float32)
target = torch.tensor(np.array(target),dtype=torch.float32)
print(train.shape)

myNet = nn.Sequential(
    nn.Linear(20, 200),
    nn.ReLU(),
    nn.Linear(200, 200),
    nn.ReLU(),
    nn.Linear(200, 200),
    nn.ReLU(),
    nn.Linear(200, 200),
    nn.ReLU(),
    nn.Linear(200, 200),
    nn.ReLU(),
    nn.Linear(200, 200),
    nn.ReLU(),
    nn.Linear(200, 1),
)

import matplotlib.pyplot as plt
def show(predict, target):
  plt.plot(predict, "r", label="predict")
  plt.plot(target, "b", label="target")
  plt.xlabel('t', color='#1C2833')
  plt.ylabel('price', color='#1C2833')
  plt.legend(loc='best')
  plt.grid()
  plt.show()

optimizer = torch.optim.Adam(myNet.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()
for epoch in range(1500):
  print(epoch)
  optimizer.zero_grad()
  sum_loss = 0
  i = 0
  while (1):
      i = i + 1
      if (i >= 1800):
          break
      out = myNet(train[i])
      # print(out)
      loss = loss_func(out, target[i])
      sum_loss = sum_loss + loss.detach()
      loss.backward()
  print(sum_loss / 1800)
  optimizer.step()