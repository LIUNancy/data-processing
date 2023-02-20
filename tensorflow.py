import pdb

import numpy as np

import torch
import torch.utils.data as data
from torch import nn


k = 2.25
b = -5.5

X = np.arange(1000)/1000
Y = k*X + b + np.random.rand()/100

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)
dataset_tensor = torch.cat([torch.unsqueeze(X_tensor,1),torch.unsqueeze(Y_tensor,1)], dim=1)

dataset = data.TensorDataset(dataset_tensor)
loader = data.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.linear = nn.Linear(1, 1, bias=True)

    def forward(self, x):
        return self.linear(x)

model = NeuralNetwork()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

loss_mse = nn.MSELoss()

for epoch in range(100):

    for data_ in loader:
        data = data_[0]

        x = data[:,0]
        y = data[:,1]
        x = torch.unsqueeze(x,1)
        y = torch.unsqueeze(y,1)

        y_hat = model(x)

        loss = loss_mse(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss)

pdb.set_trace()
print(model(torch.tensor([0.7,])))
