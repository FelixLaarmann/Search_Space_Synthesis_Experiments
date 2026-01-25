import os 

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torchdiffeq import odeint

from synthesis.utils import generate_data

class OscilatorODE(nn.Module):
    def __init__(self):
        super().__init__()

        self.left = nn.Linear(1, 1, bias=True)
        self.right = nn.Linear(1, 1, bias=True)
        self.split = nn.Linear(1, 1, bias=True)
        self.activation = nn.Tanh()

        self.left.weight.data.fill_(1.0)
        self.left.bias.data.fill_(0.0)

        self.right.weight.data.fill_(-1.0)
        self.right.bias.data.fill_(0.0)

        self.split.weight.data.fill_(1.0)
        self.split.bias.data.fill_(0.0)


    def forward(self, x):
        x = torch.sin(x)
        gate = (self.split(x) <= 0).float()
        

        left = self.left(x) * gate
        right = self.right(x) * (1 - gate)


        return left + right
        

# ===================
# Data Generation 
# ===================

generation_model = OscilatorODE()


x, y = generate_data(generation_model, xmin=-10, xmax=10, n_samples=1_000, eps=1e-5)
x_test, y_test = generate_data(generation_model, xmin=-15, xmax=15, n_samples=1_000, eps=1e-5)

print(f'Test data: x={x_test.shape} - y={y_test.shape}')

########## Save data 
torch.save({
    'x_train': x, 
    'y_train': y, 
    'x_test': x_test,
    'y_test': y_test,
    'meta_data':{
        'generation_model': 'OscilatorODE'
    }
}, 'data/ode3_dataset.pth')


########## Save model 
torch.save(generation_model.state_dict(), 'models/ode_v3.pth')

################### Plot
plt.figure(figsize=(12, 8))
plt.plot(x_test.view(-1), y_test.detach().numpy())
plt.savefig('plots/ode_v3.png')
plt.savefig('plots/ode_v3.pdf')