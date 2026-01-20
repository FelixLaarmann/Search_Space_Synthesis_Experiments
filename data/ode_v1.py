import os 

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torchdiffeq import odeint

from synthesis.utils import generate_data

class SimpleODE(nn.Module):
    def __init__(self):
        super().__init__()

        self.res = nn.Linear(1, 1, bias=True)
        self.path = nn.Linear(1, 1, bias=True)
        self.activation = nn.Tanh()

        self.res.weight.data.fill_(5.0)
        self.res.bias.data.fill_(5.0)

        self.path.weight.data.fill_(0.5)
        self.path.bias.data.fill_(1.0)


    def forward(self, x):

        return -1 * (self.res(x) * self.activation(self.path(x)))
        

# ===================
# Data Generation 
# ===================

generation_model = SimpleODE()


x, y = generate_data(generation_model, xmin=-10, xmax=10, n_samples=1_000, eps=1e-4)
x_test, y_test = generate_data(generation_model, xmin=-15, xmax=15, n_samples=1_000, eps=1e-4)

print(f'Test data: x={x_test.shape} - y={y_test.shape}')

########## Save data 
torch.save({
    'x_train': x, 
    'y_train': y, 
    'x_test': x_test,
    'y_test': y_test,
    'meta_data':{
        'generation_model': 'SimpleODE'
    }
}, 'data/ode1_dataset.pth')


########## Save model 
torch.save(generation_model.state_dict(), 'models/ode_v1.pth')

################### Plot
plt.figure(figsize=(12, 8))
plt.plot(x_test.view(-1), y_test.detach().numpy())
plt.savefig('plots/ode_v1.png')
plt.savefig('plots/ode_v1.pdf')