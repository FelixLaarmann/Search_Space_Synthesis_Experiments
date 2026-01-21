import os 

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from synthesis.utils import generate_data


class TrapezoidNetPure(nn.Module):
    def __init__(self, random_weights=False, sharpness=None):
        super().__init__()

        self.split = nn.Linear(1, 1, bias=True)  
        self.left = nn.Linear(1, 1, bias=True)
        self.right = nn.Linear(1, 1, bias=True)
        self.sharpness = sharpness

        if not random_weights:
            with torch.no_grad():
                # For left branch (x <= 0): we want output = x + 10
                # So left(x) = x + 10 => weight = 1, bias = 10
                self.left.weight.data.fill_(1.0)
                self.left.bias.data.fill_(10.0)
                
                # For right branch (x > 0): we want output = 10 - x  
                # So right(x) = 10 - x => weight = -1, bias = 10
                self.right.weight.data.fill_(-1.0)
                self.right.bias.data.fill_(10.0)
        
            self.split.weight.data.fill_(1.0)
            self.split.bias.data.fill_(0.0)
        
    def forward(self, x):
        if not self.sharpness:
            gate = (self.split(x) <= 0).float()
        else:
            gate = torch.sigmoid(-self.sharpness * self.split(x))

        left_out = self.left(x) * gate
        right_out = self.right(x) * (1 - gate)

        return left_out + right_out

if __name__ == "__main__":
    true_model = TrapezoidNetPure()

    x, y = generate_data(true_model, xmin=-10, xmax=10, n_samples=1_000, eps=1e-4)
    x_test, y_test = generate_data(true_model, xmin=-15, xmax=15, n_samples=1_000, eps=1e-4)

    ########## Save data 
    torch.save({
        'x_train': x, 
        'y_train': y, 
        'x_test': x_test,
        'y_test': y_test,
        'meta_data':{
            'generation_model': 'TrapezoidNet'
        }
    }, 'data/TrapezoidNet.pth')

########## Save model 
torch.save(true_model.state_dict(), 'models/ode_v2.pth')

################### Plot
plt.figure(figsize=(12, 8))
plt.plot(x_test.view(-1), y_test.detach().numpy())
plt.savefig('plots/ode_v2.png')
plt.savefig('plots/ode_v2.pdf')
