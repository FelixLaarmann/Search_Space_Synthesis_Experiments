import os 

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torchdiffeq import odeint

from synthesis.utils import generate_data

def sample_initial_conditions_safe(n_samples, avoid_zero=0.2):

    n_pos = n_samples // 2
    n_neg = n_samples - n_pos
    
    x0_positive = torch.FloatTensor(n_pos, 1).uniform_(avoid_zero, 2.5)
    x0_negative = torch.FloatTensor(n_neg, 1).uniform_(-2.5, -avoid_zero)
    
    x0 = torch.cat([x0_positive, x0_negative], dim=0)
    return x0[torch.randperm(x0.size(0))]  # shuffle

class NonUniformOscilatorPure(nn.Module):
    def __init__(self, random_weights: bool = False):
        super().__init__()

        # Initial 
        self.lin_lay1 = nn.Linear(1, 16, bias=random_weights)

        # Left
        self.l_lin_lay_1 = nn.Linear(16, 4, bias=random_weights)
        self.alpha_l = nn.Linear(16, 4, bias=random_weights)

        # Right 
        self.r_lin_lay_1 = nn.Linear(16, 4, bias=random_weights)

        # Result
        self.combine = nn.Linear(8, 1, bias=random_weights)
        self.result = nn.Linear(4, 1, bias=random_weights)
        self.activation = nn.Tanh()

        if not random_weights:
            torch.manual_seed(42)
            with torch.no_grad():
                nn.init.xavier_normal_(self.l_lin_lay_1.weight)
                nn.init.xavier_normal_(self.r_lin_lay_1.weight)
                nn.init.xavier_normal_(self.combine.weight)
                nn.init.xavier_normal_(self.lin_lay1.weight)
                nn.init.xavier_normal_(self.alpha_l.weight)
                nn.init.xavier_normal_(self.result.weight)
        
    
    def forward(self, t, x):
        # Initial
        x = self.activation(self.lin_lay1(x))

        # Omega 
        res_l = self.alpha_l(x) 
        omega = self.activation(self.l_lin_lay_1(x))
        omega = torch.cos(omega) + res_l

        # a        
        a = self.activation(self.r_lin_lay_1(x))

        comb = self.combine(torch.concat((omega, a), dim=-1))
        result = self.result(omega - a * torch.sin(comb))

        return result

# ===================
# Data Generation 
# ===================

generation_model = NonUniformOscilatorPure()
n_trajectories = 15
n_timesteps = 200

x0_safe = sample_initial_conditions_safe(n_trajectories, avoid_zero=0.2)

print(f'x0: {x0_safe.size()}')
x = torch.linspace(start=0, end=10, steps=n_timesteps)

x_train = [] 
y_train = []

for i, x0 in enumerate(x0_safe):
    with torch.no_grad():
        trajectory = odeint(generation_model, x0, x, method='dopri5', rtol=1e-5, atol=1e-7)
        trajectory = trajectory.squeeze()
        if torch.isnan(trajectory).any() or torch.isinf(trajectory).any():
            print(f'Nan/ Inf skipping: {i+1}')
            continue
        for x_p in trajectory: 
            x_state = x_p.view(-1, 1)
            dxdt = generation_model(torch.zeros_like(x_state), x_state)
            x_train.append(x_state)
            y_train.append(dxdt)


x = torch.cat(x_train, dim=0)
y = torch.cat(y_train, dim=0)
print(f'Training data: x={x.shape} - y={y.shape}')


x_test = torch.linspace(-5, 5, 500).view(-1, 1)
with torch.no_grad():
    y_test = generation_model(torch.zeros_like(x_test), x_test)


print(f'Test data: x={x_test.shape} - y={y_test.shape}')

########## Save data 
torch.save({
    'x_train': x, 
    'y_train': y, 
    'x_test': x_test,
    'y_test': y_test,
    'meta_data':{
        'n_trajectories': n_trajectories,
        'n_timesteps': n_timesteps,
        'generation_model': 'NonUniformOscilatorODE'
    }
}, 'data/oscilator_dataset.pth')


########## Save model 
torch.save(generation_model.state_dict(), 'models/nonuniform_oscilator.pth')

################### Plot
plt.figure(figsize=(12, 8))
plt.plot(x_test.view(-1), y_test.detach().numpy())
plt.savefig('plots/non_osci.png')
plt.savefig('plots/non_osci.pdf')