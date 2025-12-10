import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from synthesis.utils import fit_model, generate_data, get_num_parameters

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

class MLP(nn.Module):
    def __init__(self, n_hidden = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, x):
        return self.net(x)

# Generate true model + data
true_model = TrapezoidNetPure()
print(f"Original #Parameters: {get_num_parameters(true_model)}")

# Generate data is a bit noisy to make it closer to the real world
# test data is a little bit out of distribution because we change xmin/xmax a little bit
x,y = generate_data(true_model, xmin=-10, xmax=10, n_samples=1_000, eps=1e-4)
x_test,y_test = generate_data(true_model, xmin=-15,xmax=15, n_samples=1_000, eps=1e-4)

# Try a few models
models = [
    ("MLP-small", MLP(2)),
    ("MLP-medium", MLP(5)),
    ("MLP-large", MLP(10)),
    ("TrapezoidNet", TrapezoidNetPure(random_weights=True, sharpness=2))
]
results = []
for name, m in models:
    m = fit_model(m, x, y, name=name, n_epochs=10_000)
    loss_fn = nn.MSELoss()
    with torch.inference_mode():
        y_pred = m(x_test).ravel()
        loss = loss_fn(y_pred, y_test)
    results.append({"name":name, "loss":loss.item(), "parameters":get_num_parameters(m)})

# Plot true function
plt.figure(figsize=(12, 8))
plt.plot(x_test.view(-1), y_test, label="True Trapezoid", linewidth=3)

# Plot each model's prediction
for name, m in models:
    with torch.inference_mode():
        y_pred = m(x_test).ravel()
        plt.plot(x_test.view(-1), y_pred, label=f"Learned {name}", linestyle="--")

plt.title("True vs Learned Trapezoid Mapping")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.savefig("plot.pdf")
plt.savefig("plot.png")
# plt.show()

# Print results table
print(pd.DataFrame(results))