import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

def get_num_parameters(model):
    """
    Calculate the total number of parameters in a PyTorch model.
    
    Args:
        model (torch.nn.Module): The PyTorch model to analyze
        
    Returns:
        int: Total number of parameters in the model
        
    Example:
        >>> model = nn.Linear(10, 5)
        >>> num_params = get_num_parameters(model)
        >>> print(f"Model has {num_params} parameters")
    """
    return sum(p.numel() for p in model.parameters())

def generate_data(true_model, n_samples, xmin=-10, xmax=10, eps=0.0):
    """
    Generate synthetic data by evaluating a true model on a grid of points and adding Gaussian noise.
    
    Note: 
        model evaluation is not batched!

    Args:
        true_model (torch.nn.Module): The true model to evaluate
        n_samples (int): Number of samples to generate
        xmin (float, optional): Minimum x value. Defaults to -10.
        xmax (float, optional): Maximum x value. Defaults to 10.
        eps (float, optional): Variance of Gaussian noise to add. Defaults to 0.0.
        
    Returns:
        tuple: A tuple containing (x, y) where x is the input tensor and y is the output tensor with optional noise added
        
    Example:
        >>> x, y = generate_data(trapezoid_model, 100, eps=0.1)
        >>> print(f"Generated {len(x)} samples")
    """
    x = torch.linspace(xmin, xmax, n_samples).view(-1,1)
    y = true_model(x).detach().view(-1)
    
    if eps > 0:
        noise = torch.normal(mean=0, std=torch.sqrt(torch.tensor(eps)), size=y.shape)
        y = y + noise
    
    return x,y

def fit_model(model, x, y, n_epochs=2_000, verbose=True, name="model"):
    """
    Train a PyTorch model on given data using Adam optimizer and MSE loss. 
     
    Note:
        Technically, this function implements Gradient Descent and not Stochastic Gradient Descent. There is no proper batching performed. 
        
    Args:
        model (torch.nn.Module): The model to train
        x (torch.Tensor): Input tensor
        y (torch.Tensor): Target tensor
        n_epochs (int, optional): Number of training epochs. Defaults to 2000.
        verbose (bool, optional): Whether to show progress bar. Defaults to True.
        name (str, optional): Name of the model for progress bar display. Defaults to "model".
        
    Returns:
        torch.nn.Module: The trained model
        
    Example:
        >>> trained_model = fit_model(model, x_train, y_train, n_epochs=1000)
        >>> print("Model training completed")
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    
    pbar = tqdm.tqdm(range(n_epochs), total=n_epochs, desc=f"Training {name}", disable=not verbose)
    
    for _ in pbar:
        optimizer.zero_grad()
        pred = model(x).ravel()
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})
    
    return model