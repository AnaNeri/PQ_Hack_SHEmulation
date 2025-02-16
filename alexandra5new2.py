from qadence import (feature_map, hea, Z, QuantumModel, add, QuantumCircuit, 
                     kron, FeatureParameter, RX, RZ, VariationalParameter, RY,
                     chain, CNOT, X, Y, CRX, CRZ, I, identity_initialized_ansatz)
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize
import torch
import csv

seed = 55
torch.manual_seed(seed)
np.random.seed(55)

#Constant and boundary conditions
k = -2.5
x0 = 1.0  

def vqc_fit():
    n_qubits = 2
    
    t = FeatureParameter("t")
    C = VariationalParameter("C")
    C2 = VariationalParameter("C2")
    
    # block = chain(RX(0, B0 * t+phi0)*RY(0, B1 * t+phi1)*RZ(0, B1 * t+phi2))
    fm = kron(RX(i, t) for i in range(n_qubits))
    block = fm*hea(n_qubits, depth=2)
    obs = C*Z(0) # + C2*I(0)
        
    circuit = QuantumCircuit(n_qubits, block)
    model = QuantumModel(circuit, observable = obs)

    n_epochs = 500
    criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_fn(inputs = t_range, model = model)
        loss.backward()
        optimizer.step()
        y_pred = model.expectation({"t": t_range}).squeeze().detach()

    print(loss)
    return model, y_pred
    
#Since we don't have the training data, we need to define a different loss function based on the diff equation

t_range = torch.linspace(0, 1, 100)  # range of t values

def loss_fn(model, inputs):
    """Loss function encoding the problem to solve."""
    # Equation loss
    model_output = model(inputs)
    deriv_model = calc_deriv(model_output, inputs)
    deriv_exact = dfdx_equation(inputs)
    criterion = torch.nn.MSELoss()
    ode_loss = criterion(deriv_model, deriv_exact)

    # Boundary loss, f(0) = 0
    boundary_model = model(torch.tensor([[0.0]]))
    boundary_exact = torch.tensor([[0.0]])
    boundary_loss = criterion(boundary_model, boundary_exact)

    return ode_loss + boundary_loss

def calc_deriv(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Compute a derivative of model that learns f(x), computes df/dx using torch.autograd."""
    grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs = torch.ones_like(inputs),
        create_graph = True,
        retain_graph = True,
    )[0]
    return grad

def dfdx_equation(x: torch.Tensor) -> torch.Tensor:
    """Derivative as per the equation."""
    return k*x

def loss_fn_old(model, t_range, x0, k):
    
    t_range.requires_grad = True
    x_t = model.expectation({"t": t_range}).squeeze()
    
    # dx/dt
    dx_dt = torch.autograd.grad(x_t.sum(), t_range, create_graph=True)[0]
    dx_dt5 = torch.autograd.grad(x_t[5], t_range, create_graph=True)[0]
    
    # Check if learned derivative is behaving according to diff eq.
    residual = dx_dt - k * x_t
    
    # Enforce boundary conditions
    bc1 = x_t[0] - x0  # x(0) = 1.0
    
    # Total loss: MSE of residual and boundary conditions
    loss = torch.mean(residual**2) + bc1**2 
    return loss

def plot(t_range, y_pred):
    plt.plot(t_range.detach().numpy(), y_pred, label = "Prediction")
    #compare with what is expected
    #plt.plot(t_range.detach().numpy(), 1.27*np.cos(np.sqrt(k)*t_range.detach().numpy()-0.657))
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.show()

show = True

n_qubits = 1
model, y_pred = vqc_fit()
if show:
    plot(t_range, y_pred)

res = zip(t_range.detach().numpy(), y_pred.detach().numpy())
vparams = model.vparams

def write_csv(data, filename):
    with open(filename, mode='w') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)

file = "solution_5.csv"
    
write_csv(res,file)