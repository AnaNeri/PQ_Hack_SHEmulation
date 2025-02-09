from qadence import (feature_map, hea, Z, QuantumModel, add, QuantumCircuit, 
                     kron, FeatureParameter, RX, RZ, VariationalParameter, RY,
                     chain, CNOT, X, Y, CRX, CRZ, I)
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize
import torch
from sympy import exp

#Constant and boundary conditions
k = -2.5
x0 = 1.0  # x(0) = 1.0

def vqc_fit(n_qubits, n_epochs):
    
    t = FeatureParameter("t")
    C = VariationalParameter("C")
    phi = VariationalParameter("phi")
    
    block = X(0)
    obs = C*exp(k**0.5*t+phi) * Z(0)
        
    circuit = QuantumCircuit(1, block)
    model = QuantumModel(circuit, observable = obs)

    criterion = torch.nn.MSELoss()
    xi = [np.random.uniform(0, 3) for i in range(2)]
    # xi = torch.Tensor(xi)

    res = minimize(loss_fn, x0 = xi, args = (model, t_range), method='Powell')
    model.reset_vparams(res.x)

    y_pred = model.expectation({"t": t_range}).squeeze().detach()

    return model, y_pred
    
#Once again, we need to adapt the loss function to enforce the diff. eq.

t_range = torch.linspace(0, 1, 100)  # range of t values

def loss_fn(params, *args):
    C, phi = params
    model, t_range = args
    t_range.requires_grad = True
    model.reset_vparams(torch.tensor(params))
    x_t = model.expectation({"t": t_range}).squeeze().detach()
    dx_dt = torch.autograd.grad(x_t.sum(), t_range, create_graph=True)[0]
    residual = dx_dt - k * x_t
    bc1 = np.abs(x_t[0] - x0)
    loss = torch.mean(residual**2) + bc1**2 
    return loss.detach()

def plot(t_range, y_pred):
    plt.plot(t_range.detach().numpy(), y_pred, label = "Prediction")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.show()

quantum = True
show = True

if quantum: 
    n_qubits = 1
    model, y_pred = vqc_fit(n_qubits, n_epochs = 100)
    if show:
        plot(t_range, y_pred)
    vparams = model.vparams