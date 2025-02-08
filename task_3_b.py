from qadence import (feature_map, hea, Z, QuantumModel, add, QuantumCircuit, 
                     kron, FeatureParameter, RX, RZ, VariationalParameter, RY,
                     chain, CNOT, X, Y, CRX, CRZ, I)
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize
import torch
from sympy import exp
import csv

#Constant and boundary conditions
k = 3.3
d = 0.8 #damping
x0 = 1.0  # x(0) = 1.0
dx0 = 1.2  # x'(0) = 1.2

def vqc_fit(n_qubits, n_epochs):
    
    t = FeatureParameter("t")
    phi = VariationalParameter("phi")
    C = VariationalParameter("C")
    
    block = chain(RX(0, np.sqrt(k-(d/2)**2) * t + phi))
    obs = C*exp(-t*d/2) * Z(0)
        
    circuit = QuantumCircuit(1, block)
    model = QuantumModel(circuit, observable = obs)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_fn(model, t_range, x0, dx0, k, d)
        loss.backward()
        optimizer.step()
        y_pred = model.expectation({"t": t_range}).squeeze().detach()

    return model, y_pred
    
#Once again, we need to adapt the loss function to enforce the diff. eq.

t_range = torch.linspace(0, 10, 100)  # range of t values

def loss_fn(model, t_range, x0, dx0, k, d):
    
    t_range.requires_grad = True
    x_t = model.expectation({"t": t_range}).squeeze()
    
    # dx/dt
    dx_dt = torch.autograd.grad(x_t.sum(), t_range, create_graph=True)[0]
    
    # d^2x/dt^2
    d2x_dt2 = torch.autograd.grad(dx_dt.sum(), t_range, create_graph=True)[0]
    
    # Check if learned derivative is behaving according to diff eq.
    residual = d2x_dt2 + k * x_t + d*dx_dt
    
    # Enforce boundary conditions
    bc1 = x_t[0] - x0  # x(0) = 1.0
    bc2 = dx_dt[0] - dx0  # x'(0) = 1.2
    
    # Total loss: MSE of residual and boundary conditions
    loss = torch.mean(residual**2) + bc1**2 + bc2**2
    return loss

def write_csv(xx, yy, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        for x, y in zip(xx, yy):
            writer.writerow([x.item(), y.item()])

def plot(t_range, y_pred):
    plt.plot(t_range.detach().numpy(), y_pred, label = "Prediction")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.show()

def data_from_file(path):
    with open(path, "r") as file:
        xs = [float(line) for line in file]
    xs = torch.Tensor(xs)
    return xs

show = True

n_qubits = 1
model, y_pred = vqc_fit(n_qubits, n_epochs = 100)
if show:
    plot(t_range, y_pred)

ts = data_from_file("datasets/dataset_3_test.txt")
ypred = model.expectation(values = {"t": ts})
write_csv(ts.detach(), ypred.detach(), "solution_3_b.csv")