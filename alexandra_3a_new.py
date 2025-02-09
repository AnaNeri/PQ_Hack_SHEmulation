from qadence import (feature_map, hea, Z, QuantumModel, add, QuantumCircuit, 
                     kron, FeatureParameter, RX, RZ, VariationalParameter, RY,
                     chain, CNOT, X, Y, CRX, CRZ, I, BasisSet, identity_initialized_ansatz,
                     ala, rydberg_hea_layer)
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize
import torch
import csv

seed = 55
torch.manual_seed(seed)

#Constant and boundary conditions
k = 2.3
x0 = 1.0  # x(0) = 1.0
dx0 = 1.2  # x'(0) = 1.2

def vqc_fit(n_epochs):
    n_qubits = 3
    fm = feature_map(n_qubits, param = "t")#, fm_type=BasisSet.FOURIER)

    ansatz = identity_initialized_ansatz(n_qubits, depth = 3)

    
    As = [VariationalParameter(f"A{i}") for i in range(n_qubits)]
    obs =  add(As[i]*Z(i) for i in range(n_qubits))*VariationalParameter(f"B")
    block =  chain(fm*ansatz for i in range(2))
        
    circuit = QuantumCircuit(n_qubits, block)
    model = QuantumModel(circuit, observable = obs)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_fn(model, t_range, x0, dx0, k)
        loss.backward()
        optimizer.step()
        y_pred = model.expectation({"t": t_range}).squeeze().detach()
        losses.append(loss.detach().numpy())
    print(losses)
    bplot(list(range(n_epochs)), losses)
    return model, y_pred
    
#Since we don't have the training data, we need to define a different loss function based on the diff equation

t_range = torch.linspace(0, 10, 100)  # range of t values

def loss_fn(model, t_range, x0, dx0, k):
    
    t_range.requires_grad = True
    x_t = model.expectation({"t": t_range}).squeeze()
    
    # dx/dt
    dx_dt = torch.autograd.grad(x_t.sum(), t_range, create_graph=True)[0]
    
    # d^2x/dt^2
    d2x_dt2 = torch.autograd.grad(dx_dt.sum(), t_range, create_graph=True)[0]
    
    # Check if learned derivative is behaving according to diff eq.
    residual = d2x_dt2 + k * x_t
    
    # Enforce boundary conditions
    bc1 = x_t[0] - x0  # x(0) = 1.0
    bc2 = dx_dt[0] - dx0  # x'(0) = 1.2
    
    # Total loss: MSE of residual and boundary conditions
    loss = torch.mean(residual**2) + bc1**2 + bc2**2
    return loss

def plot(t_range, y_pred):
    plt.plot(t_range.detach().numpy(), y_pred, label = "Prediction")
    #compare with what is expected
    #plt.plot(t_range.detach().numpy(), 1.27*np.cos(np.sqrt(k)*t_range.detach().numpy()-0.657))
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.show()

def bplot(x, y):
    plt.scatter(x, y)
    plt.show()

show = True


model, y_pred = vqc_fit(n_epochs = 100)
if show:
    plot(t_range, y_pred)

res = zip(t_range.detach().numpy(), y_pred.detach().numpy())
vparams = model.vparams

def write_csv(data, filename):
    with open(filename, mode='w') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)

file = "solution_3_a.csv"
    
write_csv(res,file)