from qadence import (feature_map, hea, Z, QuantumModel, add, QuantumCircuit, 
                     kron, FeatureParameter, RX, RZ, VariationalParameter, RY,
                     chain, CNOT, X, Y, CRX, CRZ, I)
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize
import torch
from sympy import exp
from mpl_toolkits.mplot3d import Axes3D

#Constant and boundary conditions
k = 1
xmin = -1  
xmax = 1  
ymin = -1
ymax = 1
omega1 = np.pi
omega2 = 2*np.pi

torch.manual_seed(108)

def vqc_fit(n_qubits, n_epochs, x_range, y_range, xmin, xmax, k):
    
    x = FeatureParameter("x")
    y = FeatureParameter("y")
    phi = VariationalParameter("phi")
    phi1 = VariationalParameter("phi1")
    b1= VariationalParameter("b1")
    b2= VariationalParameter("b2")
    b3= VariationalParameter("b3")
    b4= VariationalParameter("b4")
    c1= VariationalParameter("c1")
    c2= VariationalParameter("c2")
    
    block = chain(
        kron(RX(0, b1*x), RX(1, b2*y)),
        kron(RY(0, omega1 * x*c1), RY(1, omega2 * y*c2)) ,
        kron(RX(0, b1*x ), RX(1,b2*y)),
        kron(RY(0, omega1 * x*c1), RY(1, omega2 * y*c2)),
    )    
    obs = 1/2 * kron(Z(0), Z(1), Z(2), Z(3))
        
    circuit = QuantumCircuit(6, block)
    model = QuantumModel(circuit, observable = obs)
    
    # Example of a learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_fn(model, x_range, y_range, xmin, xmax, k)
        loss.backward()
        optimizer.step()
    # Generate predictions for the entire grid
    x_grid, y_grid = torch.meshgrid(x_range, y_range)
    u_pred = model.expectation({"x": x_grid.flatten(), "y": y_grid.flatten()}).squeeze().detach()
    u_pred = u_pred.reshape(x_grid.shape)

    return model, u_pred


def loss_fn(model, x_range, y_range, xmin, xmax, k):
    
    x_range = x_range.clone().detach().requires_grad_(True)
    y_range = y_range.clone().detach().requires_grad_(True)
    
    us = model.expectation({"x": x_range, "y": y_range}).squeeze()
    
    # Compute second derivates
    dxs = torch.autograd.grad(us.sum(), x_range, create_graph=True)[0]
    dxs2 = torch.autograd.grad(dxs.sum(), x_range, create_graph=True)[0]
    
    dys = torch.autograd.grad(us.sum(), y_range, create_graph=True)[0]
    dys2 = torch.autograd.grad(dys.sum(), y_range, create_graph=True)[0]
    
    # forcing term 
    q_x_y = (k**2 - omega1**2 - omega2**2) * torch.sin(omega1 * x_range) * torch.sin(omega2 * y_range)
    
    # Check if learned derivative is behaving according to diff eq.
    residual = dxs2 + dys2 + k**2 * us - q_x_y
    
    # Enforce Dirichlet boundary conditions: u(x, ±1) = 0 and u(±1, y) = 0
    boundary_loss = (
        torch.mean(us[x_range == xmin]) +
        torch.mean(us[x_range == xmax]) +
        torch.mean(us[y_range == ymin]) +
        torch.mean(us[y_range == ymax])
    )
    
    # Total loss: MSE of residual and boundary conditions
    loss = torch.mean(residual**2) + torch.abs(boundary_loss*1000000) 
    
    return loss

def plot_2d(x_range, y_range, u_pred):
    plt.figure(figsize=(8, 6))
    
    x, y = np.meshgrid(x_range.detach().numpy(), y_range.detach().numpy())
    u = u_pred.detach().numpy().reshape(x.shape)
    
    plt.contourf(x, y, u, levels=100, cmap='viridis')
    plt.colorbar(label='u')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D Contour plot of u')
    plt.show()

def plot_3d(x_range, y_range, u_pred):
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x, y = np.meshgrid(x_range.detach().numpy(), y_range.detach().numpy())
    u = u_pred.detach().numpy().reshape(x.shape)
    
    ax.plot_surface(x, y, u, cmap='viridis')
    ax.set(xlabel='x', ylabel='y', zlabel='u')
    ax.set_title('3D Surface plot of u')
    plt.show()

show = True


# write results in csv file
import csv
def write_csv(xx, yy, uu,  filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        for x, y, u in zip(xx, yy, uu):
            writer.writerow([x.item(), y.item(), u.item()])

def data_from_file(path):
    with open(path, "r") as file:
        # each line has two floats x and y
        xs, ys = zip(*[map(float, line.split()) for line in file])
    xs = torch.Tensor(xs)
    ys = torch.Tensor(ys)
    return xs, ys

xx, yy = data_from_file("datasets/dataset_4_test.txt")
print('---',xx, yy)

n_qubits = 4
model, uu = vqc_fit(n_qubits, 200, xx, yy, xmin, xmax, k)
if show:
    plot_2d(xx,yy, uu)
    plot_3d(xx,yy, uu)
vparams = model.vparams
write_csv(xx, yy, uu, "solution_4.csv")