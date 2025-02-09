from qadence import (feature_map, hea, Z, QuantumModel, add, QuantumCircuit, 
                     kron, FeatureParameter, RX, RZ, VariationalParameter, RY,
                     chain)
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import torch

def vqc_fit(n_qubits, n_epochs):
    fm = feature_map(n_qubits, param = "x")
    x = FeatureParameter("x")
    fm = kron(RX(i, (i+1)*x) for i in range(n_qubits))

    theta1 = VariationalParameter("theta1")
    theta2 = VariationalParameter("theta2")
    A1 = VariationalParameter("A1")
    A2 = VariationalParameter("A2")
    ansatz = chain(RX(0, theta1), RX(1,theta2))

    obs = (A1*Z(0) + A2*Z(1))
    block = fm * ansatz

    circuit = QuantumCircuit(n_qubits, block)
    model = QuantumModel(circuit, observable = obs)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_fn(x_train, y_train, model, criterion)
        loss.backward()
        optimizer.step()
        y_pred = model.expectation({"x": x_train}).squeeze().detach()

    return model, y_pred

def loss_fn(x_train, y_train, model, criterion):
    output = model.expectation({"x": x_train}).squeeze()
    loss = criterion(output, y_train)
    return loss

def data_from_file(path):
    with open(path, "r") as file:
        points = [tuple(map(float, line.split())) for line in file]

    x_train = torch.Tensor([point[0] for point in points])
    y_train = torch.Tensor([point[1] for point in points])
    return x_train, y_train

def plot(x_train, y_train, y_pred):
    plt.plot(x_train, y_pred, label = "Prediction")
    plt.scatter(x_train, y_train, label = "Training points")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()

x_train, y_train = data_from_file("datasets/dataset_2_a.txt")

n_qubits = 2
model, y_pred = vqc_fit(n_qubits, n_epochs = 100)
#plot(x_train, y_train, y_pred)
vparams = model.vparams
print('theta1', vparams['theta1'].item())
print('theta2', vparams['theta2'].item())
print('A1', vparams['A1'].item())
print('A2',vparams['A2'].item())