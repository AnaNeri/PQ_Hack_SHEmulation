from qadence import (feature_map, hea, Z, QuantumModel, add, QuantumCircuit, 
                     kron, FeatureParameter, RX)
import matplotlib.pyplot as plt
import torch

def vqc_fit(n_qubits, n_epochs):
    fm = feature_map(n_qubits, param = "x")
    x = FeatureParameter("x")
    fm = kron(RX(0, x))
    ansatz = hea(n_qubits, depth = 2)
    obs = add(Z(i) for i in range(n_qubits))
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

    return y_pred

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

n_qubits = 2
x_train, y_train = data_from_file("datasets/dataset_1_a.txt")
y_pred = vqc_fit(n_qubits, n_epochs = 100)
plot(x_train, y_train, y_pred)