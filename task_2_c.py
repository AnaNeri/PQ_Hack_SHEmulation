from qadence import (feature_map, hea, Z, QuantumModel, add, QuantumCircuit, 
                     kron, FeatureParameter, RX, RZ, VariationalParameter, RY,
                     chain, CNOT, X, Y, CRX, CRZ, I)
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize
import torch
import csv

torch.manual_seed(55)
np.random.seed(55)

def vqc_fit(n_qubits, n_epochs):
    n_qubits = 3
    x = FeatureParameter("x")
    A1 = VariationalParameter("A1")
    A2 = VariationalParameter("A2")
    A3 = VariationalParameter("A3")
    f1 = VariationalParameter("f1")
    f2 = VariationalParameter("f2")
    f3 = VariationalParameter("f3")
    phi1 = VariationalParameter("phi1")
    phi2 = VariationalParameter("phi2")
    phi3 = VariationalParameter("phi3")
    B = VariationalParameter("B")
   
    fm =  RX(0, f1*x) @ RX(1, f2*x) @ RX(2, f3*x)

    theta = VariationalParameter("phi")
    ansatz =   RX(0, phi1) * RX(1, phi2) * RX(2, phi3)

    As = [A1, A2, A3]
    obs = add(As[i]*Z(i) for i in range(n_qubits)) + VariationalParameter("B")*I(0)
    block = fm * ansatz

    circuit = QuantumCircuit(n_qubits, block)
    model = QuantumModel(circuit, observable = obs)

    criterion = torch.nn.MSELoss()
    x0 = [np.random.uniform(0, 3) for i in range(10)]

    res = minimize(loss_fn, x0 = x0, args = (x_train, y_train, model, criterion), method='Powell')
    model.reset_vparams(res.x)

    y_pred = model.expectation({"x": x_train}).squeeze().detach()

    return model, y_pred

def loss_fn(params, *args):
    x_train, y_train, model, criterion = args
    model.reset_vparams(torch.tensor(params))
    output = model.expectation({"x": x_train}).squeeze()
    loss = criterion(output, y_train)
    return loss.detach()

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

def plot2(x_train, y_train, y_pred):
    plt.scatter(x_train, y_pred, label = "Predicted points", color = "red")
    plt.scatter(x_train, y_train, label = "Points from file")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()

def scipy_verification(x_data, y_data):
    def model(x, A1, f1, phi1, A2, f2, phi2, A3, f3, phi3, B):
        return A1*np.cos(f1*x + phi1) + A2*np.cos(f2*x + phi2) + A3*np.cos(f3*x + phi3) + B
    
    params, covariance = curve_fit(model, x_data, y_data, p0=[3, 1, 1, 1, 3, 1, 3, 3, 3, 3])  
    A1, f1, phi1, A2, f2, phi2, A3, f3, phi3, B = params
    param_str = ["A1", "f1", "phi1", "A2", "f2", "phi2", "A3", "f3", "phi3", "B"]
    for param, str in zip(params, param_str):
        print(str, param)
    plt.scatter(x_data, y_data, label="Data", color='red')
    plt.plot(x_data, model(x_data, *params), label="Fitted model", color='blue')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def write_csv(xx, yy):
    with open("solution_2_c.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        for x, y in zip(xx, yy):
            writer.writerow([x.item(), y.item()])

quantum = True
show = False
x_train, y_train = data_from_file("datasets/dataset_2_c.txt")
xnew, ynew = data_from_file("datasets/dataset_2_c_test.txt")

if quantum: 
    n_qubits = 2
    model, y_pred = vqc_fit(n_qubits, n_epochs = 100)
    if show:
        plot(x_train, y_train, y_pred)
    vparams = model.vparams
    #for p in ["A1", "f1", "phi1", "A2", "f2", "phi2", "A3", "f3", "phi3", "B"]:
    #    print(p, vparams[p].item()+2*np.pi if p=='phi' else vparams[p].item())

    ypred = model.expectation(values = {"x": xnew})
    # for yp, y in zip(ypred, ynew):
    #    print("pred: ", yp.item(),"| y:", y)
    write_csv(xnew.detach(), ypred.detach())
    plot2(xnew.detach(), ynew.detach(), ypred.detach())
else: 
    scipy_verification(x_train, y_train)
