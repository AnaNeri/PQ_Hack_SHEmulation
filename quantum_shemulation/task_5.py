from qadence import (feature_map, hea, Z, QuantumModel, add, QuantumCircuit, 
                     kron, FeatureParameter, RX, RZ, VariationalParameter, RY,
                     chain, CNOT, X, Y, CRX, CRZ, I, identity_initialized_ansatz)
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize
import torch
import csv
from copy import deepcopy

shotnoise = True
# Stop optimization when loss threshold is achieved. Slower convergence for shot 
# noise so higher threshold.
loss_threshold = 0.4 if shotnoise else 0.15


seed = 55
torch.manual_seed(seed)
np.random.seed(55)

#Constant and boundary conditions
k = -2.5
x0 = 1.0  

def vqc_fit():
    n_qubits = 2
    
    t = FeatureParameter("t")
    r = FeatureParameter("r")
    '''
    B0 = VariationalParameter("B0")
    B1 = VariationalParameter("B1")
    B2 = VariationalParameter("B2")
    phi0 = VariationalParameter("phi0")
    phi1 = VariationalParameter("phi1")
    phi2 = VariationalParameter("phi2")'''
    C = VariationalParameter("C")
    C2 = VariationalParameter("C2")
    
    # block = chain(RX(0, B0 * t+phi0)*RY(0, B1 * t+phi1)*RZ(0, B1 * t+phi2))
    fm = kron(RX(i, t) for i in range(n_qubits))
    block = fm*hea(n_qubits, depth=2)*kron(RX(i, r) for i in range(n_qubits))
    obs = C*Z(0) # + C2*I(0)
        
    circuit = QuantumCircuit(n_qubits, block)
    nparams = circuit.num_parameters
    model = QuantumModel(circuit, observable = obs)
    nparams = model.num_vparams
    xi = [np.random.uniform(0, 3) for i in range(nparams)]
    
    try:
        res = minimize(loss_fn, x0 = xi, args = (model, t_range, x0, k), method='Powell')
        optparams = res.x
    except LossThreshold as e:
        optparams = e.value

    model.reset_vparams(optparams)

    y_pred = model.expectation({"t": t_range, "r":  torch.zeros(1)}).squeeze().detach()

    return model, y_pred
    
#Since we don't have the training data, we need to define a different loss function based on the diff equation

t_range = torch.linspace(0, 1, 10)  # range of t values

class LossThreshold(Exception):
    def __init__(self, message, value):
        super().__init__(message)
        self.value = value

def samples_to_expectation(outcomes_dict):
    total_count = sum(outcomes_dict.values())
    weighted_sum = 0.0

    for outcome, count in outcomes_dict.items():
        z_value = 1  
        for bit in outcome:
            if bit == '1':  
                z_value *= -1
        weighted_sum += z_value * count

    z_expectation = weighted_sum / total_count
    return z_expectation

def loss_fn(params, *args):
    model, t_range, x0, k = args
    model.reset_vparams(torch.tensor(params))

    x_t = model.expectation({"t": t_range, "r": torch.zeros(1)}).squeeze()
    dx_dts = []
    for i,t in enumerate(t_range):
        # Output is x_t, input is t.
        alpha = np.pi/8
        valsplus = {"t": t, "r": t + alpha}
        valsminus = {"t": t, "r": t - alpha}

        if not shotnoise:
            expplus = model.expectation(valsplus)
            expminus = model.expectation(valsminus)
        else: 
            samplesplus = model.sample(values = valsplus, n_shots = 2048)
            expplus = torch.Tensor([samples_to_expectation(dict(samplesplus[0]))])
            samplesminus = model.sample(values = valsminus, n_shots = 2048)
            expminus = torch.Tensor([samples_to_expectation(dict(samplesminus[0]))])

        exp = 0.5*(expplus-expminus)
        dx_dts.append(exp.item())
    # dxs_dts = torch.stack([x.flatten() for x in dx_dts])
    # dx_dts = dx_dts.squeeze()
    # Check if learned derivative is behaving according to diff eq.
    dx_dts = torch.Tensor(dx_dts)
    residual = dx_dts - k * x_t
    
    # Enforce boundary conditions
    bc1 = x_t[0] - x0  # x(0) = 1.0
    
    loss = torch.mean(residual**2) + bc1**2
    if loss.item() < loss_threshold:
        raise LossThreshold(f"Loss threshold attained", params)
    return loss.detach()

def loss_fn_old(model, t_range, x0, k):
    
    t_range.requires_grad = True
    x_t = model.expectation({"t": t_range, "r": torch.zeros(1)}).squeeze()
    
    # dx/dt
    dx_dt = torch.autograd.grad(x_t.sum(), t_range, create_graph=True)[0]
    dx_dts = []
    for i,t in enumerate(t_range):
        # Output is x_t, input is t.
        alpha = np.pi/8
        exp = 0.5*(model.expectation({"t": t, "r": t + alpha}) - 
               model.expectation({"t": t, "r": t - alpha}))
        dx_dts.append(exp.detach())
    dxs_dts = torch.stack([x.flatten() for x in dx_dts])
    input()
    
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