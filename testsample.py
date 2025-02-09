import numpy as np
from qadence import H, CNOT, sample, QuantumCircuit, QuantumModel, Z, RX, FeatureParameter
import torch 

n_qubits = 3
r = FeatureParameter("r")
block = RX(0,r*np.pi/4) * CNOT(0, 1) * CNOT(0, 2)
circuit = QuantumCircuit(n_qubits, block)
obs = Z(0)
model = QuantumModel(circuit, observable = obs)
vals = {"r": torch.ones(1)}
samples = model.sample(values = vals, n_shots = 100)
exp = model.expectation(values = vals)

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

samples = dict(samples[0])
print(samples)
print(exp)
print(samples_to_expectation(samples))