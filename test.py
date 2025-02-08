from qadence import (feature_map, hea, Z, QuantumModel, add, QuantumCircuit, 
                     kron, FeatureParameter, RX, RZ, VariationalParameter, RY,
                     chain, expectation)
import matplotlib.pyplot as plt
import numpy as np

block = RX(0, 0.1234)
result = expectation(block, observable = Z(0)).squeeze().detach()
print(result)