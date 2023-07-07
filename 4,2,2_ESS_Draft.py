import qiskit as q
import numpy as np
import matplotlib as plt 

from qiskit import Aer
from qiskit import quantum_info as qI
from qiskit import visualization as qV

# Run the quantum circuit on a statevector simulator backend
backend = Aer.get_backend('qasm_simulator')

# GHZ STATE
ghzQR = q.QuantumRegister(8)
ghzCR = q.ClassicalRegister(8)
ghzC = q.QuantumCircuit(ghzQR, ghzCR)

ghzC.h(ghzQR[0])

for i in range(8):
    if i != 0:
        ghzC.cx(ghzQR[0], ghzQR[i])

ghzC.draw(output='mpl')

# EQ 12 On Page 3 in the Paper
'''
I'm having trouble implementing the encoding for the ESS in this section because I cant figure out a way to change the measurement basis from the standard basis
to the required basis. After that, we can move from the specific [4,2,2] example to the general case.
'''
beta0 = ((qI.Statevector.from_label("00") + qI.Statevector.from_label("11"))/np.sqrt(2))
beta1 = ((qI.Statevector.from_label("00") - qI.Statevector.from_label("11"))/np.sqrt(2))
beta2 = ((qI.Statevector.from_label("01") + qI.Statevector.from_label("10"))/np.sqrt(2))
beta3 = ((qI.Statevector.from_label("01") - qI.Statevector.from_label("10"))/np.sqrt(2))

eq12 = 1/2 * (beta0 + beta1 + beta2 + beta3)

