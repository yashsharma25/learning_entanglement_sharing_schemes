from qutip import *
import numpy as np

'''
    SAMPLE CODE
    def controlled_hadamard():
        # Controlled Hadamard
        return controlled_gate(
            hadamard_transform(1), 2, control=0, target=1, control_value=1)

    qc = QubitCircuit(N=3, num_cbits=3)
    qc.user_gates = {"cH": controlled_hadamard}
    qc.add_gate("QASMU", targets=[0], arg_value=[1.91063, 0, 0])
    qc.add_gate("cH", targets=[0,1])
    qc.add_gate("TOFFOLI", targets=[2], controls=[0, 1])
    qc.add_gate("X", targets=[0])
    qc.add_gate("X", targets=[1])
    qc.add_gate("CNOT", targets=[1], controls=0)

    zero_state = tensor(basis(2, 0), basis(2, 0), basis(2, 0))
    result = qc.run(state=zero_state)
    wstate = result`

    print(wstate)
'''

qc = QubitCircuit(N=3, num_cbits=0)

L = 1
n = 3

arr = np.array([[-0.894648595197737, 0.3228130349803821, -0.17866782425319155],
                [0.0902540294917668, 0.2010138681738993, -0.29340612707204483], 
                [-0.2846992020764751, 0.4195454927833951, -0.8966761977354316]])

def cnChoose(control, n):
    if control+1 == n:
        return 0
    else:
        return control + 1

# L is depth
# n is number of qubits
for d in range(L):
    for i in range(n):
        qc.add_gate("QASMU", targets=[i], arg_value=arr[i])
    for i in range(n):
        qc.add_gate("CNOT", controls=[i], targets=[cnChoose(i, 3)])

s1 = tensor(basis(2, 0), basis(2, 0), basis(2, 0))
s2 = qc.run(state=s1)
print(s2)
p2 = s2.ptrace(0)
print(p2)
p1 = s1.ptrace(1)
print(p1)

#print(qc.gates)
