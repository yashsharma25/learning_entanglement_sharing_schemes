from qutip import *
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    p1 = s1.ptrace(0)

    print("INPUT: \n", p1)

    s2 = qc.run(state=s1)
    p2 = s2.ptrace(0)
    print("OUTPUT: \n", p2)

    #print(qc.gates)
'''

# BELL STATE
print("BELL --------------------------------------------------\n")
bell = tensor(basis(2, 0), basis(2, 0))

bellC = QubitCircuit(N=2, num_cbits=0)
bellC.add_gate("SNOT", targets=0)
bellC.add_gate("CNOT", controls=0, targets=1)
bellE = bellC.run(bell)
print("INPUT: ", tensor(basis(2, 0), basis(2, 0), basis(2, 0)).ptrace(0), "\n")
print("OUTPUT: ", bellE.ptrace(0))

# GHZ STATE
print("\nGHZ ---------------------------------------------------\n")

ghz = QubitCircuit(N=3, num_cbits=3)
ghz.add_gate("SNOT", targets=0)
ghz.add_gate("CNOT", controls=0, targets=1)
ghz.add_gate("CNOT", controls=0, targets=2)

'''
ghz.add_measurement("M0", targets=[0], classical_store=0)
ghz.add_measurement("M1", targets=[1], classical_store=1)
ghz.add_measurement("M2", targets=[2], classical_store=2)

ghzE = ghz.run_statistics(state=tensor(basis(2,0), basis(2,0), basis(2,0)))
ghzES = ghzE.get_final_states()
ghzEP = ghzE.get_probabilities()

for state, probability in zip(ghzES, ghzEP):
    print("State:\n{}\nwith probability {}".format(state, probability))
'''

print("INPUT: ", tensor(basis(2, 0), basis(2, 0), basis(2, 0)).ptrace(0), "\n")
print("OUTPUT: ", ghz.run(tensor(basis(2, 0), basis(2, 0), basis(2, 0))).ptrace(0))

# W STATE
print("\nW ------------------------------------------------------\n")


def controlled_hadamard():
    # Controlled Hadamard
    return controlled_gate(
        hadamard_transform(1), 2, control=0, target=1, control_value=1)


qc = QubitCircuit(N=3, num_cbits=3)
qc.user_gates = {"cH": controlled_hadamard}
qc.add_gate("QASMU", targets=[0], arg_value=[1.91063, 0, 0])
qc.add_gate("cH", targets=[0, 1])
qc.add_gate("TOFFOLI", targets=[2], controls=[0, 1])
qc.add_gate("X", targets=[0])
qc.add_gate("X", targets=[1])
qc.add_gate("CNOT", targets=[1], controls=0)

'''
qc.add_measurement("M0", targets=[0], classical_store=0)
qc.add_measurement("M1", targets=[1], classical_store=1)
qc.add_measurement("M2", targets=[2], classical_store=2)

result = qc.run_statistics(state=tensor(basis(2, 0), basis(2, 0), basis(2, 0)))
states = result.get_final_states()
probabilities = result.get_probabilities()

for state, probability in zip(states, probabilities):
    print("State:\n{}\nwith probability {}".format(state, probability))
'''

resultE = qc.run(tensor(basis(2, 0), basis(2, 0), basis(2, 0)))
print("INPUT: ", tensor(basis(2, 0), basis(2, 0), basis(2, 0)).ptrace(0), "\n")
print("OUTPUT: ", resultE.ptrace(0))

print("\nANSATZ GENERATOR ---------------------------------------\n")

qc = QubitCircuit(N=3, num_cbits=0)

# L is depth
# n is number of qubits
L = 1
n = 3

arr1 = np.array([[[-0.894648595197737, 0.3228130349803821, -0.17866782425319155], [0.0902540294917668,
                0.2010138681738993, -0.29340612707204483], [-0.2846992020764751, 0.4195454927833951, -0.8966761977354316]]])


def cnChoose(control, n):
    if control+1 == n:
        return 0
    else:
        return control + 1

# qc.add_gate("QASMU", targets=0, arg_value=[np.pi, np.pi, np.pi/2])


for layer in range(L):
    for i in range(n):
        arr2 = arr1[layer]
        # qc.add_gate("QASMU", targets=[i], arg_value=arr[i])
        # circuit.Rz(syms[0], j).Ry(syms[1], j).Rz(syms[2], j)
        print(layer, '\t', i, '\t', n)
        qc.add_gate("RZ", targets=[i], arg_value=arr2[i][0])
        qc.add_gate("RY", targets=[i], arg_value=arr2[i][1])
        qc.add_gate("RZ", targets=[i], arg_value=arr2[i][2])

    step = layer % (n - 1) + 1
    for i in range(n):
        print(i, (i+step) % n)
        qc.add_gate("CNOT", controls=i, targets=((i+step) % n))

'''
for d in range(L):
    for i in range(n):
        qc.add_gate("QASMU", targets=[i], arg_value=arr[i])
    for i in range(n+1):
        qc.add_gate("CNOT", controls=2*i, targets=((2*i)+1)%n)
'''

#
print("INPUT: ", tensor(basis(2, 0), basis(2, 0), basis(2, 0)))
generated_state = qc.run(tensor(basis(2, 1), basis(2, 1), basis(2, 1)))
# print("gS: ", generated_state.ptrace(1))

rdm_A = generated_state.ptrace(1)

print("Type = ", type(rdm_A.data))
rdm_A_squared = rdm_A.data @ rdm_A.data

# convert to a numpy matrix
rdm_A_squared_np = np.array(rdm_A_squared.data).reshape(2, 2)

print("shape= ", rdm_A_squared_np.shape)
print("Trace of rdm_A_np squared ", np.trace(rdm_A_squared_np))


# print(qc.run(tensor(basis(2,1))))
