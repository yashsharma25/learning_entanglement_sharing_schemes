import pennylane as qml
from pennylane import numpy as np

from qiskit.quantum_info import Statevector
from qiskit.quantum_info import DensityMatrix, partial_trace, entropy
from qiskit import QuantumCircuit, execute, Aer

n_qubits = 20
dev = qml.device('default.qubit', wires=n_qubits)


#circuit definition
@qml.qnode(dev)
def sea(weights):
    for layer in range(2):
        # rotations on first 10 qubits
        for i in range(10):
            qml.Rot(weights[0, i, 0], weights[0, i, 1], weights[0, i, 2], wires=i)
            
        # rotations on second 10 qubits    
        for i in range(10,20):
            qml.Rot(weights[1, i, 0], weights[1, i, 1], weights[1, i, 2], wires=i)   
            
        # entangle first 10 qubits
        for i in range(9):
            qml.CNOT(wires=[i, i+1])
            
        # entangle second 10 qubits
        for i in range(10,19):
            qml.CNOT(wires=[i, i+1])
            
    return qml.state()

#loss functoin
def cost(weights):
  state = sea(weights)
  # Cost function from state
  
  # Calculate CE of state
  e = entanglement_entropy(state)  

  # Target CE value
  target_e = 1

  # Cost function (mean squared error)
  loss = np.mean((e - target_e)**2)

  return loss


# Training loop
def train(n_train, n_test, n_epochs, learning_rate = 0.15):
    #load data
    input_states = load_states_data()
    N = len(input_states)
    max_epochs = 10
    target_entanglement = 1.0

    weights = np.random.random(size=(2,2,3))  

    # Optimizer
    opt = qml.AdamOptimizer(0.01)

    for step in range(n_epochs):
        # Compute loss over training set 
        loss = 0 
        for state in input_states:
            output_state = sea(weights) # Apply SEA circuit
            loss += (entanglement_entropy(output_state) - target_entanglement)**2 # Compare target and output CE
            
        loss = loss/N # Take average 
        weights = opt.step(loss, weights)

def entanglement_entropy(state, subsystem):
    # Convert the state to a density matrix
    rho = DensityMatrix(state)
    
    # Compute the reduced density matrix for the specified subsystem
    rho_subsystem = partial_trace(rho, subsystem)
    
    # Calculate the entanglement entropy
    S = entropy(rho_subsystem, base=2)
    
    return S

# def evaluate():
#     # Final evaluation
#     test_states = {|φ1⟩, |φ2⟩ ... } # Generate test set
#     success_rate = 0
#     for |φ⟩ in test_states:
#     |φ'⟩ = SEA(weights, |φ⟩)
#     if (|CE(|φ'⟩) - ξ| < tolerance):
#         success_rate += 1
        
#     print(success_rate/len(test_states))    

def load_general_product_states(n_qubits):
    # Initialize a random product state
    state = np.array([1.0])
  
    for i in range(n_qubits):
        U = random_unitary()
        state = np.kron(state, U)
    return state

def load_computational_basis_states():
    num_qubits = 20
    dim = 2**num_qubits 

    states = []
    for i in range(dim):
        state = np.zeros(dim)
        state[i] = 1  
        states.append(state)
        
    print(len(states)) # 2^20 states
    print(states[0]) # |00000000000000000000>
    print(states[-1]) # |11111111111111111111>
    return states


# Generate random single qubit unitary  
def random_unitary():
    qc = QuantumCircuit(1)
    qc.u(np.random.random(), np.random.random(), np.random.random(), 0)
    backend = Aer.get_backend('unitary_simulator')
    result = execute(qc, backend).result()
    return result.get_unitary()

# Example
n_qubits = 5
product_state = load_general_product_states(n_qubits)
print(product_state)
