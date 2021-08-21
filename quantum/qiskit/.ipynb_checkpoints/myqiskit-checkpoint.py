from qiskit import assemble, Aer # QuantumCircuit,
from math import pi # sqrt
from qiskit.visualization import plot_bloch_multivector, plot_histogram

def show_bloch(qc):
    # Let's see the result
    svsim = Aer.get_backend('statevector_simulator')
    qobj = assemble(qc)
    state = svsim.run(qobj).result().get_statevector()
    return plot_bloch_multivector(state)

def show_hist(qc):
    """
    Return a history figure plotted from counts.

    Args: 
      qc (QuantumCircuit): a quantum circuit.

    Returns:
      plot_histogram(counts)
    """
    qasmsim = Aer.get_backend('qasm_simulator')  # Tell Qiskit how to simulate our circuit
    qobj = assemble(qc)  # Assemble circuit into a Qobj that can be run
    counts = qasmsim.run(qobj).result().get_counts()  # Do the simulation, returning the state vector
    return plot_histogram(counts)  # Display the output state vector

def x_measurement(qc, qubit, cbit):
    qc.h(qubit)
    qc.measure(qubit, cbit)
    return qc