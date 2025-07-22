import hashlib
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import Statevector
from scipy.stats import entropy


def uuid_to_angles(uuid: str, seed: int = 42):
    """Deterministically derive symbolic angles from UUID"""
    np.random.seed(seed)
    h = hashlib.sha256((uuid + str(seed)).encode()).digest()
    phi = (int.from_bytes(h[0:4], 'big') % 628) / 100
    theta = (int.from_bytes(h[4:8], 'big') % 314) / 100
    coords = [(b % 16) for b in h[8:16]]
    return phi, theta, coords, h


def build_qc(n_qubits: int, phi: float, theta: float, entropy_boost=True):
    """Construct a symbolic quantum circuit to represent RGNN entropy or recursion state"""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
        qc.rz(phi * 0.7, i)
        qc.rx(theta * 0.5, i)
        if entropy_boost:
            qc.ry(phi * 0.3, i)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def extract_entropy_state(qc):
    """Simulate a quantum circuit and return entropy and statevector hash"""
    backend = Aer.get_backend("statevector_simulator")
    transpiled = transpile(qc, backend, optimization_level=0)
    job = backend.run(transpiled)
    sv = Statevector(job.result().get_statevector())
    entropy_value = entropy(np.abs(sv.data) ** 2)
    hash_digest = hashlib.sha256(sv.data.tobytes()).hexdigest()
    return entropy_value, hash_digest


def symbolic_entropy_score(uuid: str, qubits=8, seed=42):
    """External interface to RGNN - returns entropy score for a symbolic UUID"""
    phi, theta, _, _ = uuid_to_angles(uuid, seed)
    qc = build_qc(qubits, phi, theta)
    entropy_value, hash_digest = extract_entropy_state(qc)
    return entropy_value, hash_digest
