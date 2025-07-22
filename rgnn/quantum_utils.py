"""
quantum_utils.py

Utilities for symbolic entropy computation, entanglement modeling,
and recursive verification steps used in RGNN-QASM bridging.
"""

import numpy as np
import torch
from scipy.stats import entropy
import hashlib


def von_neumann_entropy(tensor, eps=1e-10):
    """
    Calculate von Neumann entropy of a complex tensor interpreted as a quantum state
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    prob = np.abs(tensor.flatten()) ** 2
    prob = prob[prob > eps]
    prob /= np.sum(prob)
    return float(entropy(prob))


def entangle_tensors(t1, t2, mode='cx'):
    """
    Simulate entanglement between two tensors via simple transformation
    """
    assert t1.shape == t2.shape, "Tensor shapes must match for entanglement"

    if mode == 'cx':
        return t1 * torch.exp(1j * t2)
    elif mode == 'sum':
        return (t1 + t2) / 2
    elif mode == 'xor':
        return torch.sin(t1) * torch.cos(t2)
    else:
        raise ValueError(f"Unknown entanglement mode: {mode}")


def hash_tensor(tensor, precision=4):
    """
    Generate a SHA-256 hash of a tensor for Merkle-style certification
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    real = np.round(tensor.real, precision)
    imag = np.round(tensor.imag, precision)
    return hashlib.sha256(real.tobytes() + imag.tobytes()).hexdigest()


def amplitude_map(tensor, normalize=True):
    """
    Return amplitude magnitudes for visualization or encoding
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    amps = np.abs(tensor.flatten())
    if normalize:
        amps /= amps.max() + 1e-10
    return amps


def collapse_measure(t1, t2):
    """
    Compute similarity between two quantum states (fidelity-style measure)
    """
    if isinstance(t1, torch.Tensor): t1 = t1.detach().cpu().numpy()
    if isinstance(t2, torch.Tensor): t2 = t2.detach().cpu().numpy()
    
    dot = np.vdot(t1.flatten(), t2.flatten())
    return np.abs(dot) ** 2  # Fidelity-style
