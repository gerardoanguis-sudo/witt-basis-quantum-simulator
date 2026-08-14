"""
Pruebas mínimas. Requieren instalar clifford.

Ejecuta:
    python -m pytest tests
"""

import numpy as np

from wittqsim import WittRegister


def test_x_flips_one_qubit():
    w = WittRegister(2)
    psi = w.X(1) * w.ket("00")
    assert abs(w.amplitude("10", psi) - 1) < 1e-10


def test_cnot():
    w = WittRegister(2)
    psi = w.CNOT(1, 2) * w.ket("10")
    assert abs(w.amplitude("11", psi) - 1) < 1e-10


def test_h_norm():
    w = WittRegister(1)
    psi = w.H(1) * w.ket("0")
    probs = w.probabilities(psi)
    assert abs(probs["0"] - 0.5) < 1e-10
    assert abs(probs["1"] - 0.5) < 1e-10
