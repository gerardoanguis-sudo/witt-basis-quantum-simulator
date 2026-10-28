"""Referencias matriciales independientes para la convención de la tesis."""

from pathlib import Path
import runpy

import numpy as np

from wittqsim import WittRegister


def test_generators_match_jordan_wigner_pauli_matrices():
    """La orientación de e_{n+k} debe dar +JW(k) Y_k, no su negativo."""
    w = WittRegister(2)
    identity = np.eye(2)
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.diag([1, -1])
    expected = [
        np.kron(x, identity),
        np.kron(z, x),
        np.kron(y, identity),
        np.kron(z, y),
    ]
    basis = [w.ket(label) for label in w.labels()]
    for k, reference in enumerate(expected, start=1):
        actual = np.column_stack([
            w.statevector(w.e[k] * ket) for ket in basis
        ])
        np.testing.assert_allclose(actual, reference, atol=1e-12, rtol=0)


def test_shor21_matches_inverse_fourier_reference():
    """Compara las 32 amplitudes complejas con la definición de order finding."""
    example = Path(__file__).resolve().parents[1] / "examples" / "shor21_compiled.py"
    result = runpy.run_path(str(example))
    actual = result["w"].statevector(result["psi"])

    # Codificación del trabajo: 1 -> 00, 4 -> 01, 16 -> 10.
    work_index = {1: 0, 4: 1, 16: 2}
    before_fourier = np.zeros((8, 4), dtype=complex)
    for x in range(8):
        before_fourier[x, work_index[pow(4, x, 21)]] = 1 / np.sqrt(8)

    indices = np.arange(8)
    inverse_fourier = np.exp(
        -2j * np.pi * np.outer(indices, indices) / 8
    ) / np.sqrt(8)
    # En esta instancia, R deja invariante el estado previo a Fourier.
    expected = (inverse_fourier @ before_fourier).reshape(32)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
