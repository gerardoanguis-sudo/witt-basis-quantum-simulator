import pytest
import numpy as np

from wittqsim import WittRegister


def assert_prob_one(probs, label, tol=1e-10):
    for k, v in probs.items():
        if k == label:
            assert abs(v - 1.0) < tol, f"{label} debería tener probabilidad 1, pero tiene {v}"
        else:
            assert abs(v) < tol, f"{k} debería tener probabilidad 0, pero tiene {v}"


def assert_probs_close(probs, expected, tol=1e-10):
    for k, v in expected.items():
        assert abs(probs[k] - v) < tol, f"{k}: esperado {v}, obtenido {probs[k]}"


# ------------------------------------------------------------
# 1. Estados base
# ------------------------------------------------------------

def test_basis_states_are_correct():
    w = WittRegister(3)

    for label in w.labels():
        psi = w.ket(label)
        probs = w.probabilities(psi)
        assert_prob_one(probs, label)


# ------------------------------------------------------------
# 2. Compuertas de un qubit
# ------------------------------------------------------------

def test_x_on_each_qubit():
    w = WittRegister(3)

    psi = w.X(1) * w.ket("000")
    assert_prob_one(w.probabilities(psi), "100")

    psi = w.X(2) * w.ket("000")
    assert_prob_one(w.probabilities(psi), "010")

    psi = w.X(3) * w.ket("000")
    assert_prob_one(w.probabilities(psi), "001")


def test_z_signs():
    w = WittRegister(1)

    psi0 = w.Z(1) * w.ket("0")
    psi1 = w.Z(1) * w.ket("1")

    assert abs(w.amplitude("0", psi0) - 1) < 1e-10
    assert abs(w.amplitude("1", psi1) + 1) < 1e-10


def test_y_action():
    w = WittRegister(1)

    psi0 = w.Y(1) * w.ket("0")
    psi1 = w.Y(1) * w.ket("1")

    # Convención usual:
    # Y|0> = i|1>
    # Y|1> = -i|0>
    assert abs(w.amplitude("1", psi0) - 1j) < 1e-10
    assert abs(w.amplitude("0", psi1) + 1j) < 1e-10


def test_hadamard_on_zero_and_one():
    w = WittRegister(1)

    psi0 = w.H(1) * w.ket("0")
    psi1 = w.H(1) * w.ket("1")

    assert abs(w.amplitude("0", psi0) - 1 / np.sqrt(2)) < 1e-10
    assert abs(w.amplitude("1", psi0) - 1 / np.sqrt(2)) < 1e-10

    assert abs(w.amplitude("0", psi1) - 1 / np.sqrt(2)) < 1e-10
    assert abs(w.amplitude("1", psi1) + 1 / np.sqrt(2)) < 1e-10


def test_phase_gates():
    w = WittRegister(1)

    psi = w.S(1) * w.ket("1")
    assert abs(w.amplitude("1", psi) - 1j) < 1e-10

    psi = w.Sd(1) * w.ket("1")
    assert abs(w.amplitude("1", psi) + 1j) < 1e-10

    psi = w.T(1) * w.ket("1")
    assert abs(w.amplitude("1", psi) - np.exp(1j * np.pi / 4)) < 1e-10

    psi = w.Td(1) * w.ket("1")
    assert abs(w.amplitude("1", psi) - np.exp(-1j * np.pi / 4)) < 1e-10


# ------------------------------------------------------------
# 3. Compuertas controladas
# ------------------------------------------------------------

def test_cnot_truth_table():
    w = WittRegister(2)

    cases = {
        "00": "00",
        "01": "01",
        "10": "11",
        "11": "10",
    }

    for initial, final in cases.items():
        psi = w.CNOT(1, 2) * w.ket(initial)
        assert_prob_one(w.probabilities(psi), final)


def test_cnot_reverse_direction():
    w = WittRegister(2)

    cases = {
        "00": "00",
        "01": "11",
        "10": "10",
        "11": "01",
    }

    for initial, final in cases.items():
        psi = w.CNOT(2, 1) * w.ket(initial)
        assert_prob_one(w.probabilities(psi), final)


def test_cz_truth_table():
    w = WittRegister(2)

    for label in ["00", "01", "10"]:
        psi = w.CZ(1, 2) * w.ket(label)
        assert abs(w.amplitude(label, psi) - 1) < 1e-10

    psi = w.CZ(1, 2) * w.ket("11")
    assert abs(w.amplitude("11", psi) + 1) < 1e-10


def test_ccnot_truth_table():
    w = WittRegister(3)

    cases = {
        "000": "000",
        "001": "001",
        "010": "010",
        "011": "011",
        "100": "100",
        "101": "101",
        "110": "111",
        "111": "110",
    }

    for initial, final in cases.items():
        psi = w.CCNOT(1, 2, 3) * w.ket(initial)
        assert_prob_one(w.probabilities(psi), final)


def test_swap_truth_table():
    w = WittRegister(2)

    cases = {
        "00": "00",
        "01": "10",
        "10": "01",
        "11": "11",
    }

    for initial, final in cases.items():
        psi = w.SWAP(1, 2) * w.ket(initial)
        assert_prob_one(w.probabilities(psi), final)


# ------------------------------------------------------------
# 4. Circuitos conocidos
# ------------------------------------------------------------

def test_bell_state():
    w = WittRegister(2)

    psi = w.run(w.ket("00"), [
        w.H(1),
        w.CNOT(1, 2),
    ])

    probs = w.probabilities(psi)

    assert_probs_close(probs, {
        "00": 0.5,
        "01": 0.0,
        "10": 0.0,
        "11": 0.5,
    })


def test_ghz_state_three_qubits():
    w = WittRegister(3)

    psi = w.run(w.ket("000"), [
        w.H(1),
        w.CNOT(1, 2),
        w.CNOT(2, 3),
    ])

    probs = w.probabilities(psi)

    assert_probs_close(probs, {
        "000": 0.5,
        "001": 0.0,
        "010": 0.0,
        "011": 0.0,
        "100": 0.0,
        "101": 0.0,
        "110": 0.0,
        "111": 0.5,
    })


def test_x_then_x_is_identity():
    w = WittRegister(3)

    for label in w.labels():
        psi = w.run(w.ket(label), [
            w.X(1),
            w.X(1),
        ])
        assert_prob_one(w.probabilities(psi), label)


def test_h_then_h_is_identity():
    w = WittRegister(2)

    for label in w.labels():
        psi = w.run(w.ket(label), [
            w.H(1),
            w.H(1),
        ])
        assert_prob_one(w.probabilities(psi), label)


def test_cnot_then_cnot_is_identity():
    w = WittRegister(2)

    for label in w.labels():
        psi = w.run(w.ket(label), [
            w.CNOT(1, 2),
            w.CNOT(1, 2),
        ])
        assert_prob_one(w.probabilities(psi), label)


# ------------------------------------------------------------
# 5. Conservación de norma
# ------------------------------------------------------------

def test_norm_preservation_small_circuit():
    w = WittRegister(4)

    psi = w.run(w.ket("0000"), [
        w.H(1),
        w.H(2),
        w.CNOT(1, 3),
        w.CNOT(2, 4),
        w.T(3),
        w.Sd(4),
        w.CZ(3, 4),
        w.H(1),
    ])

    assert abs(w.norm2(psi) - 1.0) < 1e-10


# ------------------------------------------------------------
# 6. Casos límite / errores esperados
# ------------------------------------------------------------

def test_invalid_qubit_index_raises_error():
    w = WittRegister(2)

    with pytest.raises(ValueError):
        w.X(0)

    with pytest.raises(ValueError):
        w.X(3)

    with pytest.raises(ValueError):
        w.CNOT(1, 1)


def test_invalid_ket_raises_error():
    w = WittRegister(3)

    with pytest.raises(ValueError):
        w.ket("00")

    with pytest.raises(ValueError):
        w.ket("0000")

    with pytest.raises(ValueError):
        w.ket("02A")
