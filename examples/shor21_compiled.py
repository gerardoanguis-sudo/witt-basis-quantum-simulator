"""
Ejemplo: circuito compilado de Shor N=21, a=4 con 5 qubits.

Qubits:
- 1,2,3: registro de control c0,c1,c2
- 4,5: registro de trabajo w0,w1
"""

from math import pi

from wittqsim import WittRegister


w = WittRegister(5)

gates = [
    # Hadamards en el registro de control
    w.step("H(1)", w.H(1)),
    w.step("H(2)", w.H(2)),
    w.step("H(3)", w.H(3)),

    # U^1
    w.step("CNOT(3,5)", w.CNOT(3, 5)),

    # U^2: orden cronológico
    w.step("CNOT(2,5)", w.CNOT(2, 5)),
    w.step("CNOT(5,4)", w.CNOT(5, 4)),
    w.step("CCNOT(2,4,5)", w.CCNOT(2, 4, 5)),
    w.step("CNOT(5,4)", w.CNOT(5, 4)),

    # U^4: orden cronológico
    w.step("X(5)", w.X(5)),
    w.step("CCNOT(1,5,4)", w.CCNOT(1, 5, 4)),
    w.step("X(5)", w.X(5)),
    w.step("CNOT(5,4)", w.CNOT(5, 4)),
    w.step("CCNOT(1,4,5)", w.CCNOT(1, 4, 5)),
    w.step("CNOT(5,4)", w.CNOT(5, 4)),

    # Bloque Fourier W = F_8^dagger R, en orden cronológico.
    # x = 4*c0 + 2*c1 + c2; R|c0 c1 c2> = |c2 c1 c0>.
    # El estado de entrada de esta instancia es invariante bajo R.
    w.step("H(3)", w.H(3)),
    w.step("CSd(2,3)", w.CSd(2, 3)),
    w.step("CTd(1,3)", w.CTd(1, 3)),
    w.step("H(2)", w.H(2)),
    w.step("CSd(1,2)", w.CSd(1, 2)),
    w.step("H(1)", w.H(1)),
]

psi0 = w.ket("00000")
psi = w.run(psi0, gates)

print("Norma^2:", w.norm2(psi))
print("Probabilidades del registro de control c0 c1 c2:")
probs = w.probabilities(psi, measured=[1, 2, 3])
for label, p in probs.items():
    print(f"{label}: {p:.8f}")
