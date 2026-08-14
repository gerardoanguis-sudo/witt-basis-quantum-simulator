from wittqsim import WittRegister

w = WittRegister(2)

psi0 = w.ket("00")

circuit = [
    w.H(1),
    w.CNOT(1, 2),
]

psi = w.run(psi0, circuit)

print("Norma^2:", w.norm2(psi))
print("Probabilidades:")
print(w.probabilities(psi))
