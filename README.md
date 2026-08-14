# wittqsim

Librería inicial para simular circuitos cuánticos usando base de Witt en álgebra de Clifford.

## Convención

- `f_k`: aniquilación.
- `fd_k = f_k^\dagger`: creación.
- `I = Π_k f_k fd_k`: vacío `|00...0>`.
- `|x_1...x_n> = (fd_1)^x1 ... (fd_n)^xn I`.
- Los qubits se indexan desde `1` hasta `n`.

## Instalación local

```bash
cd wittqsim
python -m pip install -e .
```

## Ejemplo mínimo

```python
from wittqsim import WittRegister

w = WittRegister(3)
psi0 = w.ket("000")

circuit = [
    w.H(1),
    w.CNOT(1, 2),
    w.CNOT(2, 3),
]

psi = w.run(psi0, circuit)
print(w.probabilities(psi))
```

## Compuertas incluidas

- `X(k), Y(k), Z(k), H(k)`
- `phase(k, theta) = diag(1, exp(i theta))`
- `S(k), Sd(k), T(k), Td(k)`
- `CNOT(control, target)`
- `CZ(control, target)`
- `CSd(control, target)`
- `CTd(control, target)`
- `CPhase(control, target, theta)`
- `CCNOT(control1, control2, target)`
- `SWAP(a, b)`

## Circuito Shor N=21, a=4

```bash
python examples/shor21_compiled.py
```

## Nota sobre escalabilidad

Este backend construye `Cl(2n)`, cuya dimensión completa es `2^(2n)`. Es muy útil para tu formulación de tesis y validaciones pequeñas, pero no será eficiente para muchos qubits. Para escalar, conviene añadir después un backend matricial/sparse que conserve la misma interfaz.
