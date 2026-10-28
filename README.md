# wittqsim

Biblioteca para simular circuitos cuánticos usando la base de Witt en el álgebra de Clifford compleja.

## Convención

- `f_k = (e_k + i e_{n+k})/2`: aniquilación.
- `fd_k = f_k^dagger = (e_k - i e_{n+k})/2`: creación.
- `I = Π_k f_k fd_k`: vacío `|00...0>`.
- `|x_1...x_n> = (fd_1)^x1 ... (fd_n)^xn I`.
- `<x_1...x_n| = I (f_n)^xn ... (f_1)^x1`.
- Los qubits se indexan desde `1` hasta `n`.

En la base de ocupación de un modo, `f = |0><1| = (X+iY)/2` y
`fd = |1><0| = (X-iY)/2`. El operador número es `fd*f`, y
`Z = 1 - 2*fd*f`. Para varios modos, las cadenas de Jordan-Wigner
dan `e_k = JW(k) X_k` y `e_{n+k} = JW(k) Y_k`.

La versión `0.1.1` adopta los signos de la tesis. La versión `0.1.0`
utilizaba los signos opuestos. Los estados y operadores deben reconstruirse
con el mismo `WittRegister` de la versión actual: sus coeficientes en la base
de blades cambian, aunque las amplitudes y probabilidades de los circuitos
construidos consistentemente se conservan.

## Instalación local

```bash
git clone https://github.com/gerardoanguis-sudo/witt-basis-quantum-simulator.git
cd witt-basis-quantum-simulator
python -m pip install -e .
```

Para recuperar una revisión citada en la tesis, selecciona ese commit con
`git checkout` antes de instalar el paquete.

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
- `phase_minus(k, theta) = diag(1, exp(-i theta))`
- `S(k), Sd(k), T(k), Td(k)`
- `Rz_diag(k, theta) = diag(exp(i theta), exp(-i theta))`
- `CNOT(control, target)`
- `CZ(control, target)`
- `CSd(control, target)`
- `CTd(control, target)`
- `CPhase(control, target, theta)`
- `CCNOT(control1, control2, target)`
- `SWAP(a, b)`

### Correspondencia con las fases de la tesis

| Notación de la tesis | Función de la biblioteca |
| --- | --- |
| `T_k(theta) = diag(1, exp(-i theta))` | `phase_minus(k, theta)` |
| `S_k^dagger = T_k(pi/2)` | `Sd(k)` |
| `T_k^dagger = T_k(pi/4)` | `Td(k)` |
| `R_z,k(theta) = exp(-i theta Z_k/2)` | `Rz_diag(k, -theta/2)` |

La relación de fases es `T_k(theta) = exp(-i theta/2) R_z,k(-theta)`.
Las compuertas controladas se construyen directamente con proyectores,
conservando la fase relativa entre las ramas del control.

## Circuito Shor N=21, a=4

```bash
python examples/shor21_compiled.py
```

El ejemplo calcula las probabilidades marginales directamente, sin muestreo
con shots, y presenta las etiquetas en el orden `c0 c1 c2`:

```text
000: 0.34375000
001: 0.01451457
010: 0.06250000
011: 0.23548543
100: 0.03125000
101: 0.23548543
110: 0.06250000
111: 0.01451457
```

Con `x = 4*c0 + 2*c1 + c2`, las seis compuertas finales implementan
`W = F_8^dagger R`, donde `R` intercambia `c0` y `c2` en la entrada.
El estado previo a ese bloque es invariante bajo `R` para esta instancia,
porque `4^x mod 21` tiene período tres. Por ello el ejemplo produce el mismo
estado final que `F_8^dagger`, aunque ambos operadores se distinguen sobre
entradas generales. Esta propiedad no cambia el orden de las etiquetas
de medición.

## Entorno verificado y pruebas

La revisión `0.1.1` se validó con CPython `3.12.14` en Linux `x86_64`.
`requirements-verified.txt` fija las versiones de las dependencias y de las
herramientas de prueba de ese entorno. Este registro corresponde a la
validación de esta revisión; no identifica el entorno original del notebook
de la tesis ni incluye la comparación mediante Qiskit.

Desde la raíz de la revisión seleccionada, crea y activa un entorno virtual:

```bash
python -m venv .venv
```

La activación es `source .venv/bin/activate` en Linux/macOS o
`.venv\Scripts\Activate.ps1` en PowerShell. Después ejecuta:

```bash
python -m pip install -r requirements-verified.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest -q
python examples/shor21_compiled.py
```

Las pruebas incluyen la representación de los generadores mediante matrices
de Pauli con cadenas de Jordan-Wigner. También comparan las 32 amplitudes
complejas del ejemplo de Shor con una referencia independiente construida
mediante la exponenciación modular y la matriz de Fourier inversa.

## Nota sobre escalabilidad

Este backend construye `Cl(2n)`, cuya dimensión completa es `2^(2n)`. Es muy útil para tu formulación de tesis y validaciones pequeñas, pero no será eficiente para muchos qubits. Para escalar, conviene añadir después un backend matricial/sparse que conserve la misma interfaz.
