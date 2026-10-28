from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import pi, sqrt
from typing import Iterable, Sequence, Union, Optional, Dict, List, Tuple

import numpy as np


BitsLike = Union[str, Sequence[int], Sequence[str]]


@dataclass(frozen=True)
class GateStep:
    """Paso de circuito para uso legible en listas cronológicas."""
    name: str
    op: object


class WittRegister:
    """
    Registro de n qubits representado en Cl(2n,C) mediante base de Witt.

    Convención de índices:
    - Los qubits se numeran desde 1 hasta n.
    - El bitstring "0101" significa q1=0, q2=1, q3=0, q4=1.
    - La acción de compuertas se hace por multiplicación izquierda.

    Convención de Witt:
    - f_k  = 1/2 (e_k + i e_{n+k})      aniquilación
    - fd_k = 1/2 (e_k - i e_{n+k})      creación
    - {f_i, fd_j} = delta_ij, {f_i,f_j}=0, {fd_i,fd_j}=0
    """

    def __init__(self, n_qubits: int, *, warn_large: bool = True):
        if not isinstance(n_qubits, int) or n_qubits < 1:
            raise ValueError("n_qubits debe ser un entero positivo.")

        if warn_large and n_qubits > 6:
            # Cl(2n) tiene dimensión 2^(2n). El backend de Clifford completo crece muy rápido.
            print(
                "Advertencia: Cl(2n) tiene dimensión 2^(2n). "
                "Para n > 6 este backend puede volverse pesado."
            )

        try:
            from clifford import Cl
        except ImportError as exc:
            raise ImportError(
                "No se encontró el paquete 'clifford'. Instala con: pip install clifford"
            ) from exc

        self.n_qubits = n_qubits
        self.layout, self.blades = Cl(2 * n_qubits)
        self.i = 1j

        self.e = {
            k: self.blades[f"e{k}"]
            for k in range(1, 2 * n_qubits + 1)
        }

        self.f = {}
        self.fd = {}
        for k in range(1, n_qubits + 1):
            self.f[k] = 0.5 * (self.e[k] + self.i * self.e[n_qubits + k])
            self.fd[k] = 0.5 * (self.e[k] - self.i * self.e[n_qubits + k])

        self.I = self._build_vacuum_idempotent()
        self.I_scalar = self.scalar(self.I)

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------

    def _check_qubit(self, k: int) -> int:
        if not isinstance(k, int) or not (1 <= k <= self.n_qubits):
            raise ValueError(f"Índice inválido: {k}. Usa enteros 1..{self.n_qubits}.")
        return k

    def _check_distinct(self, *qubits: int) -> None:
        if len(set(qubits)) != len(qubits):
            raise ValueError("Los índices de qubits deben ser distintos.")

    def _build_vacuum_idempotent(self):
        out = 1
        for k in range(1, self.n_qubits + 1):
            out = out * self.f[k] * self.fd[k]
        return out

    def _parse_bits(self, bits: BitsLike) -> List[int]:
        if isinstance(bits, str):
            clean = bits.replace(" ", "")
            if len(clean) != self.n_qubits or any(ch not in "01" for ch in clean):
                raise ValueError(
                    f"El bitstring debe tener longitud {self.n_qubits} y solo contener 0/1."
                )
            return [int(ch) for ch in clean]

        out = [int(x) for x in bits]
        if len(out) != self.n_qubits or any(x not in (0, 1) for x in out):
            raise ValueError(
                f"La lista de bits debe tener longitud {self.n_qubits} y valores 0/1."
            )
        return out

    # ------------------------------------------------------------------
    # Generadores y operadores básicos
    # ------------------------------------------------------------------

    def ann(self, k: int):
        """Aniquilador f_k."""
        k = self._check_qubit(k)
        return self.f[k]

    def cre(self, k: int):
        """Creador f_k^dagger."""
        k = self._check_qubit(k)
        return self.fd[k]

    def scalar(self, A) -> complex:
        """Extrae la parte escalar compleja de un multivector."""
        return complex(A[()])

    def proj0(self, k: int):
        """Proyector |0><0|_k = f_k f_k^dagger."""
        k = self._check_qubit(k)
        return self.f[k] * self.fd[k]

    def proj1(self, k: int):
        """Proyector |1><1|_k = n_k = f_k^dagger f_k."""
        return self.N(k)

    def N(self, k: int):
        """Operador número n_k = f_k^dagger f_k."""
        k = self._check_qubit(k)
        return self.fd[k] * self.f[k]

    def Z(self, k: int):
        """Z_k = 1 - 2 n_k."""
        k = self._check_qubit(k)
        return 1 - 2 * self.N(k)

    def JW(self, k: int):
        """Cadena Jordan-Wigner JW(k)=Π_{j<k} Z_j."""
        k = self._check_qubit(k)
        out = 1
        for j in range(1, k):
            out = out * self.Z(j)
        return out

    # ------------------------------------------------------------------
    # Estados, bras, amplitudes y probabilidades
    # ------------------------------------------------------------------

    def ket(self, bits: BitsLike):
        """
        Construye |x1...xn> = (fd_1)^x1 ... (fd_n)^xn I.
        """
        parsed = self._parse_bits(bits)
        out = self.I
        # Para preservar la convención: fd_1 fd_2 ... fd_n I.
        out = self.I
        for k, bit in reversed(list(enumerate(parsed, start=1))):
            # Multiplicar por izquierda en orden ascendente equivale a acumular así:
            # si queremos fd_1 fd_2 I, primero I, luego fd_2*I, luego fd_1*(fd_2 I).
            if bit:
                out = self.fd[k] * out
        return out

    def bra(self, bits: BitsLike):
        """
        Construye el bra dual usado para proyectar amplitudes.

        Si |x> = fd_1^x1 ... fd_n^xn I, entonces
        <x| se representa como I f_n^xn ... f_1^x1.
        """
        parsed = self._parse_bits(bits)
        out = self.I
        for k in range(self.n_qubits, 0, -1):
            if parsed[k - 1]:
                out = out * self.f[k]
        return out

    def amplitude(self, bits: BitsLike, state) -> complex:
        """Amplitud <bits|state> normalizada por la parte escalar de I."""
        return self.scalar(self.bra(bits) * state) / self.I_scalar

    def labels(self) -> List[str]:
        """Etiquetas |00...0>,...,|11...1> en orden lexicográfico binario."""
        return [format(i, f"0{self.n_qubits}b") for i in range(2 ** self.n_qubits)]

    def statevector(self, state) -> np.ndarray:
        """Vector de amplitudes en la base computacional."""
        return np.array([self.amplitude(label, state) for label in self.labels()], dtype=complex)

    def probabilities(
        self,
        state,
        measured: Optional[Sequence[int]] = None,
        *,
        atol: float = 1e-14,
    ) -> Dict[str, float]:
        """
        Probabilidades en la base computacional.

        measured=None devuelve probabilidades completas de n qubits.
        measured=[1,2,3] suma sobre los qubits no medidos y devuelve etiquetas
        en el orden indicado por measured.
        """
        if measured is None:
            probs = {
                label: float(abs(self.amplitude(label, state)) ** 2)
                for label in self.labels()
            }
        else:
            measured = [self._check_qubit(k) for k in measured]
            probs = {"".join(bits): 0.0 for bits in product("01", repeat=len(measured))}
            for label in self.labels():
                key = "".join(label[k - 1] for k in measured)
                probs[key] += float(abs(self.amplitude(label, state)) ** 2)

        # Limpieza numérica
        return {k: (0.0 if abs(v) < atol else v) for k, v in probs.items()}

    def norm2(self, state) -> float:
        """Norma cuadrada calculada en la base computacional."""
        return float(sum(abs(a) ** 2 for a in self.statevector(state)))

    # ------------------------------------------------------------------
    # Compuertas de un qubit
    # ------------------------------------------------------------------

    def X(self, k: int):
        """X_k = JW(k)(f_k + f_k^dagger)."""
        k = self._check_qubit(k)
        return self.JW(k) * (self.f[k] + self.fd[k])

    def Y(self, k: int):
        """Y_k = JW(k)(i f_k^dagger - i f_k)."""
        k = self._check_qubit(k)
        return self.JW(k) * (self.i * self.fd[k] - self.i * self.f[k])

    def H(self, k: int):
        """Hadamard H_k = (X_k + Z_k)/sqrt(2)."""
        k = self._check_qubit(k)
        return (self.X(k) + self.Z(k)) / sqrt(2)

    def phase(self, k: int, theta: float):
        """
        P_k(theta)=diag(1, exp(i theta)).
        Para S† usa theta=-pi/2; para T† usa theta=-pi/4.
        """
        k = self._check_qubit(k)
        return self.proj0(k) + np.exp(1j * theta) * self.proj1(k)

    def phase_minus(self, k: int, theta: float):
        """
        Familia de la tesis: T_k(theta)=diag(1, exp(-i theta)).

        phase_minus(k, pi/2) = Sd(k); phase_minus(k, pi/4) = Td(k).
        """
        return self.phase(k, -theta)

    def S(self, k: int):
        return self.phase(k, pi / 2)

    def Sd(self, k: int):
        return self.phase(k, -pi / 2)

    def T(self, k: int):
        return self.phase(k, pi / 4)

    def Td(self, k: int):
        return self.phase(k, -pi / 4)

    def Rz_diag(self, k: int, theta: float):
        """
        Diagonal D(theta)=diag(exp(i theta), exp(-i theta)).

        Con R_z(phi)=exp(-i phi Z/2), D(theta)=R_z(-2 theta).
        Para la rotación R_z(phi) de la tesis, usar Rz_diag(k, -phi/2).
        """
        k = self._check_qubit(k)
        return np.exp(1j * theta) * self.proj0(k) + np.exp(-1j * theta) * self.proj1(k)

    # ------------------------------------------------------------------
    # Compuertas controladas y de varios qubits
    # ------------------------------------------------------------------

    def Controlled(self, control: int, A_target):
        """
        C(A)_control = |0><0|_control + |1><1|_control A_target.

        A_target debe ser un m-operador que actúa correctamente sobre su target,
        por ejemplo self.X(target), self.Sd(target), self.Td(target), etc.
        """
        control = self._check_qubit(control)
        return self.proj0(control) + self.proj1(control) * A_target

    def CNOT(self, control: int, target: int):
        control = self._check_qubit(control)
        target = self._check_qubit(target)
        self._check_distinct(control, target)
        return self.Controlled(control, self.X(target))

    def CZ(self, control: int, target: int):
        control = self._check_qubit(control)
        target = self._check_qubit(target)
        self._check_distinct(control, target)
        return self.Controlled(control, self.Z(target))

    def CSd(self, control: int, target: int):
        return self.Controlled(control, self.Sd(target))

    def CTd(self, control: int, target: int):
        return self.Controlled(control, self.Td(target))

    def CPhase(self, control: int, target: int, theta: float):
        return self.Controlled(control, self.phase(target, theta))

    def CCNOT(self, control1: int, control2: int, target: int):
        control1 = self._check_qubit(control1)
        control2 = self._check_qubit(control2)
        target = self._check_qubit(target)
        self._check_distinct(control1, control2, target)

        p11 = self.proj1(control1) * self.proj1(control2)
        return (1 - p11) + p11 * self.X(target)

    def SWAP(self, a: int, b: int):
        a = self._check_qubit(a)
        b = self._check_qubit(b)
        self._check_distinct(a, b)
        return 0.5 * (1 + self.X(a) * self.X(b) + self.Y(a) * self.Y(b) + self.Z(a) * self.Z(b))

    # ------------------------------------------------------------------
    # Ejecución de circuitos
    # ------------------------------------------------------------------

    def run(self, initial_state, gates: Sequence[object]):
        """
        Aplica una lista cronológica de compuertas.

        Si gates=[G1,G2,G3], devuelve G3 G2 G1 |psi>.
        """
        state = initial_state
        for gate in gates:
            if isinstance(gate, GateStep):
                op = gate.op
            else:
                op = gate
            state = op * state
        return state

    def compose(self, gates: Sequence[object]):
        """
        Compone una lista cronológica de compuertas.

        Si gates=[G1,G2,G3], devuelve G3 G2 G1.
        """
        op_total = 1
        for gate in gates:
            op = gate.op if isinstance(gate, GateStep) else gate
            op_total = op * op_total
        return op_total

    def step(self, name: str, op) -> GateStep:
        """Crea un paso etiquetado para circuitos legibles."""
        return GateStep(name=name, op=op)

    # ------------------------------------------------------------------
    # Verificación básica de CAR
    # ------------------------------------------------------------------

    def car_errors(self) -> Dict[str, float]:
        """
        Devuelve errores numéricos simples para las relaciones CAR.
        Útil para diagnosticar que la base se construyó bien.
        """
        errors: Dict[str, float] = {}
        for i in range(1, self.n_qubits + 1):
            for j in range(1, self.n_qubits + 1):
                anti = self.f[i] * self.fd[j] + self.fd[j] * self.f[i]
                target = 1 if i == j else 0
                errors[f"{{f{i},fd{j}}}"] = abs(self.scalar(anti - target))
                anti_ff = self.f[i] * self.f[j] + self.f[j] * self.f[i]
                anti_dd = self.fd[i] * self.fd[j] + self.fd[j] * self.fd[i]
                errors[f"{{f{i},f{j}}}"] = abs(self.scalar(anti_ff))
                errors[f"{{fd{i},fd{j}}}"] = abs(self.scalar(anti_dd))
        return errors
