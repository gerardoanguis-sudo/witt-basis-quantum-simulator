"""
wittqsim: simulación de circuitos cuánticos con base de Witt en álgebra de Clifford.

Convención:
- f_k: aniquilación
- fd_k = f_k^dagger: creación
- I = Π_k f_k fd_k representa el vacío |00...0>
- |x_1...x_n> = (fd_1)^x1 ... (fd_n)^xn I
"""

from .clifford_backend import WittRegister, GateStep

__all__ = ["WittRegister", "GateStep"]
