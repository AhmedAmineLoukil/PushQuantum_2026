"""Unencoded (physical-qubit) memory baseline on the same 17 qubits.

The fair comparison for a logical memory is a *bare* qubit that sits through
the same time and the same environment, not the raw parity of the encoded
circuit. Here every one of the 9 data qubits of the code layout stores one of
the cardinal states |0>, |1>, |+>, |->, and for each "round" the circuit

  * idles all qubits for the gate time of one syndrome round (explicit delay),
  * performs the same mid-circuit readout of the 8 ancilla qubits that the
    code performs (so the data qubits see the same readout windows, readout
    crosstalk and server-side DD windows),

but applies no entangling gates. After n rounds the data qubits are measured
in the basis of their state. Each shot therefore gives 9 independent physical
memory samples.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from .code import RotatedSurfaceCode

STATES = ("0", "1", "+", "-")
EXPECTED = {"0": 0, "1": 1, "+": 0, "-": 1}
BASIS = {"0": "Z", "1": "Z", "+": "X", "-": "X"}
STATE_KEY = {"0": "0", "1": "1", "+": "p", "-": "m"}     # file-name safe labels

# Emerald timings from the compiled PulLA schedule (RESULTS.md E4).
PRX_NS = 24.0
CZ_NS = 40.0


def round_gate_time_ns(code: RotatedSurfaceCode | None = None, basis: str = "Z") -> float:
    """Critical-path duration of the gate part of one syndrome round of the
    native-gate code circuit (H layers + CZ layers between two readouts)."""
    from .circuit import build_memory_circuit

    code = code or RotatedSurfaceCode(3)
    c = build_memory_circuit(code.distance, 3, basis, True, None)
    # Durations of the instruction layers between the 1st and 2nd ancilla readout.
    layers, seen_m = [], 0
    for inst in c.flattened():
        if inst.name == "M":
            seen_m += 1
            if seen_m == 2:
                break
            continue
        if seen_m == 1 and inst.name in ("H", "CZ"):
            layers.append(PRX_NS if inst.name == "H" else CZ_NS)
    return float(sum(layers))


def build_physical_baseline(rounds: int, state: str, round_ns: float,
                            code: RotatedSurfaceCode | None = None) -> QuantumCircuit:
    """17-qubit circuit on the code's dense indices (same layout as the code).

    Clbits: 8 per round for the ancilla readouts (round-major, ancillas in
    sorted order, as in the code), then 9 for the data qubits.
    """
    if state not in STATES:
        raise ValueError(state)
    code = code or RotatedSurfaceCode(3)
    data, anc = code.data_qubits, sorted(code.ancillas)
    qc = QuantumCircuit(code.num_qubits, len(anc) * rounds + len(data))
    for q in data:
        if state in ("1", "-"):
            qc.x(q)
        if state in ("+", "-"):
            qc.h(q)
    qc.barrier()
    m = 0
    for _ in range(rounds):
        if round_ns > 0:
            for q in range(code.num_qubits):
                qc.delay(round_ns * 1e-9, q, unit="s")
        qc.barrier()
        for a in anc:
            qc.measure(a, m)
            m += 1
        qc.barrier()
    for q in data:
        if state in ("+", "-"):
            qc.h(q)
    for i, q in enumerate(data):
        qc.measure(q, m + i)
    return qc


def data_errors(raw: np.ndarray, rounds: int, state: str, n_data: int = 9) -> np.ndarray:
    """(shots, n_data) bool: data qubit read out differently from the stored state."""
    data = raw[:, 8 * rounds: 8 * rounds + n_data]
    return data != bool(EXPECTED[state])
