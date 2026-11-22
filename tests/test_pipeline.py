"""Correctness checks for the whole offline pipeline (no token needed).

Run:  .venv/Scripts/python -m pytest -q
"""

import itertools

import numpy as np
import pytest
import stim
from qiskit_aer import AerSimulator

from qec import NoiseModel, RotatedSurfaceCode, build_memory_circuit
from qec.analysis import fit_ler, p_logical_model
from qec.decode import decode, measurements_to_detections
from qec.hardware import coupling_edges, memory_to_array, stim_to_qiskit
from qec.layout import find_embeddings, is_native

VARIANTS = list(itertools.product([3, 5], "ZX", [True, False], ["cz", "cx"]))


@pytest.mark.parametrize("d,basis,no_reset,native", VARIANTS)
def test_detectors_deterministic_and_full_distance(d, basis, no_reset, native):
    c = build_memory_circuit(d, 3, basis, no_reset, NoiseModel.uniform(1e-3), native)
    # detector_error_model raises if any detector/observable is non-deterministic,
    # which is exactly what a wrong no-reset detector definition would cause.
    c.detector_error_model(decompose_errors=True)
    assert len(c.shortest_graphlike_error()) == d
    assert c.num_observables == 1


@pytest.mark.parametrize("basis", "ZX")
def test_no_reset_detectors_match_reset_detectors(basis):
    """Same physical error -> same detection events, with or without reset.

    The detection events are the syndrome differences s_r XOR s_{r-1}; they
    must not depend on whether the ancilla was reset, only the raw records do.
    """
    for q, after_round in [(4, 1), (0, 2), (7, 0)]:
        dets = []
        for no_reset in (True, False):
            c = build_memory_circuit(3, 4, basis, no_reset)
            err_basis = "Y_ERROR"  # anticommutes with both stabilizer types, any frame
            c = _inject_after_measurement(c, after_round, err_basis, q)
            s = c.compile_detector_sampler().sample(1, separate_observables=True)
            dets.append(np.concatenate([s[0][0], s[1][0]]))
        assert np.array_equal(dets[0], dets[1])
        assert dets[0][:-1].any()


def _inject_after_measurement(c: stim.Circuit, k: int, gate: str, q: int) -> stim.Circuit:
    """Insert a deterministic Pauli on qubit q after the k-th measurement layer
    (k=0 -> right after initialisation)."""
    out, seen = stim.Circuit(), 0
    done = False
    for inst in c.flattened():
        out.append(inst)
        if not done and k == 0 and inst.name == "R":
            out.append(gate, [q], 1.0)
            done = True
        if inst.name == "M":
            seen += 1
            if seen == k and not done:
                out.append(gate, [q], 1.0)
                done = True
    return out


@pytest.mark.parametrize("basis,rounds", [("Z", 1), ("Z", 3), ("X", 2)])
def test_qiskit_translation_matches_stim(basis, rounds):
    """Run the translated Qiskit circuit in Aer: noiseless -> no detections,
    and a deliberately injected error gives the same detectors as Stim."""
    c = build_memory_circuit(3, rounds, basis, True)
    qc = stim_to_qiskit(c, barriers=False)
    sim = AerSimulator(method="stabilizer")
    mem = sim.run(qc, shots=64, memory=True, seed_simulator=1).result().get_memory()
    det, obs = measurements_to_detections(memory_to_array(mem, c.num_measurements), c)
    assert not det.any() and not obs.any()

    # Same single error in both simulators (bit flip on data qubit 4 after round 1).
    gate = "Y_ERROR"
    c_err = _inject_after_measurement(c, 1, gate, 4)
    qc_err = _qiskit_with_error(c, 1, 4, "y")
    mem = sim.run(qc_err, shots=8, memory=True, seed_simulator=1).result().get_memory()
    det_q, obs_q = measurements_to_detections(memory_to_array(mem, c.num_measurements), c)
    det_s, obs_s = c_err.compile_detector_sampler().sample(1, separate_observables=True)
    assert (det_q == det_s[0]).all() and (obs_q == obs_s[0]).all()


def _qiskit_with_error(c, k, q, pauli):
    from qiskit import QuantumCircuit
    qc = stim_to_qiskit(c, barriers=False)
    out = QuantumCircuit(*qc.qregs, *qc.cregs)
    seen_layers, prev_meas = 0, False
    inserted = False
    for inst in qc.data:
        is_meas = inst.operation.name == "measure"
        if prev_meas and not is_meas:
            seen_layers += 1
            if seen_layers == k and not inserted:
                getattr(out, pauli)(q)
                inserted = True
        out.append(inst)
        prev_meas = is_meas
    return out


def test_layout_is_native_on_emerald():
    code = RotatedSurfaceCode(3)
    edges = coupling_edges()
    embs = find_embeddings(code, edges)
    assert len(embs) > 0
    assert all(is_native(code, e, edges) for e in embs)
    # distance 5 needs 49 of 54 qubits in a full diamond: not SWAP-free on Emerald
    assert find_embeddings(RotatedSurfaceCode(5), edges) == []


def test_simulated_ler_and_fit():
    noise = NoiseModel.uniform(2e-3)
    rounds, errs, shots = [1, 3, 5, 7], [], []
    for r in rounds:
        c = build_memory_circuit(3, r, "Z", True, noise)
        det, obs = c.compile_detector_sampler(seed=3).sample(40_000, separate_observables=True)
        res = decode(det, obs, c)
        errs.append(res.errors)
        shots.append(res.shots)
        assert res.p_logical < decode(det, obs, c, "none").p_logical
    fit = fit_ler(rounds, errs, shots)
    assert 5e-4 < fit.eps < 1e-2
    # fit reproduces the data within a few sigma
    p = np.array(errs) / np.array(shots)
    sig = np.sqrt(p * (1 - p) / np.array(shots))
    assert np.all(np.abs(p - p_logical_model(rounds, fit.eps)) < 5 * sig + 1e-3)


def test_fit_recovers_known_rate():
    rng = np.random.default_rng(0)
    rounds = np.arange(1, 11)
    n = 20_000
    k = rng.binomial(n, p_logical_model(rounds, 0.03, 0.95))
    f = fit_ler(rounds, k, [n] * len(rounds), free_amplitude=True)
    assert abs(f.eps - 0.03) < 4 * f.eps_err
    assert abs(f.amp - 0.95) < 4 * f.amp_err + 0.01


def test_learned_weights_recover_hot_ancilla():
    """p_ij learning finds a deliberately bad ancilla and beats a wrong prior."""
    from qec.learn import cross_validated_decode, summarize_learned

    code = RotatedSurfaceCode(3)
    true = NoiseModel.si1000(4e-3)
    hot = code.stabilizers[3].ancilla
    true.p_meas_q[hot] = 0.25
    c_true = build_memory_circuit(3, 3, "Z", True, true)
    c_wrong = build_memory_circuit(3, 3, "Z", True, NoiseModel.uniform(4e-3))
    det, obs = c_true.compile_detector_sampler(seed=5).sample(20_000, separate_observables=True)
    res, learned = cross_validated_decode(det, obs, c_wrong)
    names = {str(q): f"q{q}" for q in range(code.num_qubits)}
    rates = {r["stabilizer"]: r["timelike_learned"]
             for r in summarize_learned(c_wrong, learned, code, names)}
    hot_name = next(k for k in rates if k.endswith(f"q{hot}"))
    assert abs(rates[hot_name] - 0.25) < 0.03
    assert max(v for k, v in rates.items() if k != hot_name) < 0.06
    assert res.p_logical < 0.8 * decode(det, obs, c_wrong).p_logical


def test_scaffold_entry_points():
    """The challenge's function names work end to end on simulated shots."""
    import surface_code as sc

    rounds, pl, shots = [1, 2, 3, 4], [], []
    for r in rounds:
        noisy = sc.make_stim_circuit(3, r, "Z", no_reset=True, noise=sc.DEFAULT_NOISE)
        clean = sc.make_stim_circuit(3, r, "Z", no_reset=True, noise=None)
        raw = noisy.compile_sampler(seed=r).sample(20_000)
        syn = sc.extract_syndromes(raw, clean) | {"rounds": r, "basis": "Z"}
        ler, err = sc.decode_hardware_results(syn, noise=sc.DEFAULT_NOISE)
        assert 0 < ler < 0.1 and err > 0
        pl.append(ler)
        shots.append(20_000)
    eps, eps_err = sc.fit_logical_error_rate(rounds, pl, shots)
    assert 5e-4 < eps < 2e-2 and eps_err > 0

    qc = sc.make_qiskit_circuit(3, 2, "X", no_reset=True)
    assert qc.num_qubits == 17 and qc.num_clbits == 2 * 8 + 9
    assert sc.get_circuit_info(sc.make_stim_circuit(3, 2, noise=sc.DEFAULT_NOISE))["circuit_distance"] == 3
    clean = sc.make_stim_circuit(3, 1)
    assert sc.print_qubit_map(clean, sc.build_emerald_qubit_map(clean))


def test_calibration_file_resolves_across_machines():
    """Runs saved on another machine store an absolute foreign path; it must
    resolve to the copy in data/ (and a missing file must give None)."""
    from qec.hardware import resolve_calibration_file, run_calibration
    foreign = r"C:\Users\someone\QEC-Emerald\data\calibration_emerald_run20261121-151940.json"
    p = resolve_calibration_file(foreign)
    assert p is not None and p.name == "calibration_emerald_run20261121-151940.json"
    assert run_calibration({"calibration_file": foreign})["readout"]
    assert resolve_calibration_file(r"C:\nowhere\calibration_does_not_exist.json") is None
    with pytest.raises(FileNotFoundError):
        run_calibration({"calibration_file": "nope.json"}, required=True)


def test_fit_goodness_and_profile_interval():
    rng = np.random.default_rng(1)
    rounds = np.arange(1, 9)
    n = 4000
    k = rng.binomial(n, p_logical_model(rounds, 0.03))
    f = fit_ler(rounds, k, [n] * len(rounds))
    assert f.ci68[0] < f.eps < f.ci68[1] and f.ci95[0] < f.ci68[0] and f.ci68[1] < f.ci95[1]
    assert abs((f.ci68[1] - f.ci68[0]) / 2 - f.eps_err) < 0.2 * f.eps_err   # ~Gaussian here
    assert f.model_ok and f.p_gof > 0.01
    # A curve with a large first-round offset is not exponential with A = 1.
    k_bad = rng.binomial(n, np.r_[0.15, p_logical_model(rounds[1:], 0.03)])
    assert not fit_ler(rounds, k_bad, [n] * len(rounds)).model_ok


def test_independent_detectors_match_stim():
    """The numpy re-derivation used by reproduce_all.py agrees with Stim's
    DETECTOR definitions on noisy samples (both bases)."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "reproduce_all", Path(__file__).resolve().parent.parent / "experiments" / "reproduce_all.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for basis in "ZX":
        for r in (1, 2, 5):
            noisy = build_memory_circuit(3, r, basis, True, NoiseModel.si1000(0.01))
            clean = build_memory_circuit(3, r, basis, True, None)
            raw = noisy.compile_sampler(seed=r).sample(2000)
            det, obs = measurements_to_detections(raw, clean)
            d2, o2 = mod.numpy_detectors(raw, r, basis)
            assert np.array_equal(det, d2) and np.array_equal(obs[:, 0], o2)


@pytest.mark.parametrize("state", ["0", "1", "+", "-"])
def test_physical_baseline_circuit(state):
    """Noiseless: every data qubit reads back its stored state; no entangling
    gates; ancillas are read out once per round like in the code."""
    from qec.physical import build_physical_baseline, data_errors, round_gate_time_ns
    ns = round_gate_time_ns()
    assert 150 < ns < 600
    for r in (0, 1, 3):
        qc = build_physical_baseline(r, state, ns)
        assert qc.num_clbits == 8 * r + 9
        assert all(inst.operation.num_qubits == 1 or inst.operation.name == "barrier"
                   for inst in qc.data)
        mem = AerSimulator(method="stabilizer").run(qc, shots=32, memory=True,
                                                    seed_simulator=2).result().get_memory()
        assert not data_errors(memory_to_array(mem, qc.num_clbits), r, state).any()


@pytest.mark.parametrize("basis", "ZX")
def test_pauli_twirl_is_exact(basis):
    """A twirled circuit has the same ideal behaviour as the untwirled one:
    sampled noiselessly and converted with the UNTWIRLED reference (as the
    hardware data are), no detector or observable ever fires - in Stim and
    after translation to Qiskit (Aer). Seeds give different circuits, and the
    twirl is fenced by barriers so the transpiler cannot cancel it."""
    ref = build_memory_circuit(3, 3, basis, True, None)
    circuits = []
    for seed in range(4):
        tw = build_memory_circuit(3, 3, basis, True, None, twirl_seed=seed)
        circuits.append(str(tw))
        det, obs = measurements_to_detections(tw.compile_sampler(seed=seed).sample(300), ref)
        assert not det.any() and not obs.any()
        assert tw.num_detectors == ref.num_detectors
    assert len(set(circuits)) == 4
    qc = stim_to_qiskit(build_memory_circuit(3, 2, basis, True, None, twirl_seed=9), barriers=False)
    assert qc.count_ops().get("barrier", 0) == 2 * 4 * 2          # 2 per CZ layer, 4 layers, 2 rounds
    ref2 = build_memory_circuit(3, 2, basis, True, None)
    mem = AerSimulator(method="stabilizer").run(qc, shots=64, memory=True,
                                                seed_simulator=1).result().get_memory()
    det, obs = measurements_to_detections(memory_to_array(mem, ref2.num_measurements), ref2)
    assert not det.any() and not obs.any()
    # Same Pauli error, twirled or not: identical detection events.
    noisy_ref = _inject_after_measurement(ref, 1, "Y_ERROR", 4)
    noisy_tw = _inject_after_measurement(build_memory_circuit(3, 3, basis, True, None, twirl_seed=5),
                                         1, "Y_ERROR", 4)
    a = measurements_to_detections(noisy_ref.compile_sampler(seed=1).sample(8), ref)
    b = measurements_to_detections(noisy_tw.compile_sampler(seed=1).sample(8), ref)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_soft_information_decoding():
    """Synthetic IQ readout: the per-shot misread probabilities are calibrated,
    every record maps to one graph edge, and soft decoding beats the same graph
    with average (hard-decision) weights on identical shots."""
    from qec.soft import SoftMatcher, fit_iq_model, record_effects, without_misreads
    rng = np.random.default_rng(0)
    base = NoiseModel.si1000(0.004)
    base.p_meas = 0.0
    clean = build_memory_circuit(3, 3, "Z", True, None)
    assert len(record_effects(clean)) == clean.num_measurements
    m = build_memory_circuit(3, 3, "Z", True, base).compile_sampler(seed=3).sample(8000)
    sig = 0.24
    noise = lambda shape: rng.normal(0, sig, shape) + 1j * rng.normal(0, sig, shape)  # noqa: E731
    iq = m + noise(m.shape)
    bits = iq.real > 0.5
    model = fit_iq_model(noise((3000, m.shape[1])), 1 + noise((3000, m.shape[1])))
    p = model.misread_probability(iq, bits)
    assert abs(p.mean() - (bits != m).mean()) < 0.003
    det, obs = measurements_to_detections(bits, clean)
    sm = SoftMatcher(build_memory_circuit(3, 3, "Z", True, without_misreads(base)), clean)
    hard = (sm.decode(det, p.mean(axis=0)) != obs).any(1).mean()
    soft = (sm.decode(det, p) != obs).any(1).mean()
    assert soft < 0.85 * hard


def test_calibrated_final_readout_prior():
    """Per-qubit calibrated readout errors replace the generic readout only on
    the given (data) qubits, as a classical misread, and lower decoding error
    when they are the truth."""
    code = RotatedSurfaceCode(3)
    names = {q: f"QB{q + 1}" for q in range(code.num_qubits)}
    errs = {f"QB{q + 1}": 0.002 for q in code.data_qubits}
    base = NoiseModel.si1000(0.004)
    m = base.with_readout_errors(errs, names, code.data_qubits)
    assert all(m.meas(q) == 0 and m.meas_record(q) == 0.002 for q in code.data_qubits)
    assert all(m.meas(a) == base.meas(a) and m.meas_record(a) == 0 for a in code.ancillas)
    truth = build_memory_circuit(3, 2, "Z", True, m)
    det, obs = truth.compile_detector_sampler(seed=4).sample(40_000, separate_observables=True)
    right = decode(det, obs, truth).p_logical
    wrong = decode(det, obs, build_memory_circuit(3, 2, "Z", True, base)).p_logical
    assert right <= wrong + 0.002


def test_final_readout_choice_only_touches_final_data_measurements():
    """'fidelity' final readout swaps only the final data measurements to the
    shelved high-fidelity readout; mid-circuit ancilla readouts stay QND and
    every measurement keeps its classical-bit key."""
    pytest.importorskip("iqm.pulse")
    from iqm.pulse import Circuit, CircuitOperation
    from qec.hardware import with_final_readout
    rounds, n = 2, 2 * 8 + 9
    ops = [CircuitOperation(name="prx", locus=("QB1",), args={"angle": 1.0, "phase": 0.0})]
    ops += [CircuitOperation(name="measure", locus=(f"QB{j}",), args={"key": f"c_{n}_0_{j}"})
            for j in range(n)]
    out = with_final_readout(Circuit(name="t", instructions=tuple(ops)), 8 * rounds, "fidelity")
    meas = [op for op in out.instructions if op.name.startswith("measure")]
    assert [op.args["key"] for op in meas] == [f"c_{n}_0_{j}" for j in range(n)]
    assert all(op.name == "measure" and op.implementation is None for op in meas[:16])
    assert all(op.name == "measure_fidelity" and op.implementation == "shelved_constant"
               for op in meas[16:])
    assert out.instructions[0] == ops[0]


@pytest.mark.parametrize("decoder", ["union_find", "bposd", "belief_matching"])
def test_ldpc_decoders(decoder):
    """Union-Find, BP+OSD and belief-matching all correct errors in simulation,
    and the BP-based ones are at least as good as plain MWPM."""
    c = build_memory_circuit(3, 3, "Z", True, NoiseModel.si1000(3e-3))
    det, obs = c.compile_detector_sampler(seed=11).sample(5_000, separate_observables=True)
    res = decode(det, obs, c, decoder)
    assert res.p_logical < 0.5 * decode(det, obs, c, "none").p_logical
    if decoder != "union_find":
        assert res.p_logical <= decode(det, obs, c, "mwpm").p_logical + 3 * res.stderr
