"""Bloch-gauge transforms and eigenvector overlap validation helpers.

All functions in this module use NumPy only.  The primary eigenvector layout is
``(Nq, nb, nat, 3)``: q-point, mode (band), atom, Cartesian component.  The
flat layout accepted by :func:`apply_bloch_gauge_flat` is the layout required by
``Phono3py.set_phonon_data``: ``(Nq, nat * 3, mode)``, where the row is
``atom * 3 + Cartesian component`` and mode is the final (column) axis.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np


GaugeTransform = namedtuple("GaugeTransform", "sign conjugate qflip")
"""Complete, reproducible choice made by :func:`select_gauge`."""


def _validate_sign(sign: int) -> int:
    """Return ``sign`` as an integer after requiring it to be exactly +1 or -1."""
    if isinstance(sign, (bool, np.bool_)):
        raise ValueError("sign must be +1 or -1, not a boolean")
    try:
        value = int(sign)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("sign must be +1 or -1") from exc
    if value != sign or value not in (-1, 1):
        raise ValueError("sign must be +1 or -1")
    return value


def _validate_q_tau(q_cryst: np.ndarray, tau_frac: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert and validate q-points and fractional atom positions."""
    q = np.asarray(q_cryst)
    tau = np.asarray(tau_frac)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q_cryst must have shape (Nq, 3)")
    if tau.ndim != 2 or tau.shape[1] != 3:
        raise ValueError("tau_frac must have shape (nat, 3)")
    return q, tau


def apply_bloch_gauge(eigvecs: np.ndarray, q_cryst: np.ndarray, tau_frac: np.ndarray, sign: int) -> np.ndarray:
    """Apply an atom-dependent Bloch phase to primary-layout eigenvectors.

    ``eigvecs`` has shape ``(Nq, nb, nat, 3)``.  Its axes are, respectively,
    q-point, mode/band, atom, and Cartesian component.  ``q_cryst`` has shape
    ``(Nq, 3)`` and ``tau_frac`` has shape ``(nat, 3)``.  The returned array is
    ``eigvecs * exp(sign * 2j*pi*q.tau)[..., None, :, None]``; ``sign`` must be
    exactly ``+1`` or ``-1``.
    """
    e = np.asarray(eigvecs)
    q, tau = _validate_q_tau(q_cryst, tau_frac)
    sign = _validate_sign(sign)
    if e.ndim != 4 or e.shape[-1] != 3:
        raise ValueError("eigvecs must have shape (Nq, nb, nat, 3)")
    if e.shape[0] != q.shape[0] or e.shape[2] != tau.shape[0]:
        raise ValueError("eigvecs, q_cryst, and tau_frac have incompatible shapes")
    phase = np.exp(sign * 2j * np.pi * (q @ tau.T))
    return e * phase[:, None, :, None]


def apply_bloch_gauge_flat(
    eigvecs_flat: np.ndarray, q_cryst: np.ndarray, tau_frac: np.ndarray, sign: int
) -> np.ndarray:
    """Apply the same Bloch phase to Phono3py's flattened eigenvector layout.

    ``eigvecs_flat`` has shape ``(Nq, nat*3, mode)`` as required by
    ``Phono3py.set_phonon_data``.  The second-axis row is ``atom*3 + cart`` and
    the final axis is the mode column.  ``q_cryst`` is ``(Nq, 3)`` and
    ``tau_frac`` is ``(nat, 3)``.  Each three-row atom block receives
    ``exp(sign * 2j*pi*q.tau_atom)``.  Thus, for primary-layout ``e``, this
    function agrees exactly with applying :func:`apply_bloch_gauge` to ``e``
    and then converting via ``e.reshape(Nq, nb, nat*3).transpose(0, 2, 1)``.
    """
    e = np.asarray(eigvecs_flat)
    q, tau = _validate_q_tau(q_cryst, tau_frac)
    sign = _validate_sign(sign)
    if e.ndim != 3:
        raise ValueError("eigvecs_flat must have shape (Nq, nat*3, mode)")
    if e.shape[0] != q.shape[0] or e.shape[1] != 3 * tau.shape[0]:
        raise ValueError("eigvecs_flat, q_cryst, and tau_frac have incompatible shapes")
    phase = np.exp(sign * 2j * np.pi * (q @ tau.T))
    return e * np.repeat(phase, 3, axis=1)[:, :, None]


def _validate_primary_pair(e_a: np.ndarray, e_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate two equal-shape arrays in the primary ``(Nq, nb, nat, 3)`` layout."""
    a = np.asarray(e_a)
    b = np.asarray(e_b)
    if a.ndim != 4 or a.shape[-1] != 3:
        raise ValueError("e_a and e_b must have shape (Nq, nb, nat, 3)")
    if a.shape != b.shape:
        raise ValueError("e_a and e_b must have identical shapes")
    return a, b


def overlap(e_a: np.ndarray, e_b: np.ndarray) -> np.ndarray:
    """Return normalized per-mode overlap magnitudes for primary-layout arrays.

    Both inputs have shape ``(Nq, nb, nat, 3)`` (q-point, mode, atom,
    Cartesian).  The result has shape ``(Nq, nb)`` and contains
    ``abs(<e_a|e_b>) / (||e_a|| * ||e_b||)``, contracting atom and Cartesian
    axes.  Zero-norm modes receive a score of zero; nonzero modes are clipped
    to the physical interval ``[0, 1]``.
    """
    a, b = _validate_primary_pair(e_a, e_b)
    raw = np.sum(np.conj(a) * b, axis=(2, 3))
    norms = np.sqrt(np.sum(np.abs(a) ** 2, axis=(2, 3)) * np.sum(np.abs(b) ** 2, axis=(2, 3)))
    result = np.divide(np.abs(raw), norms, out=np.zeros_like(norms, dtype=float), where=norms > 0)
    return np.clip(result, 0.0, 1.0)


def subspace_overlap(
    e_a: np.ndarray, e_b: np.ndarray, freqs: np.ndarray | None = None, degen_tol: float = 1e-4
) -> np.ndarray:
    """Return the worst degenerate-subspace overlap score at each q-point.

    Inputs use the primary ``(Nq, nb, nat, 3)`` layout (q-point, mode, atom,
    Cartesian).  If ``freqs`` with shape ``(Nq, nb)`` is supplied, modes are
    clustered independently at each q-point: consecutive modes belong to one
    cluster when their consecutive frequency gap is strictly less than
    ``degen_tol``.  If ``freqs`` is ``None``, all modes at a q-point form one
    cluster.  For every cluster this constructs the complex, individually
    normalized block ``S[i,j]=<a_i|b_j>/(||a_i|| ||b_j||)`` and takes its
    smallest singular value.  The returned real array has shape ``(Nq,)`` and
    is the minimum cluster score at each q-point.  Eigenvector modes are
    expected to be orthonormal within each input set, as phonon eigenvectors
    normally are.
    """
    a, b = _validate_primary_pair(e_a, e_b)
    nq, nb = a.shape[:2]
    if not np.isfinite(degen_tol) or degen_tol < 0:
        raise ValueError("degen_tol must be a finite non-negative number")
    if freqs is None:
        frequencies = None
    else:
        frequencies = np.asarray(freqs)
        if frequencies.shape != (nq, nb):
            raise ValueError("freqs must have shape (Nq, nb)")

    scores = np.empty(nq, dtype=float)
    for iq in range(nq):
        if frequencies is None:
            clusters = [np.arange(nb)]
        else:
            breaks = np.flatnonzero(np.diff(frequencies[iq]) >= degen_tol) + 1
            clusters = np.split(np.arange(nb), breaks)
        cluster_scores = []
        for indices in clusters:
            a_block = a[iq, indices].reshape(len(indices), -1)
            b_block = b[iq, indices].reshape(len(indices), -1)
            norms_a = np.linalg.norm(a_block, axis=1)
            norms_b = np.linalg.norm(b_block, axis=1)
            if np.any(norms_a == 0) or np.any(norms_b == 0):
                cluster_scores.append(0.0)
                continue
            matrix = (np.conj(a_block) @ b_block.T) / (norms_a[:, None] * norms_b[None, :])
            cluster_scores.append(float(np.linalg.svd(matrix, compute_uv=False).min()))
        scores[iq] = min(cluster_scores)
    return np.clip(scores, 0.0, 1.0)


def _qflip_permutation(q: np.ndarray, *, atol: float = 1e-8) -> np.ndarray:
    """Return indices mapping every grid point to its reciprocal-space inverse."""
    q = np.asarray(q, dtype=float)
    permutation = np.empty(len(q), dtype=np.intp)
    for index, point in enumerate(q):
        difference = q + point
        difference -= np.rint(difference)
        matches = np.flatnonzero(np.all(np.abs(difference) < atol, axis=1))
        if matches.size != 1:
            raise ValueError(
                "qflip requires a q grid with one unique -q partner for every point"
            )
        permutation[index] = matches[0]
    return permutation


def apply_selected_gauge(
    e_qe: np.ndarray,
    q: np.ndarray,
    tau: np.ndarray,
    transform: GaugeTransform,
    qflip_perm: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a :func:`select_gauge` result exactly and return QE modes at ``q``.

    If ``transform.qflip`` is set, ``qflip_perm`` maps each requested q-point
    to the input eigenvectors at its -q partner.  The Bloch phase is evaluated
    at that source (-q) point, matching the transform considered in selection.
    """
    qe = np.asarray(e_qe)
    q_array, tau_array = _validate_q_tau(q, tau)
    if qe.ndim != 4 or qe.shape[-1] != 3:
        raise ValueError("e_qe must have shape (Nq, nb, nat, 3)")
    if qe.shape[0] != len(q_array) or qe.shape[2] != len(tau_array):
        raise ValueError("e_qe, q, and tau have incompatible shapes")
    try:
        sign = _validate_sign(transform.sign)
        conjugate = bool(transform.conjugate)
        qflip = bool(transform.qflip)
    except AttributeError as exc:
        raise ValueError("transform must provide sign, conjugate, and qflip") from exc
    if qflip:
        if qflip_perm is None:
            raise ValueError("qflip_perm is required when transform.qflip is True")
        permutation = np.asarray(qflip_perm, dtype=np.intp)
        if permutation.shape != (len(q_array),) or np.any(permutation < 0) or np.any(
            permutation >= len(q_array)
        ):
            raise ValueError("qflip_perm must be a valid (Nq,) integer index array")
        qe = qe[permutation]
        phase_q = -q_array
    else:
        phase_q = q_array
    if conjugate:
        qe = np.conj(qe)
    return apply_bloch_gauge(qe, phase_q, tau_array, sign)


def select_gauge(
    e_ph: np.ndarray,
    e_qe: np.ndarray,
    q: np.ndarray,
    tau: np.ndarray,
    freqs: np.ndarray | None = None,
    degen_tol: float = 1e-4,
) -> tuple[GaugeTransform, float, np.ndarray | None]:
    """Select a complete Bloch-gauge transform using degenerate subspaces.

    ``e_ph`` and ``e_qe`` both use the primary ``(Nq, nb, nat, 3)`` layout:
    q-point, mode, atom, Cartesian component.  ``e_ph`` is the reference and
    ``e_qe`` is transformed.  The function evaluates both signs as well as
    optional complex conjugation and q inversion of ``e_qe``/``q`` internally,
    then maximizes the mean :func:`subspace_overlap`, which is insensitive to
    arbitrary rotations within degenerate mode clusters.  It returns
    ``(transform, residual, qflip_perm)``.  ``transform`` contains the chosen
    sign, conjugation, and q-flip flags; the permutation is supplied when and
    only when q-flip is selected, so :func:`apply_selected_gauge` can reproduce
    the winning eigenvectors exactly.
    """
    reference, qe = _validate_primary_pair(e_ph, e_qe)
    if freqs is not None and np.asarray(freqs).shape != reference.shape[:2]:
        raise ValueError("freqs must have shape (Nq, nb)")
    if not np.isfinite(degen_tol) or degen_tol < 0:
        raise ValueError("degen_tol must be a finite non-negative number")

    # A deterministic near-tie rule favors candidates requiring fewer extra
    # operations, then the declared sign order (+1 before -1).  The tolerance
    # prevents roundoff from choosing q-flip over the equivalent inverse sign.
    best_score = -np.inf
    best_cost = np.inf
    best_transform = GaugeTransform(sign=1, conjugate=False, qflip=False)
    best_permutation: np.ndarray | None = None
    try:
        qflip_permutation = _qflip_permutation(q)
    except ValueError:
        # Arbitrary q lists (including small unit-test fixtures) need not be
        # inversion closed.  They still admit the non-q-flipped candidates.
        qflip_permutation = None
    tolerance = 1e-12
    for sign in (1, -1):
        for conjugate in (False, True):
            for qflip in (False, True):
                if qflip and qflip_permutation is None:
                    continue
                transform = GaugeTransform(sign=sign, conjugate=conjugate, qflip=qflip)
                permutation = qflip_permutation if qflip else None
                gauged = apply_selected_gauge(qe, q, tau, transform, permutation)
                score = float(np.mean(subspace_overlap(reference, gauged, freqs, degen_tol)))
                cost = int(conjugate) + int(qflip)
                if score > best_score + tolerance or (
                    abs(score - best_score) <= tolerance and cost < best_cost
                ):
                    best_score = score
                    best_cost = cost
                    best_transform = transform
                    best_permutation = permutation
    return best_transform, float(max(0.0, 1.0 - best_score)), best_permutation
