"""Rule-complexity metrics and the Wolfram-class rule panel.

Wolfram-class labels are qualitative and contested at the III/IV boundary, so the *operational*
independent variable for the rule sweep is a quantitative complexity score: the Lempel-Ziv
complexity of a rule's spacetime diagram (cheap; used by "Intelligence at the Edge of Chaos").
"""

from __future__ import annotations

import numpy as np

from procedural_warmup.data.ca.eca import simulate_spacetime

# Canonical exemplars per Wolfram class (docs/cellular-automata.md).
WOLFRAM_CLASSES: dict[str, list[int]] = {
    "I": [0, 32, 160],  # homogeneous
    "II": [108, 250],  # periodic / stable
    "III": [30, 90, 150],  # chaotic
    "IV": [54, 110, 137, 193],  # complex / edge-of-chaos
}


def rule_class(rule: int) -> str | None:
    for cls, rules in WOLFRAM_CLASSES.items():
        if rule in rules:
            return cls
    return None


def default_panel() -> list[int]:
    """Flat list of all panel rules across the four classes."""
    return [r for rules in WOLFRAM_CLASSES.values() for r in rules]


def lempel_ziv_complexity(sequence: str) -> int:
    """Lempel-Ziv (LZ78-style distinct-factor) complexity of a binary string."""
    sub_strings: set[str] = set()
    n = len(sequence)
    ind, inc = 0, 1
    while ind + inc <= n:
        sub = sequence[ind : ind + inc]
        if sub in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub)
            ind += inc
            inc = 1
    return len(sub_strings)


def rule_complexity(
    rule: int,
    width: int = 128,
    rows: int = 256,
    burn_in: int = 128,
    seed: int = 0,
) -> dict:
    """Compute LZ complexity of a fixed-size spacetime diagram for ``rule``.

    Returns the raw LZ count, a length-normalized score, and the live-cell density (a
    coarse Class-I/II vs III/IV discriminator).
    """
    rng = np.random.default_rng(seed)
    st = simulate_spacetime(rule, width=width, n_rows=rows, burn_in=burn_in, rng=rng)
    bits = "".join(str(int(b)) for b in st.reshape(-1))
    lz = lempel_ziv_complexity(bits)
    n = len(bits)
    # Normalize by the asymptotic max LZ ~ n / log2(n) for a random binary string.
    norm = lz * np.log2(n) / n if n > 1 else 0.0
    return {
        "rule": rule,
        "class": rule_class(rule),
        "lz": lz,
        "lz_norm": float(norm),
        "density": float(st.mean()),
    }


def panel_complexity(panel: list[int] | None = None, **kwargs) -> list[dict]:
    """Complexity scores for every rule in ``panel`` (default: the full Wolfram panel)."""
    panel = panel if panel is not None else default_panel()
    return [rule_complexity(r, **kwargs) for r in panel]
