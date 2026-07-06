"""Explanatory figure: one DW_k picture and its corner-close-only pretraining task.

Renders three panels from a real sampler output (no hand-drawing):

  (a) the sampled picture — every cell is one corner symbol a_i/b_i/c_i/d_i, and every
      cell belongs to exactly ONE matched quadruple whose four corners form a
      rectangle (drawn as a colored box), so an m x n picture always contains exactly
      m*n/4 quadruples (len(rects) == m*n/4, asserted by the generator and tests);
  (b) the masked input the ViT actually sees — eligible d-corners (bottom-right,
      rectangle bigger than 2x2) masked with probability mask_ratio, everything else
      visible; matching lines from one masked d to its visible a/b/c partners show why
      the target is uniquely forced (LIFO row/column Dyck parsing);
  (c) the training targets — cross-entropy is taken ONLY at the masked positions, and
      the label is always some d_i (a 32-way choice under k=32).

    python -m procedural_warmup.analysis.dyck2d_example            # 8x8, k=4 (legible)
    python -m procedural_warmup.analysis.dyck2d_example --full     # adds a 14x14 panel
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from procedural_warmup.data.dyck2d.alphabet import Role, index_of, role_of
from procedural_warmup.data.dyck2d.generator import dw_picture

_ROLE_CHAR = {Role.A: "a", Role.B: "b", Role.C: "c", Role.D: "d"}


def _label(tid: int, k: int) -> str:
    return f"{_ROLE_CHAR[role_of(tid, k)]}$_{{{index_of(tid, k)}}}$"


def _draw_grid(ax, grid, k, rects=None, mask=None, elig=None, targets_only=False,
               show_match_for=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    m, n = grid.shape
    cmap = plt.get_cmap("tab20")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(m - 0.5, -0.5)  # row 0 on top
    ax.set_xticks(range(n))
    ax.set_yticks(range(m))
    ax.tick_params(length=0, labelsize=7)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Quadruple bounding boxes, one color per quadruple.
    if rects is not None:
        for j, r in enumerate(rects):
            color = cmap(j % 20)
            ax.add_patch(Rectangle(
                (r.c1 - 0.42, r.r1 - 0.42), r.col_span + 0.84, r.row_span + 0.84,
                fill=False, edgecolor=color, lw=1.6, alpha=0.9, zorder=1,
            ))

    for r in range(m):
        for c in range(n):
            masked_here = mask is not None and mask[r, c]
            if targets_only and not masked_here:
                ax.text(c, r, "·", ha="center", va="center", fontsize=8, color="0.75")
                continue
            if masked_here and not targets_only:
                ax.add_patch(Rectangle((c - 0.45, r - 0.45), 0.9, 0.9,
                                       facecolor="0.15", zorder=2))
                ax.text(c, r, "?", ha="center", va="center", fontsize=10,
                        color="white", zorder=3, weight="bold")
                continue
            weight = "bold" if (elig is not None and elig[r, c]) else "normal"
            color = "crimson" if masked_here else ("darkgreen" if weight == "bold" else "black")
            ax.text(c, r, _label(int(grid[r, c]), k), ha="center", va="center",
                    fontsize=9, zorder=3, weight=weight, color=color)

    # Matching lines from one masked d to its visible a/b/c partners.
    if show_match_for is not None:
        rect = show_match_for
        d = (rect.r2, rect.c2)
        for partner, style in (((rect.r1, rect.c1), "-"), ((rect.r1, rect.c2), "-"),
                               ((rect.r2, rect.c1), "-")):
            ax.annotate("", xy=(partner[1], partner[0]), xytext=(d[1], d[0]),
                        arrowprops=dict(arrowstyle="->", color="crimson", lw=1.8,
                                        linestyle=style, shrinkA=12, shrinkB=12),
                        zorder=4)


def make_figure(out_dir: Path, m: int = 8, n: int = 8, k: int = 4, seed: int = 7,
                mask_ratio: float = 0.5, min_match_distance: int = 2,
                filename: str = "dyck2d_task_example.png") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Deterministic example with at least one accretion (a big rectangle) for clarity.
    rng = random.Random(seed)
    while True:
        grid, rects = dw_picture(m, n, k, p_acc=0.7, rng=rng)
        if max(max(r.row_span, r.col_span) for r in rects) >= 5:
            break

    elig = np.zeros((m, n), dtype=bool)
    for r in rects:
        if max(r.row_span, r.col_span) >= min_match_distance:
            elig[r.d_pos] = True
    mask_rng = np.random.default_rng(seed)
    mask = elig & (mask_rng.random((m, n)) < mask_ratio)
    # Highlight the largest masked rectangle's matching structure.
    masked_rects = [r for r in rects if mask[r.d_pos]]
    showcase = max(masked_rects, key=lambda r: r.row_span + r.col_span)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    _draw_grid(axes[0], grid, k, rects=rects)
    axes[0].set_title(
        f"(a) DW$_{{{k}}}$ picture, {m}x{n}: every cell is one corner of exactly one\n"
        f"matched quadruple -> {m}*{n}/4 = {m * n // 4} quadruples (boxes)",
        fontsize=10,
    )
    _draw_grid(axes[1], grid, k, mask=mask, elig=elig, show_match_for=showcase)
    axes[1].set_title(
        f"(b) model input: eligible d-corners (bold green) masked w.p. {mask_ratio}\n"
        f"-> {int(mask.sum())} MASK tokens; arrows: the a/b/c partners forcing one d",
        fontsize=10,
    )
    _draw_grid(axes[2], grid, k, mask=mask, targets_only=True)
    axes[2].set_title(
        "(c) training targets: cross-entropy ONLY at masked cells;\n"
        "each label is the d$_i$ uniquely forced by LIFO row+column parsing",
        fontsize=10,
    )
    for ax, lbl in zip(axes, "abc"):
        ax.set_xlabel("column", fontsize=8)
    axes[0].set_ylabel("row", fontsize=8)
    fig.suptitle(
        "DW$_k$ corner-close-only masked prediction (H6) — at the real 14x14 / k=32 scale: "
        "49 quadruples, ~34.5 eligible d's, ~17 masked per picture",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="DW_k pretraining-task explanatory figure.")
    ap.add_argument("--size", type=int, nargs=2, default=[8, 8], metavar=("M", "N"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    path = make_figure(Path(args.out), m=args.size[0], n=args.size[1], k=args.k,
                       seed=args.seed)
    print(f"[dyck2d-example] wrote {path}")


if __name__ == "__main__":
    main()
