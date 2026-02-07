#!/usr/bin/env python3
"""Vocabulary health report — co-occurrence analysis and dead enum detection.

Generates a PDF report at data/reports/vocab_health_YYYY-MM-DD.pdf
Run after regenerating HDF5 embeddings or modifying vocabulary/parser.

Usage:
    python3 scripts/vocab_health_report.py
    python3 scripts/vocab_health_report.py --h5 data/card_mechanics_commander.h5
    python3 scripts/vocab_health_report.py --threshold 0.85  # co-occurrence threshold
"""

import argparse
import datetime
import sys
from pathlib import Path

import h5py
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.mechanics.vocabulary import Mechanic  # noqa: E402

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def load_mechanics(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        return f["mechanics"][:]


def compute_stats(mechanics: np.ndarray, threshold: float = 0.90):
    binary = (mechanics != 0).astype(np.float32)
    counts = binary.sum(axis=0)

    enum_names = {m.value: m.name for m in Mechanic}
    named = {v: n for v, n in enum_names.items() if not n.startswith("UNK_")}

    n_cards, vocab_size = mechanics.shape

    # Active / dead / low-fire
    active_named = [(v, n, int(counts[v])) for v, n in sorted(named.items()) if counts[v] >= 5]
    dead_named = [(v, n) for v, n in sorted(named.items()) if counts[v] == 0]
    low_fire = [(v, n, int(counts[v])) for v, n in sorted(named.items()) if 0 < counts[v] < 5]

    # Co-occurrence pairs
    active_idxs = np.where(counts >= 5)[0]
    cooccur = []
    for i_pos, i in enumerate(active_idxs):
        for j in active_idxs[i_pos + 1:]:
            both = (binary[:, i] * binary[:, j]).sum()
            if both == 0:
                continue
            p_ij = both / counts[j] if counts[j] > 0 else 0
            p_ji = both / counts[i] if counts[i] > 0 else 0
            if min(p_ij, p_ji) > threshold:
                cooccur.append({
                    "a_idx": int(i), "b_idx": int(j),
                    "a_name": enum_names.get(i, f"UNK_{i}"),
                    "b_name": enum_names.get(j, f"UNK_{j}"),
                    "p_a_given_b": float(p_ij),
                    "p_b_given_a": float(p_ji),
                    "count_a": int(counts[i]),
                    "count_b": int(counts[j]),
                    "count_both": int(both),
                })
    cooccur.sort(key=lambda x: min(x["p_a_given_b"], x["p_b_given_a"]), reverse=True)

    # Top 20 most-fired mechanics
    top20 = sorted(
        [(v, n, int(counts[v])) for v, n in named.items() if counts[v] > 0],
        key=lambda x: x[2], reverse=True,
    )[:20]

    return {
        "n_cards": n_cards,
        "vocab_size": vocab_size,
        "named_total": len(named),
        "active_named": active_named,
        "dead_named": dead_named,
        "low_fire": low_fire,
        "cooccur": cooccur,
        "top20": top20,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# PDF rendering (matplotlib)
# ---------------------------------------------------------------------------

def _new_page(pdf, plt):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax


# Line height constants (fraction of page height)
LH_TITLE = 0.028
LH_ROW = 0.016
LH_SECTION = 0.022
LH_GAP = 0.012


def render_pdf(stats: dict, output_path: str, h5_path: str):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    today = datetime.date.today().isoformat()

    with PdfPages(output_path) as pdf:
        # --- Page 1: Summary + Co-occurrence + Top 20 ---
        fig, ax = _new_page(pdf, plt)

        y = 0.96
        ax.text(0.5, y, "Vocabulary Health Report", ha="center", fontsize=16, fontweight="bold")
        y -= LH_TITLE
        ax.text(0.5, y, f"{today}  |  {h5_path}", ha="center", fontsize=8, color="gray")
        y -= LH_SECTION

        # Summary table
        summary = [
            ("Cards analyzed", f"{stats['n_cards']:,}"),
            ("VOCAB_SIZE", str(stats["vocab_size"])),
            ("Named enums", str(stats["named_total"])),
            ("Active (fire >= 5)", str(len(stats["active_named"]))),
            ("Low-fire (1-4 cards)", str(len(stats["low_fire"]))),
            ("Dead (0 fires)", str(len(stats["dead_named"]))),
            ("Co-occurrence pairs (>{:.0f}%)".format(stats["threshold"] * 100),
             str(len(stats["cooccur"]))),
        ]
        for label, val in summary:
            ax.text(0.08, y, label, fontsize=9, fontfamily="monospace")
            ax.text(0.55, y, val, fontsize=9, fontweight="bold", fontfamily="monospace")
            y -= LH_ROW

        # Co-occurrence table
        y -= LH_GAP
        ax.text(0.5, y, "High Co-occurrence Pairs", ha="center", fontsize=12, fontweight="bold")
        y -= LH_SECTION

        if stats["cooccur"]:
            headers = ["Mechanic A", "Mechanic B", "P(A|B)", "P(B|A)", "#A", "#B"]
            col_x = [0.05, 0.28, 0.52, 0.62, 0.72, 0.82]
            for i, h in enumerate(headers):
                ax.text(col_x[i], y, h, fontsize=7, fontweight="bold", fontfamily="monospace")
            y -= 0.004
            ax.plot([0.05, 0.90], [y, y], color="black", linewidth=0.5)
            y -= LH_ROW

            for pair in stats["cooccur"]:
                row = [
                    pair["a_name"][:24],
                    pair["b_name"][:24],
                    f"{pair['p_a_given_b']:.2f}",
                    f"{pair['p_b_given_a']:.2f}",
                    str(pair["count_a"]),
                    str(pair["count_b"]),
                ]
                for i, val in enumerate(row):
                    ax.text(col_x[i], y, val, fontsize=7, fontfamily="monospace")
                y -= LH_ROW
        else:
            ax.text(0.5, y, "None found", ha="center", fontsize=9, color="gray")
            y -= LH_ROW

        # Top 20
        y -= LH_GAP
        ax.text(0.5, y, "Top 20 Most-Fired Mechanics", ha="center", fontsize=12, fontweight="bold")
        y -= LH_SECTION

        # Render as 2 columns of 10
        left_10 = stats["top20"][:10]
        right_10 = stats["top20"][10:20]
        y_start = y
        for rank, (v, n, c) in enumerate(left_10, 1):
            ax.text(0.06, y, f"{rank:2d}.", fontsize=7, fontfamily="monospace", color="gray")
            ax.text(0.10, y, n, fontsize=7, fontfamily="monospace")
            ax.text(0.42, y, f"{c:,}", fontsize=7, fontfamily="monospace", ha="right")
            y -= LH_ROW

        y2 = y_start
        for rank, (v, n, c) in enumerate(right_10, 11):
            ax.text(0.52, y2, f"{rank:2d}.", fontsize=7, fontfamily="monospace", color="gray")
            ax.text(0.56, y2, n, fontsize=7, fontfamily="monospace")
            ax.text(0.90, y2, f"{c:,}", fontsize=7, fontfamily="monospace", ha="right")
            y2 -= LH_ROW

        y = min(y, y2)

        # Recommendations
        y -= LH_GAP
        ax.text(0.5, y, "Recommendations", ha="center", fontsize=12, fontweight="bold")
        y -= LH_SECTION
        recs = [
            "1. SET_POWER + SET_TOUGHNESS: merge candidate (100% mutual co-occurrence)",
            "2. TO_HAND: delete candidate (fully redundant with BUYBACK, 40/40 overlap)",
            "3. Dead enums: no action needed (zero columns carry zero gradient)",
            "4. Low-fire enums: keep (real mechanics, just rare)",
            "5. DAYBOUND/NIGHTBOUND, LOYALTY_*: keep separate (semantically distinct)",
        ]
        for rec in recs:
            if y < 0.03:
                break
            ax.text(0.06, y, rec, fontsize=7, fontfamily="monospace")
            y -= LH_ROW

        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 2: Dead mechanics ---
        fig, ax = _new_page(pdf, plt)

        y = 0.96
        ax.text(0.5, y, "Dead Mechanics (0 fires)", ha="center", fontsize=12, fontweight="bold")
        y -= LH_ROW
        ax.text(0.5, y, f"{len(stats['dead_named'])} named enums with zero occurrences",
                ha="center", fontsize=8, color="gray")
        y -= LH_SECTION

        # Categorize dead mechanics
        categories = {
            "Unimplemented targeting/conditions": [
                "TARGET_ABILITY", "TARGET_SELF", "TARGET_ANY_CONTROLLER",
                "TARGET_CARD_IN_HAND", "NO_TARGET", "IF_CONDITION",
                "IF_SPELL_CAST", "IF_MANA_SPENT", "IF_CREATURE_DIED",
                "IF_OPPONENT_ATTACKED",
            ],
            "Zone transitions (never wired)": [
                "FROM_BATTLEFIELD", "TO_EXILE", "TO_LIBRARY_TOP",
                "TO_LIBRARY_BOTTOM", "TO_BATTLEFIELD", "EXILE_WITH_COUNTER",
            ],
            "Subsumed by KEYWORD_IMPLICATIONS": [
                "DASH_HASTE", "BLITZ_DRAW", "EXPLOIT_CREATURE", "WARD_COST",
                "MAGECRAFT_COPY", "GOAD_CREATURE", "FOR_MIRRODIN", "DISCOVER_X",
            ],
            "Niche/obsolete keywords": [
                "POPULATE_AURA", "SPLICE_ARCANE", "RADIANCE", "HAUNT",
                "TEMPTING_OFFER", "JOIN_FORCES", "MYRIAD_TOKENS",
            ],
            "Other": [],
        }
        categorized = set()
        for cat_items in categories.values():
            categorized.update(cat_items)

        # Put uncategorized into "Other"
        for v, n in stats["dead_named"]:
            if n not in categorized:
                categories["Other"].append(n)

        # Render as 2 columns to save vertical space
        all_cat_items = []
        for cat_name, members in categories.items():
            matching = [n for n in members if any(d[1] == n for d in stats["dead_named"])]
            if matching:
                all_cat_items.append((cat_name, matching))

        for cat_name, matching in all_cat_items:
            if y < 0.04:
                pdf.savefig(fig)
                plt.close(fig)
                fig, ax = _new_page(pdf, plt)
                y = 0.96
            ax.text(0.06, y, f"{cat_name} ({len(matching)})", fontsize=9, fontweight="bold")
            y -= LH_ROW
            for name in matching:
                idx = next((v for v, n in stats["dead_named"] if n == name), "?")
                ax.text(0.10, y, f"{idx}: {name}", fontsize=7, fontfamily="monospace", color="#555555")
                y -= LH_ROW
            y -= LH_GAP * 0.5

        # Low-fire mechanics
        if stats["low_fire"]:
            y -= LH_GAP
            ax.text(0.5, y, "Low-Fire Mechanics (1-4 cards)", ha="center", fontsize=10, fontweight="bold")
            y -= LH_SECTION
            for v, n, c in stats["low_fire"]:
                ax.text(0.10, y, f"{v}: {n}", fontsize=7, fontfamily="monospace")
                ax.text(0.60, y, f"{c} card{'s' if c > 1 else ''}", fontsize=7, fontfamily="monospace")
                y -= LH_ROW

        pdf.savefig(fig)
        plt.close(fig)

    print(f"Report saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vocabulary health report")
    parser.add_argument("--h5", default="data/card_mechanics_commander.h5",
                        help="Path to HDF5 mechanics file")
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="Co-occurrence threshold (default: 0.90)")
    parser.add_argument("--output", default=None,
                        help="Output PDF path (default: data/reports/vocab_health_YYYY-MM-DD.pdf)")
    args = parser.parse_args()

    h5_path = str(Path(project_root / args.h5))
    if not Path(h5_path).exists():
        print(f"Error: HDF5 file not found: {h5_path}")
        print("Run: python3 -m src.mechanics.precompute_embeddings --format commander --bulk-json data/scryfall_bulk_cards.json")
        sys.exit(1)

    today = datetime.date.today().isoformat()
    output = args.output or str(project_root / f"data/reports/vocab_health_{today}.pdf")
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing {h5_path} ...")
    mechanics = load_mechanics(h5_path)
    stats = compute_stats(mechanics, threshold=args.threshold)

    print(f"  {stats['n_cards']:,} cards, {stats['vocab_size']} vocab slots")
    print(f"  {len(stats['active_named'])} active, {len(stats['dead_named'])} dead, "
          f"{len(stats['low_fire'])} low-fire")
    print(f"  {len(stats['cooccur'])} co-occurrence pairs (>{stats['threshold']:.0%})")

    render_pdf(stats, output, args.h5)


if __name__ == "__main__":
    main()
