#!/usr/bin/env python3
"""Training report generator and email notifier.

Creates a comprehensive multi-page PDF covering:
  1. Training Summary — model info, loss, checkpoint
  2. Network Architecture — visual diagram with parameter counts
  3. Pipeline Timing Profile — horizontal bar chart + table
  4. Monitoring Links — TensorBoard, W&B, S3
  5. Training Curves — loss/accuracy over epochs (from TensorBoard logs or metrics dict)
  6. Deck Mechanics Analysis — centroid feature profile of training decks

Usage:
    # First-time setup (creates .env with Gmail App Password):
    ./scripts/setup_email.sh

    # Generate + send dummy report (for testing email pipeline):
    FORGERL_NOTIFY_EMAIL=user@example.com uv run python3 scripts/send_test_report.py

    # Send an existing PDF:
    uv run python3 scripts/send_test_report.py --pdf data/reports/vocab_health_2026-02-07.pdf

    # Generate PDF only (no email):
    uv run python3 scripts/send_test_report.py --save-only

    # Custom subject:
    uv run python3 scripts/send_test_report.py --subject "Nightly Training Report"

Note: Email requires .env file with FORGERL_SMTP_PASS (Gmail App Password).
      Run ./scripts/setup_email.sh for interactive setup wizard.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for local testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.email_notifier import EmailNotifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
S3_BUCKET = "mtg-rl-checkpoints-20260124190118616600000001"
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "mtg-selfplay")


# ---------------------------------------------------------------------------
# Helper: try to read TensorBoard event files for loss curves
# ---------------------------------------------------------------------------

def _read_tb_scalars(log_dir: str, tag: str, max_points: int = 200) -> List[tuple]:
    """Return [(step, value), ...] for *tag* from TensorBoard event files.

    Returns an empty list if tensorboard or the log directory is unavailable.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return []

    log_path = Path(log_dir)
    if not log_path.is_dir():
        return []

    # EventAccumulator can take a directory with multiple event files
    ea = EventAccumulator(str(log_path), size_guidance={"scalars": max_points})
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        return []

    events = ea.Scalars(tag)
    return [(e.step, e.value) for e in events]


# ---------------------------------------------------------------------------
# PDF Generation
# ---------------------------------------------------------------------------

def generate_training_report(
    metrics: Optional[Dict[str, Any]] = None,
    timing: Optional[Dict[str, float]] = None,
    timing_detailed: Optional[Dict[str, Dict[str, float]]] = None,
    output_path: Optional[str] = None,
    tb_log_dir: str = "runs",
) -> str:
    """Generate a multi-page training report PDF.

    Args:
        metrics: Training metrics dict.  Expected keys (all optional):
            model_name, checkpoint_path, total_time_s, epochs,
            policy_loss, value_loss, accuracy, best_checkpoint,
            games_played, win_rate
        timing: ``{stage_name: total_seconds}`` from PipelineTimer.summary()
        timing_detailed: ``{stage_name: {total_s, count, mean_s, pct_of_wall}}``
        output_path: Where to write the PDF.  Auto-generated if None.
        tb_log_dir: Root directory for TensorBoard event files.

    Returns:
        Absolute path to the generated PDF.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        logger.error("matplotlib not installed. Install with: uv sync --extra dev")
        sys.exit(1)

    if metrics is None:
        metrics = _dummy_metrics()
    if timing is None:
        timing = _dummy_timing()

    output_dir = Path("data/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(output_dir / f"training_report_{timestamp}.pdf")

    with PdfPages(output_path) as pdf:
        _page_training_summary(pdf, metrics)
        _page_network_architecture(pdf)
        _page_pipeline_timing(pdf, timing, timing_detailed)
        _page_monitoring_links(pdf, metrics)
        _page_training_curves(pdf, metrics, tb_log_dir)
        _page_deck_mechanics(pdf)

    logger.info(f"Created training report PDF: {output_path}")
    return output_path


# -- Page 1: Training Summary -----------------------------------------------

def _page_training_summary(pdf, metrics: Dict[str, Any]):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    model_name = metrics.get("model_name", "AlphaZero policy/value")
    ckpt = metrics.get("checkpoint_path", "checkpoints/latest.pt")
    total_time = metrics.get("total_time_s", 0)
    epochs = metrics.get("epochs", 0)
    policy_loss = metrics.get("policy_loss", float("nan"))
    value_loss = metrics.get("value_loss", float("nan"))
    accuracy = metrics.get("accuracy", float("nan"))
    best_ckpt = metrics.get("best_checkpoint", ckpt)
    games = metrics.get("games_played", 0)
    win_rate = metrics.get("win_rate", None)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    hours = total_time / 3600
    mins = (total_time % 3600) / 60

    lines = [
        "ForgeRL Training Report",
        "=" * 50,
        "",
        f"  Date:              {now}",
        f"  Model:             {model_name}",
        f"  Checkpoint:        {ckpt}",
        f"  Best checkpoint:   {best_ckpt}",
        "",
        f"  Total time:        {int(hours)}h {int(mins)}m ({total_time:.0f}s)",
        f"  Epochs completed:  {epochs}",
        f"  Games played:      {games}",
    ]
    if win_rate is not None:
        lines.append(f"  Win rate:          {win_rate:.1f}%")

    lines += [
        "",
        "Final Loss Values",
        "-" * 50,
        f"  Policy loss:       {policy_loss:.4f}",
        f"  Value loss:        {value_loss:.4f}",
    ]
    if accuracy is not None and accuracy == accuracy:  # not NaN
        lines.append(f"  Accuracy:          {accuracy:.3f}")

    text = "\n".join(lines)
    ax.text(
        0.08, 0.92, text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.5),
    )

    fig.suptitle("Page 1 — Training Summary", fontsize=9, y=0.02, color="grey")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# -- Page 2: Network Architecture -------------------------------------------

def _page_network_architecture(pdf):
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    import subprocess

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    # Generate the network diagram
    diagram_script = Path(__file__).parent / "generate_network_diagram.py"
    diagram_path = Path("data/reports/network_architecture.png")

    # Run the diagram generator if image doesn't exist or is stale
    if not diagram_path.exists():
        try:
            subprocess.run(
                ["python3", str(diagram_script)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to generate network diagram: {e.stderr}")
            ax.text(
                0.5, 0.5, "Network diagram not available\n(graphviz may not be installed)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="grey",
            )
            fig.suptitle("Page 2 — Network Architecture", fontsize=9, y=0.02, color="grey")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            return
        except FileNotFoundError:
            logger.warning("python3 not found in PATH")
            ax.text(
                0.5, 0.5, "Network diagram not available\n(generation failed)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="grey",
            )
            fig.suptitle("Page 2 — Network Architecture", fontsize=9, y=0.02, color="grey")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            return

    # Load and display the diagram
    if diagram_path.exists():
        img = imread(diagram_path)
        ax.imshow(img, aspect='auto')
        ax.axis("off")
    else:
        ax.text(
            0.5, 0.5, "Network diagram not found",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=14, color="grey",
        )

    fig.suptitle("Page 2 — Network Architecture", fontsize=9, y=0.02, color="grey")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# -- Page 3: Pipeline Timing Profile ----------------------------------------

def _page_pipeline_timing(pdf, timing: Dict[str, float], detailed=None):
    import matplotlib.pyplot as plt

    fig, (ax_bar, ax_tbl) = plt.subplots(
        2, 1, figsize=(8.5, 11), gridspec_kw={"height_ratios": [1, 1]}
    )

    stages = list(timing.keys())
    durations = list(timing.values())
    total = sum(durations) or 1.0

    # Horizontal bar chart
    colors = plt.cm.Set2([i / max(len(stages), 1) for i in range(len(stages))])
    y_pos = range(len(stages))
    ax_bar.barh(y_pos, durations, color=colors, edgecolor="grey", linewidth=0.5)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(stages, fontsize=9)
    ax_bar.set_xlabel("Seconds")
    ax_bar.set_title("Pipeline Stage Durations", fontweight="bold")
    ax_bar.invert_yaxis()

    for i, v in enumerate(durations):
        pct = v / total * 100
        ax_bar.text(v + total * 0.01, i, f"{v:.1f}s ({pct:.0f}%)", va="center", fontsize=8)

    # Table with detailed stats
    ax_tbl.axis("off")

    if detailed:
        col_labels = ["Stage", "Total (s)", "Count", "Mean (s)", "% of Wall"]
        table_data = []
        for stage in stages:
            d = detailed.get(stage, {})
            table_data.append([
                stage,
                f"{d.get('total_s', timing.get(stage, 0)):.2f}",
                str(d.get("count", 1)),
                f"{d.get('mean_s', 0):.3f}",
                f"{d.get('pct_of_wall', 0):.1f}%",
            ])
    else:
        col_labels = ["Stage", "Duration (s)", "% of Total"]
        table_data = [
            [s, f"{d:.2f}", f"{d / total * 100:.1f}%"]
            for s, d in zip(stages, durations)
        ]

    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)
    # Header row styling
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#4472C4")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax_tbl.set_title("Timing Details", fontweight="bold", pad=20)

    fig.suptitle("Page 3 — Pipeline Timing Profile", fontsize=9, y=0.02, color="grey")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# -- Page 3: Monitoring Links -----------------------------------------------

def _page_monitoring_links(pdf, metrics: Dict[str, Any]):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    tb_url = metrics.get("tensorboard_url", "http://localhost:6006")
    wandb_url = _build_wandb_url(metrics)
    s3_path = metrics.get("s3_path", f"s3://{S3_BUCKET}/")

    lines = [
        "Monitoring & Artifact Links",
        "=" * 55,
        "",
        "TensorBoard",
        "-" * 55,
        f"  Local:   {tb_url}",
        f"  Log dir: {metrics.get('tb_log_dir', 'runs/')}",
        "",
        "Weights & Biases",
        "-" * 55,
        f"  Dashboard: {wandb_url}",
        "",
        "S3 Checkpoints",
        "-" * 55,
        f"  Bucket:  {s3_path}",
        f"  Latest:  {metrics.get('checkpoint_path', 'N/A')}",
        "",
        "Quick Commands",
        "-" * 55,
        "  # View TensorBoard locally",
        "  tensorboard --logdir runs/ --port 6006",
        "",
        "  # Download latest checkpoint from S3",
        f"  aws s3 cp {s3_path}checkpoints/ checkpoints/ --recursive",
        "",
        "  # Tail training logs on EC2",
        "  ./scripts/ssh-instance.sh logs",
    ]

    text = "\n".join(lines)
    ax.text(
        0.08, 0.92, text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#e8f4e8", alpha=0.5),
    )

    # Clickable URL annotation (matplotlib supports url= on text)
    ax.annotate(
        wandb_url,
        xy=(0.08, 0.52), xycoords="axes fraction",
        fontsize=1,  # invisible — just carries the URL
        url=wandb_url,
    )

    fig.suptitle("Page 4 — Monitoring Links", fontsize=9, y=0.02, color="grey")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# -- Page 4: Training Curves ------------------------------------------------

def _page_training_curves(pdf, metrics: Dict[str, Any], tb_log_dir: str):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
    ax_loss, ax_acc = axes

    # Try reading from TensorBoard first
    loss_data = _read_tb_scalars(tb_log_dir, "imitation/loss")
    acc_data = _read_tb_scalars(tb_log_dir, "imitation/accuracy")

    # Fall back to metrics["history"] if present
    history = metrics.get("history", [])
    if not loss_data and history:
        loss_data = [(i, h.get("loss", h.get("policy_loss", 0))) for i, h in enumerate(history)]
    if not acc_data and history:
        acc_data = [(i, h.get("accuracy", 0)) for i, h in enumerate(history)]

    # Policy loss from selfplay TB logs
    if not loss_data:
        loss_data = _read_tb_scalars(tb_log_dir, "selfplay/policy_loss")

    if loss_data:
        steps, values = zip(*loss_data)
        ax_loss.plot(steps, values, color="#D32F2F", linewidth=1.5, label="Loss")
        ax_loss.set_xlabel("Epoch / Step")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Training Loss", fontweight="bold")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend()
    else:
        ax_loss.text(
            0.5, 0.5, "No training loss data available yet",
            transform=ax_loss.transAxes, ha="center", va="center",
            fontsize=14, color="grey",
        )
        ax_loss.set_title("Training Loss", fontweight="bold")

    if acc_data:
        steps, values = zip(*acc_data)
        ax_acc.plot(steps, values, color="#1976D2", linewidth=1.5, label="Accuracy")
        ax_acc.set_xlabel("Epoch / Step")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title("Training Accuracy", fontweight="bold")
        ax_acc.grid(True, alpha=0.3)
        ax_acc.legend()
    else:
        ax_acc.text(
            0.5, 0.5, "No training accuracy data available yet",
            transform=ax_acc.transAxes, ha="center", va="center",
            fontsize=14, color="grey",
        )
        ax_acc.set_title("Training Accuracy", fontweight="bold")

    fig.suptitle("Page 5 — Training Curves", fontsize=9, y=0.02, color="grey")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# -- Page 6: Deck Mechanics Analysis ----------------------------------------

def _parse_dck_file(filepath: Path) -> tuple[str, list[str]]:
    """Parse a Forge .dck deck file into (deck_name, [card_names]).

    Returns the deck name from metadata and a list of mainboard card names
    (one entry per copy, e.g. "4 Shock" yields 4 entries of "Shock").
    """
    deck_name = filepath.stem
    cards: list[str] = []
    in_main = False

    for line in filepath.read_text().splitlines():
        line = line.strip()
        if line.startswith("Name="):
            deck_name = line.split("=", 1)[1].strip()
        elif line == "[main]":
            in_main = True
        elif line.startswith("["):
            in_main = False
        elif in_main and line:
            import re
            m = re.match(r"^(\d+)\s+(.+)$", line)
            if m:
                qty = int(m.group(1))
                name = m.group(2).strip()
                cards.extend([name] * qty)
    return deck_name, cards


def _load_h5_for_analysis() -> tuple:
    """Load the HDF5 card mechanics database and return (mechanics, card_index, vocab_size, mechanic_names).

    Tries commander first, falls back to standard. Returns (None, ...) if unavailable.
    """
    import json

    from mechanics.vocabulary import Mechanic

    project_root = Path(__file__).parent.parent
    h5_paths = [
        project_root / "data" / "card_mechanics_commander.h5",
        project_root / "data" / "card_mechanics_standard.h5",
    ]

    h5_path = None
    for p in h5_paths:
        if p.exists():
            h5_path = p
            break
    if h5_path is None:
        return None, None, 0, {}

    try:
        import h5py
    except ImportError:
        return None, None, 0, {}

    with h5py.File(h5_path, "r") as f:
        mechanics = f["mechanics"][:]
        card_index = json.loads(f.attrs["card_index"])
        vocab_size = int(f.attrs["vocab_size"])

    # Build mechanic name lookup
    mechanic_names = {}
    for m in Mechanic:
        if m.value < vocab_size:
            mechanic_names[m.value] = m.name

    return mechanics, card_index, vocab_size, mechanic_names


def _page_deck_mechanics(pdf):
    """Page 6: Deck mechanics analysis — centroid feature profile of training decks."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(8.5, 11))

    # Load HDF5 card data
    mechanics, card_index, vocab_size, mechanic_names = _load_h5_for_analysis()

    if mechanics is None:
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            "Deck mechanics analysis not available\n\n"
            "(HDF5 card database not found in data/)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=14, color="grey",
        )
        fig.suptitle("Page 6 — Deck Mechanics Analysis", fontsize=9, y=0.02, color="grey")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    # Find all deck files
    decks_dir = Path(__file__).parent.parent / "decks"
    dck_files = sorted(decks_dir.glob("*.dck")) if decks_dir.is_dir() else []

    if not dck_files:
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            "No deck files found in decks/ directory",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=14, color="grey",
        )
        fig.suptitle("Page 6 — Deck Mechanics Analysis", fontsize=9, y=0.02, color="grey")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    # Build a front-face index for DFC name resolution
    front_face_index: dict[str, str] = {}
    for full_name in card_index:
        if " // " in full_name:
            front = full_name.split(" // ")[0]
            front_face_index[front] = full_name

    def resolve_name(name: str):
        if name in card_index:
            return name
        if name in front_face_index:
            return front_face_index[name]
        name_lower = name.lower()
        for key in card_index:
            if key.lower() == name_lower:
                return key
        for front, full in front_face_index.items():
            if front.lower() == name_lower:
                return full
        return None

    # Mechanics to skip in display (too generic)
    skip_display = {
        "TRIGGERED_ABILITY", "ACTIVATED_ABILITY", "SORCERY_SPEED", "INSTANT_SPEED",
        "UNTIL_END_OF_TURN", "TARGET_CREATURE", "TARGET_PLAYER",
        "ADD_MANA", "MANA_OF_ANY_COLOR", "MANA_FIXING", "TO_BATTLEFIELD_TAPPED",
        "TUTOR_LAND", "SEARCH_LIBRARY", "TAP_FOR_EFFECT", "SACRIFICE_COST",
        "ENTERS_THE_BATTLEFIELD", "MANA_COST", "COLOR_IDENTITY",
    }

    # Parse decks and compute centroids
    deck_summaries = []
    all_vecs = []

    for dck_file in dck_files:
        deck_name, card_names = _parse_dck_file(dck_file)
        resolved = []
        for name in card_names:
            key = resolve_name(name)
            if key is not None:
                resolved.append(key)

        if not resolved:
            deck_summaries.append({
                "name": deck_name,
                "total_cards": len(card_names),
                "resolved": 0,
                "top_mechanics": [],
            })
            continue

        # Get vectors and compute centroid
        vecs = np.array([mechanics[card_index[name]] for name in resolved], dtype=np.float64)
        centroid = vecs.mean(axis=0)
        all_vecs.append(vecs)

        # Top mechanics (skip generic ones)
        top_indices = np.argsort(-centroid)[:30]
        top_mechs = []
        for idx in top_indices:
            if centroid[idx] < 0.05:
                break
            mname = mechanic_names.get(idx, "")
            if mname and mname not in skip_display:
                top_mechs.append((mname, float(centroid[idx])))
            if len(top_mechs) >= 10:
                break

        deck_summaries.append({
            "name": deck_name,
            "total_cards": len(card_names),
            "resolved": len(resolved),
            "top_mechanics": top_mechs,
        })

    # Compute global centroid across all decks
    global_top_mechs = []
    if all_vecs:
        combined = np.vstack(all_vecs)
        global_centroid = combined.mean(axis=0)
        top_indices = np.argsort(-global_centroid)[:30]
        for idx in top_indices:
            if global_centroid[idx] < 0.05:
                break
            mname = mechanic_names.get(idx, "")
            if mname and mname not in skip_display:
                global_top_mechs.append((mname, float(global_centroid[idx])))
            if len(global_top_mechs) >= 15:
                break

    # Layout: top half = deck listing table, bottom half = global centroid bar chart
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.3], hspace=0.35, top=0.94, bottom=0.06)

    # --- Top: Deck listing ---
    ax_table = fig.add_subplot(gs[0])
    ax_table.axis("off")

    # Build deck summary text
    lines = ["Training Decks", "=" * 55, ""]
    for ds in deck_summaries:
        pct = (ds["resolved"] / ds["total_cards"] * 100) if ds["total_cards"] > 0 else 0
        lines.append(f"  {ds['name']:<30} {ds['total_cards']:>3} cards ({ds['resolved']} resolved, {pct:.0f}%)")
        if ds["top_mechanics"]:
            top3 = [f"{m[0].replace('_', ' ').title()}" for m in ds["top_mechanics"][:5]]
            lines.append(f"    Top mechanics: {', '.join(top3)}")
        lines.append("")

    lines.append(f"  Total decks: {len(dck_files)}")

    deck_text = "\n".join(lines)
    ax_table.text(
        0.05, 0.95, deck_text,
        transform=ax_table.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f5f0e8", alpha=0.6),
    )

    # --- Bottom: Global centroid bar chart ---
    ax_bar = fig.add_subplot(gs[1])

    if global_top_mechs:
        mech_labels = [m[0].replace("_", " ").title() for m in reversed(global_top_mechs)]
        mech_values = [m[1] for m in reversed(global_top_mechs)]

        colors = plt.cm.viridis([v / max(mech_values) for v in mech_values])
        ax_bar.barh(range(len(mech_labels)), mech_values, color=colors, edgecolor="grey", linewidth=0.5)
        ax_bar.set_yticks(range(len(mech_labels)))
        ax_bar.set_yticklabels(mech_labels, fontsize=8)
        ax_bar.set_xlabel("Mean Centroid Weight", fontsize=9)
        ax_bar.set_title("Global Mechanics Centroid (All Training Decks)", fontweight="bold", fontsize=10)
        ax_bar.grid(axis="x", alpha=0.3)

        # Add value labels
        for i, v in enumerate(mech_values):
            ax_bar.text(v + max(mech_values) * 0.01, i, f"{v:.2f}", va="center", fontsize=7)
    else:
        ax_bar.text(
            0.5, 0.5, "No card data resolved for centroid analysis",
            transform=ax_bar.transAxes, ha="center", va="center",
            fontsize=12, color="grey",
        )
        ax_bar.set_title("Global Mechanics Centroid", fontweight="bold", fontsize=10)

    fig.suptitle("Page 6 — Deck Mechanics Analysis", fontsize=9, y=0.02, color="grey")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_wandb_url(metrics: Dict[str, Any]) -> str:
    entity = metrics.get("wandb_entity", WANDB_ENTITY)
    project = metrics.get("wandb_project", WANDB_PROJECT)
    if entity:
        return f"https://wandb.ai/{entity}/{project}"
    return f"https://wandb.ai/<your-entity>/{project}"


def _dummy_metrics() -> Dict[str, Any]:
    """Placeholder metrics for testing the report layout."""
    return {
        "model_name": "AlphaZero policy/value network",
        "checkpoint_path": "checkpoints/bc_best.pt",
        "total_time_s": 5400,
        "epochs": 10,
        "policy_loss": 0.35,
        "value_loss": 0.12,
        "accuracy": 0.92,
        "best_checkpoint": "checkpoints/bc_best.pt",
        "games_played": 1000,
        "win_rate": 62.5,
        "history": [
            {"loss": 1.5, "accuracy": 0.60},
            {"loss": 1.2, "accuracy": 0.65},
            {"loss": 0.9, "accuracy": 0.72},
            {"loss": 0.7, "accuracy": 0.78},
            {"loss": 0.6, "accuracy": 0.82},
            {"loss": 0.5, "accuracy": 0.85},
            {"loss": 0.45, "accuracy": 0.87},
            {"loss": 0.4, "accuracy": 0.89},
            {"loss": 0.38, "accuracy": 0.91},
            {"loss": 0.35, "accuracy": 0.92},
        ],
    }


def _dummy_timing() -> Dict[str, float]:
    """Placeholder timing for testing the report layout."""
    return {
        "data_collection": 3000.0,
        "data_encoding": 840.0,
        "training": 1200.0,
        "evaluation": 360.0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate training report PDF and optionally email it"
    )
    parser.add_argument(
        "--to",
        help="Recipient email (overrides FORGERL_NOTIFY_EMAIL env var)",
    )
    parser.add_argument(
        "--subject",
        default="ForgeRL Training Report",
        help="Email subject line",
    )
    parser.add_argument(
        "--pdf",
        help="Path to existing PDF to send (skip generation)",
    )
    parser.add_argument(
        "--pdf-only",
        "--save-only",
        action="store_true",
        help="Generate PDF without sending email (save locally only)",
    )
    parser.add_argument(
        "--tb-log-dir",
        default="runs",
        help="TensorBoard log directory to read curves from",
    )
    args = parser.parse_args()

    # ── Get or create PDF ──────────────────────────────────────────────
    if args.pdf:
        pdf_path = args.pdf
        if not Path(pdf_path).exists():
            logger.error(f"PDF not found: {pdf_path}")
            sys.exit(1)
        logger.info(f"Using existing PDF: {pdf_path}")
    else:
        logger.info("Generating training report PDF...")
        pdf_path = generate_training_report(tb_log_dir=args.tb_log_dir)

    if args.pdf_only:
        print(f"\nPDF generated: {pdf_path}")
        return

    # ── Send email ─────────────────────────────────────────────────────
    if args.to:
        os.environ["FORGERL_NOTIFY_EMAIL"] = args.to
        logger.info(f"Overriding recipient: {args.to}")

    try:
        notifier = EmailNotifier()
    except ValueError as e:
        logger.error(str(e))
        logger.info("Set FORGERL_NOTIFY_EMAIL or use --to flag")
        logger.info("Run ./scripts/setup_email.sh for interactive setup")
        sys.exit(1)

    # Check if SMTP password is configured
    if not notifier.smtp_pass:
        logger.error("SMTP password not configured")
        logger.info("")
        logger.info("Gmail with 2FA requires an App Password (not your regular password).")
        logger.info("Run the setup wizard: ./scripts/setup_email.sh")
        logger.info("")
        logger.info("Or generate a PDF without email: --save-only")
        sys.exit(1)

    body = (
        "ForgeRL training report attached.\n\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Report: {Path(pdf_path).name}\n"
    )

    logger.info("Sending email...")
    success = notifier.send_report(
        subject=args.subject,
        body_text=body,
        attachments=[pdf_path],
    )

    if success:
        print(f"\nEmail sent successfully to {notifier.recipient}")
        print(f"  Subject: {args.subject}")
        print(f"  Attachment: {Path(pdf_path).name}")
    else:
        print("\nFailed to send email. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
