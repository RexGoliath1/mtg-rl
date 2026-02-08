#!/usr/bin/env python3
"""Training report generator and email notifier.

Creates a comprehensive multi-page PDF covering:
  1. Training Summary — model info, loss, checkpoint
  2. Pipeline Timing Profile — horizontal bar chart + table
  3. Monitoring Links — TensorBoard, W&B, S3
  4. Training Curves — loss/accuracy over epochs (from TensorBoard logs or metrics dict)

Usage:
    # Generate + send dummy report (for testing email pipeline):
    FORGERL_NOTIFY_EMAIL=user@example.com uv run python3 scripts/send_test_report.py

    # Send an existing PDF:
    uv run python3 scripts/send_test_report.py --pdf data/reports/vocab_health_2026-02-07.pdf

    # Generate PDF only (no email):
    uv run python3 scripts/send_test_report.py --pdf-only

    # Custom subject:
    uv run python3 scripts/send_test_report.py --subject "Nightly Training Report"
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
        _page_pipeline_timing(pdf, timing, timing_detailed)
        _page_monitoring_links(pdf, metrics)
        _page_training_curves(pdf, metrics, tb_log_dir)

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


# -- Page 2: Pipeline Timing Profile ----------------------------------------

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

    fig.suptitle("Page 2 — Pipeline Timing Profile", fontsize=9, y=0.02, color="grey")
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

    fig.suptitle("Page 3 — Monitoring Links", fontsize=9, y=0.02, color="grey")
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

    fig.suptitle("Page 4 — Training Curves", fontsize=9, y=0.02, color="grey")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
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
        action="store_true",
        help="Generate PDF without sending email",
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
