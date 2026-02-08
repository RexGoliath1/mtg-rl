#!/usr/bin/env python3
"""Training report generator and email notifier.

Creates a professional LaTeX-based PDF combining project overview (whitepaper
content) with dynamic training data:
  1. Cover Page — title, date, model name, run ID
  2. Project Overview — ForgeRL description, mechanics encoding, AlphaZero architecture
  3. Network Architecture — embedded diagram with parameter count table
  4. Training Summary — model info, checkpoint, epochs, loss values (table)
  5. Pipeline Timing Profile — horizontal bar chart (PNG) + timing table
  6. Training Curves — loss/accuracy plots (PNG) from TensorBoard or metrics
  7. Deck Mechanics Analysis — centroid bar chart (PNG) + deck listing table
  8. Monitoring & Artifact Links — TensorBoard, W&B, S3, quick commands

Falls back to matplotlib PdfPages if pdflatex is not installed.

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
import shutil
import subprocess
import sys
import tempfile
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
S3_BUCKET = "mtg-rl-checkpoints-20260124190118616600000001"
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "mtg-selfplay")


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def _tex_escape(s: str) -> str:
    """Escape special LaTeX characters."""
    for ch in ["&", "%", "$", "#", "_", "{", "}"]:
        s = s.replace(ch, "\\" + ch)
    s = s.replace("~", "\\textasciitilde{}")
    s = s.replace("^", "\\textasciicircum{}")
    return s


def _fmt_params(n: int) -> str:
    """Format a parameter count like 33.1M or 530.0K."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Project data collection (adapted from generate_whitepaper.py)
# ---------------------------------------------------------------------------

def _get_network_params() -> dict:
    """Instantiate the AlphaZero network and count parameters."""
    try:
        from forge.game_state_encoder import ForgeGameStateEncoder, GameStateConfig
        from forge.policy_value_heads import PolicyHead, ValueHead, PolicyValueConfig, ActionConfig

        cfg = GameStateConfig()
        enc = ForgeGameStateEncoder(cfg)
        enc_params = sum(p.numel() for p in enc.parameters())

        pv_cfg = PolicyValueConfig()
        ac = ActionConfig()
        pol = PolicyHead(pv_cfg)
        val = ValueHead(pv_cfg, num_players=2)
        pol_params = sum(p.numel() for p in pol.parameters())
        val_params = sum(p.numel() for p in val.parameters())

        return {
            "encoder_params": enc_params,
            "policy_params": pol_params,
            "value_params": val_params,
            "total_params": enc_params + pol_params + val_params,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "d_ff": cfg.d_ff,
            "output_dim": cfg.output_dim,
            "zone_emb_dim": cfg.zone_embedding_dim,
            "global_emb_dim": cfg.global_embedding_dim,
            "action_dim": ac.total_actions,
            "policy_hidden": pv_cfg.policy_hidden_dim,
            "value_hidden": pv_cfg.value_hidden_dim,
            "dropout": cfg.dropout,
        }
    except Exception as e:
        logger.warning(f"Could not load network ({e}), using defaults")
        return {
            "encoder_params": 33_100_000, "policy_params": 530_000,
            "value_params": 300_000, "total_params": 33_930_000,
            "d_model": 512, "n_heads": 8, "n_layers": 3, "d_ff": 1024,
            "output_dim": 768, "zone_emb_dim": 512, "global_emb_dim": 192,
            "action_dim": 203, "policy_hidden": 384, "value_hidden": 384,
            "dropout": 0.1,
        }


def _get_vocab_stats() -> dict:
    """Get vocabulary size and mechanic count."""
    try:
        from mechanics.vocabulary import VOCAB_SIZE, Mechanic
        return {"vocab_size": VOCAB_SIZE, "mechanic_count": len(Mechanic)}
    except Exception:
        return {"vocab_size": 1387, "mechanic_count": 426}


def _get_h5_stats() -> dict:
    """Read stats from HDF5 embeddings file."""
    try:
        import h5py
        for name in ["card_mechanics_commander.h5", "card_mechanics_standard.h5"]:
            p = PROJECT_ROOT / "data" / name
            if p.exists():
                with h5py.File(p, "r") as f:
                    return {
                        "card_count": int(f.attrs.get("card_count", f["mechanics"].shape[0])),
                        "vocab_size": int(f.attrs["vocab_size"]),
                        "format": name.replace("card_mechanics_", "").replace(".h5", ""),
                        "file_size_mb": p.stat().st_size / 1024 / 1024,
                    }
    except Exception:
        pass
    return {"card_count": 30462, "vocab_size": 1387, "format": "commander", "file_size_mb": 1.34}


# ---------------------------------------------------------------------------
# Chart generation helpers (save matplotlib figures as PNGs for LaTeX embedding)
# ---------------------------------------------------------------------------

def _generate_timing_chart(timing: Dict[str, float], output_path: str) -> bool:
    """Generate horizontal bar chart for pipeline timing. Returns True on success."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        stages = list(timing.keys())
        durations = list(timing.values())
        total = sum(durations) or 1.0

        fig, ax = plt.subplots(figsize=(7, max(2.5, len(stages) * 0.6)))
        colors = plt.cm.Set2([i / max(len(stages), 1) for i in range(len(stages))])
        y_pos = range(len(stages))
        ax.barh(y_pos, durations, color=colors, edgecolor="grey", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(stages, fontsize=9)
        ax.set_xlabel("Seconds")
        ax.set_title("Pipeline Stage Durations", fontweight="bold")
        ax.invert_yaxis()

        for i, v in enumerate(durations):
            pct = v / total * 100
            ax.text(v + total * 0.01, i, f"{v:.1f}s ({pct:.0f}%)", va="center", fontsize=8)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        logger.warning(f"Failed to generate timing chart: {e}")
        return False


def _generate_training_curves(metrics: Dict[str, Any], tb_log_dir: str, output_path: str) -> bool:
    """Generate loss/accuracy plots. Returns True on success."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(7, 6))
        ax_loss, ax_acc = axes

        # Try TensorBoard first
        loss_data = _read_tb_scalars(tb_log_dir, "imitation/loss")
        acc_data = _read_tb_scalars(tb_log_dir, "imitation/accuracy")

        # Fall back to metrics["history"]
        history = metrics.get("history", [])
        if not loss_data and history:
            loss_data = [(i, h.get("loss", h.get("policy_loss", 0))) for i, h in enumerate(history)]
        if not acc_data and history:
            acc_data = [(i, h.get("accuracy", 0)) for i, h in enumerate(history)]

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
            ax_loss.text(0.5, 0.5, "No training loss data available yet",
                         transform=ax_loss.transAxes, ha="center", va="center", fontsize=14, color="grey")
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
            ax_acc.text(0.5, 0.5, "No training accuracy data available yet",
                        transform=ax_acc.transAxes, ha="center", va="center", fontsize=14, color="grey")
            ax_acc.set_title("Training Accuracy", fontweight="bold")

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        logger.warning(f"Failed to generate training curves: {e}")
        return False


def _generate_deck_centroid_chart(output_path: str) -> tuple:
    """Generate deck centroid bar chart. Returns (success, deck_summaries)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        mechanics, card_index, vocab_size, mechanic_names = _load_h5_for_analysis()
        if mechanics is None:
            return False, []

        decks_dir = PROJECT_ROOT / "decks"
        dck_files = sorted(decks_dir.glob("*.dck")) if decks_dir.is_dir() else []
        if not dck_files:
            return False, []

        # Build front-face index for DFC resolution
        front_face_index: dict = {}
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

        skip_display = {
            "TRIGGERED_ABILITY", "ACTIVATED_ABILITY", "SORCERY_SPEED", "INSTANT_SPEED",
            "UNTIL_END_OF_TURN", "TARGET_CREATURE", "TARGET_PLAYER",
            "ADD_MANA", "MANA_OF_ANY_COLOR", "MANA_FIXING", "TO_BATTLEFIELD_TAPPED",
            "TUTOR_LAND", "SEARCH_LIBRARY", "TAP_FOR_EFFECT", "SACRIFICE_COST",
            "ENTERS_THE_BATTLEFIELD", "MANA_COST", "COLOR_IDENTITY",
        }

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
                deck_summaries.append({"name": deck_name, "total_cards": len(card_names), "resolved": 0, "top_mechanics": []})
                continue

            vecs = np.array([mechanics[card_index[name]] for name in resolved], dtype=np.float64)
            centroid = vecs.mean(axis=0)
            all_vecs.append(vecs)

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

            deck_summaries.append({"name": deck_name, "total_cards": len(card_names), "resolved": len(resolved), "top_mechanics": top_mechs})

        # Global centroid chart
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

        if global_top_mechs:
            fig, ax = plt.subplots(figsize=(7, max(3, len(global_top_mechs) * 0.4)))
            mech_labels = [m[0].replace("_", " ").title() for m in reversed(global_top_mechs)]
            mech_values = [m[1] for m in reversed(global_top_mechs)]

            colors = plt.cm.viridis([v / max(mech_values) for v in mech_values])
            ax.barh(range(len(mech_labels)), mech_values, color=colors, edgecolor="grey", linewidth=0.5)
            ax.set_yticks(range(len(mech_labels)))
            ax.set_yticklabels(mech_labels, fontsize=8)
            ax.set_xlabel("Mean Centroid Weight", fontsize=9)
            ax.set_title("Global Mechanics Centroid (All Training Decks)", fontweight="bold", fontsize=10)
            ax.grid(axis="x", alpha=0.3)
            for i, v in enumerate(mech_values):
                ax.text(v + max(mech_values) * 0.01, i, f"{v:.2f}", va="center", fontsize=7)

            fig.tight_layout()
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return True, deck_summaries

        return False, deck_summaries
    except Exception as e:
        logger.warning(f"Failed to generate deck centroid chart: {e}")
        return False, []


# ---------------------------------------------------------------------------
# LaTeX template builder
# ---------------------------------------------------------------------------

def _build_latex_document(
    metrics: Dict[str, Any],
    timing: Dict[str, float],
    timing_detailed: Optional[Dict[str, Dict[str, float]]],
    net: dict,
    vocab: dict,
    h5: dict,
    deck_summaries: list,
    timing_chart_path: Optional[str],
    curves_chart_path: Optional[str],
    centroid_chart_path: Optional[str],
    arch_image_path: Optional[str],
) -> str:
    """Build the complete LaTeX source for the training report."""
    today = datetime.now().strftime("%B %d, %Y")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    year = datetime.now().strftime("%Y")

    model_name = _tex_escape(metrics.get("model_name", "AlphaZero policy/value"))
    ckpt = _tex_escape(metrics.get("checkpoint_path", "checkpoints/latest.pt"))
    total_time = metrics.get("total_time_s", 0)
    epochs = metrics.get("epochs", 0)
    policy_loss = metrics.get("policy_loss", float("nan"))
    value_loss = metrics.get("value_loss", float("nan"))
    accuracy = metrics.get("accuracy", float("nan"))
    best_ckpt = _tex_escape(metrics.get("best_checkpoint", metrics.get("checkpoint_path", "N/A")))
    games = metrics.get("games_played", 0)
    win_rate = metrics.get("win_rate", None)

    hours = int(total_time / 3600)
    mins = int((total_time % 3600) / 60)

    # --- Architecture figure ---
    arch_figure = ""
    if arch_image_path:
        arch_figure = rf"""
\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.85\textwidth]{{{arch_image_path}}}
  \caption{{AlphaZero network architecture for ForgeRL. The state encoder produces a
  {net['output_dim']}-dimensional embedding consumed by both policy and value heads.}}
  \label{{fig:architecture}}
\end{{figure}}
"""

    # --- Timing chart figure ---
    timing_figure = ""
    if timing_chart_path:
        timing_figure = rf"""
\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.9\textwidth]{{{timing_chart_path}}}
  \caption{{Pipeline stage durations for this training run.}}
  \label{{fig:timing}}
\end{{figure}}
"""

    # --- Timing table rows ---
    stages = list(timing.keys())
    durations = list(timing.values())
    total_dur = sum(durations) or 1.0

    if timing_detailed:
        timing_table_cols = r"\textbf{Stage} & \textbf{Total (s)} & \textbf{Count} & \textbf{Mean (s)} & \textbf{\% of Wall} \\"
        timing_table_rows = []
        for stage in stages:
            d = timing_detailed.get(stage, {})
            sname = _tex_escape(stage)
            total_s = d.get("total_s", timing.get(stage, 0))
            count = d.get("count", 1)
            mean_s = d.get("mean_s", 0)
            pct = d.get("pct_of_wall", 0)
            timing_table_rows.append(f"{sname} & {total_s:.2f} & {count} & {mean_s:.3f} & {pct:.1f}\\% \\\\")
    else:
        timing_table_cols = r"\textbf{Stage} & \textbf{Duration (s)} & \textbf{\% of Total} \\"
        timing_table_rows = []
        for s, d in zip(stages, durations):
            timing_table_rows.append(f"{_tex_escape(s)} & {d:.2f} & {d / total_dur * 100:.1f}\\% \\\\")
    timing_table_rows_str = "\n".join(timing_table_rows)
    timing_col_spec = "@{}lrrrr@{}" if timing_detailed else "@{}lrr@{}"

    # --- Training curves figure ---
    curves_figure = ""
    if curves_chart_path:
        curves_figure = rf"""
\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.9\textwidth]{{{curves_chart_path}}}
  \caption{{Training loss and accuracy over epochs.}}
  \label{{fig:curves}}
\end{{figure}}
"""
    else:
        curves_figure = r"\textit{No training curve data available yet (TensorBoard logs or metrics history not found).}"

    # --- Deck summaries table ---
    deck_table_rows = []
    for ds in deck_summaries:
        dname = _tex_escape(ds["name"])
        tc = ds["total_cards"]
        rc = ds["resolved"]
        pct = (rc / tc * 100) if tc > 0 else 0
        top3 = ", ".join(m[0].replace("_", " ").title() for m, _ in zip(ds.get("top_mechanics", [])[:3], range(3)))
        if not top3:
            top3 = "---"
        top3 = _tex_escape(top3)
        deck_table_rows.append(f"{dname} & {tc} & {rc} ({pct:.0f}\\%) & {top3} \\\\")
    deck_table_rows_str = "\n".join(deck_table_rows) if deck_table_rows else r"\textit{No decks found} & & & \\"

    # --- Centroid chart figure ---
    centroid_figure = ""
    if centroid_chart_path:
        centroid_figure = rf"""
\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.9\textwidth]{{{centroid_chart_path}}}
  \caption{{Global mechanics centroid computed across all training decks.}}
  \label{{fig:centroid}}
\end{{figure}}
"""

    # --- Monitoring links ---
    tb_url = metrics.get("tensorboard_url", "http://localhost:6006")
    wandb_url = _build_wandb_url(metrics)
    s3_path = metrics.get("s3_path", f"s3://{S3_BUCKET}/")
    tb_log_dir_display = _tex_escape(metrics.get("tb_log_dir", "runs/"))

    # --- Win rate row ---
    win_rate_row = ""
    if win_rate is not None:
        win_rate_row = rf"Win rate & {win_rate:.1f}\% \\"

    # --- Accuracy row ---
    accuracy_row = ""
    if accuracy == accuracy:  # not NaN
        accuracy_row = rf"Accuracy & {accuracy:.3f} \\"

    vocab_size = vocab.get("vocab_size", 1387)

    return rf"""\documentclass[11pt,letterpaper]{{article}}

% ---------------------------------------------------------------------------
% Packages
% ---------------------------------------------------------------------------
\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\usepackage{{fancyhdr}}
\usepackage{{titlesec}}
\usepackage{{enumitem}}
\usepackage{{float}}
\usepackage{{caption}}
\usepackage{{tabularx}}
\usepackage{{listings}}

% ---------------------------------------------------------------------------
% Style
% ---------------------------------------------------------------------------
\definecolor{{forgeblue}}{{HTML}}{{1a3a5c}}
\definecolor{{forgegrey}}{{HTML}}{{666666}}

\hypersetup{{
  colorlinks=true,
  linkcolor=forgeblue,
  citecolor=forgeblue,
  urlcolor=forgeblue,
}}

\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{\small\textit{{ForgeRL Training Report}}}}
\fancyhead[R]{{\small\thepage}}
\fancyfoot[C]{{\footnotesize Generated by \texttt{{send\_test\_report.py}} \textbullet\ ForgeRL {year}}}
\renewcommand{{\headrulewidth}}{{0.4pt}}
\renewcommand{{\footrulewidth}}{{0.4pt}}

\titleformat{{\section}}{{\Large\bfseries\color{{forgeblue}}}}{{\thesection.}}{{0.5em}}{{}}
\titleformat{{\subsection}}{{\large\bfseries}}{{\thesubsection}}{{0.5em}}{{}}

\captionsetup{{font=small,labelfont=bf}}

% ---------------------------------------------------------------------------
\begin{{document}}

% ===== COVER PAGE ==========================================================
\begin{{titlepage}}
\vspace*{{2cm}}
\begin{{center}}

{{\Huge\bfseries\color{{forgeblue}} ForgeRL Training Report}}\\[0.8cm]
{{\Large\color{{forgegrey}} Reinforcement Learning for Magic: The Gathering}}

\vspace{{2cm}}

{{\large {today}}}

\vspace{{1.5cm}}

\begin{{tabular}}{{ll}}
\textbf{{Model:}} & {model_name} \\[0.2cm]
\textbf{{Total Parameters:}} & {_fmt_params(net['total_params'])} \\[0.2cm]
\textbf{{Checkpoint:}} & \texttt{{{ckpt}}} \\[0.2cm]
\textbf{{Epochs:}} & {epochs} \\[0.2cm]
\textbf{{Wall Time:}} & {hours}h {mins}m \\
\end{{tabular}}

\vspace{{2cm}}

\begin{{tabular}}{{l}}
\textbf{{Repository:}} \url{{https://github.com/RexGoliath1/mtg}} \\[0.2cm]
\textbf{{Forge Engine:}} \url{{https://github.com/RexGoliath1/forge}} \\
\end{{tabular}}

\vfill
{{\small Generated: {now}}}
\end{{center}}
\end{{titlepage}}

% ===== TABLE OF CONTENTS ===================================================
\tableofcontents
\newpage

% ===== SECTION 1: PROJECT OVERVIEW =========================================
\section{{Project Overview}}
\label{{sec:overview}}

\textbf{{ForgeRL}} is a reinforcement learning system for Magic: The Gathering
draft and gameplay, built on top of the Forge open-source MTG engine.
The goal is to train an AI that can draft competitively and play games using
human gameplay data (17lands.com) and self-play reinforcement learning.

\subsection{{Mechanics-Based Card Encoding}}

Rather than using text embeddings or one-hot card identifiers, ForgeRL
decomposes each card into a multi-hot vector over \textbf{{{vocab_size}}} atomic
mechanics primitives. This design enables:

\begin{{itemize}}[nosep]
  \item \textbf{{Compositional generalization}} --- new cards are novel combinations
        of known primitives.
  \item \textbf{{Format transfer}} --- the same vocabulary covers Draft, Standard,
        and Commander.
  \item \textbf{{Compact storage}} --- all {h5['card_count']:,} {h5['format']}-legal cards
        fit in {h5['file_size_mb']:.2f}\,MB (HDF5).
\end{{itemize}}

\subsection{{AlphaZero Architecture}}

ForgeRL employs an AlphaZero-style architecture comprising a shared state
encoder, a policy head, and a value head. The system is designed for the
complex, partially observable, variable-action environment of Magic: The Gathering.

\begin{{enumerate}}[nosep]
  \item \textbf{{State Encoder}} --- Card Embedding MLP + per-zone self-attention
        + stack encoder + global encoder + cross-zone attention
        $\to \mathbb{{R}}^{{{net['output_dim']}}}$.
  \item \textbf{{Policy Head}} --- Maps state embedding to a probability distribution
        over the {net['action_dim']}-dimensional action space with legal-action masking.
  \item \textbf{{Value Head}} --- Estimates expected game outcome ($\tanh$ for 1v1,
        softmax for multiplayer).
\end{{enumerate}}

Training proceeds in three phases: behavioral cloning on human data (17lands),
reinforcement learning via self-play with MCTS, and evaluation against baselines.


% ===== SECTION 2: NETWORK ARCHITECTURE ====================================
\newpage
\section{{Network Architecture}}
\label{{sec:architecture}}

{arch_figure}

\subsection{{Parameter Breakdown}}

\begin{{table}}[H]
\centering
\caption{{Parameter count by component.}}
\label{{tab:params}}
\begin{{tabular}}{{@{{}}lrr@{{}}}}
\toprule
\textbf{{Component}} & \textbf{{Parameters}} & \textbf{{\% of Total}} \\
\midrule
State Encoder & {net['encoder_params']:,} & {net['encoder_params']/net['total_params']*100:.1f}\% \\
Policy Head   & {net['policy_params']:,}  & {net['policy_params']/net['total_params']*100:.1f}\% \\
Value Head    & {net['value_params']:,}   & {net['value_params']/net['total_params']*100:.1f}\% \\
\midrule
\textbf{{Total}} & \textbf{{{net['total_params']:,}}} & 100.0\% \\
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{Key Dimensions}}

\begin{{table}}[H]
\centering
\caption{{Architecture hyperparameters.}}
\begin{{tabular}}{{@{{}}lr@{{}}}}
\toprule
\textbf{{Parameter}} & \textbf{{Value}} \\
\midrule
$d_{{\text{{model}}}}$ & {net['d_model']} \\
$d_{{\text{{ff}}}}$ & {net['d_ff']} \\
Attention heads & {net['n_heads']} \\
Cross-zone layers & {net['n_layers']} \\
Output dimension & {net['output_dim']} \\
Global embedding dim & {net['global_emb_dim']} \\
Policy hidden dim & {net['policy_hidden']} \\
Value hidden dim & {net['value_hidden']} \\
Dropout & {net['dropout']} \\
Vocabulary size & {vocab_size} \\
\bottomrule
\end{{tabular}}
\end{{table}}


% ===== SECTION 3: TRAINING SUMMARY ========================================
\newpage
\section{{Training Summary}}
\label{{sec:summary}}

\begin{{table}}[H]
\centering
\caption{{Training run summary.}}
\label{{tab:summary}}
\begin{{tabular}}{{@{{}}lr@{{}}}}
\toprule
\textbf{{Metric}} & \textbf{{Value}} \\
\midrule
Model & {model_name} \\
Checkpoint & \texttt{{{ckpt}}} \\
Best checkpoint & \texttt{{{best_ckpt}}} \\
Total time & {hours}h {mins}m ({total_time:.0f}s) \\
Epochs completed & {epochs} \\
Games played & {games:,} \\
{win_rate_row}
\midrule
Policy loss & {policy_loss:.4f} \\
Value loss & {value_loss:.4f} \\
{accuracy_row}
\bottomrule
\end{{tabular}}
\end{{table}}


% ===== SECTION 4: PIPELINE TIMING PROFILE =================================
\section{{Pipeline Timing Profile}}
\label{{sec:timing}}

{timing_figure}

\begin{{table}}[H]
\centering
\caption{{Detailed timing breakdown by pipeline stage.}}
\label{{tab:timing}}
\begin{{tabular}}{{{timing_col_spec}}}
\toprule
{timing_table_cols}
\midrule
{timing_table_rows_str}
\midrule
\textbf{{Total}} & \textbf{{{total_dur:.2f}}} & {"& &" if timing_detailed else ""} \\
\bottomrule
\end{{tabular}}
\end{{table}}


% ===== SECTION 5: TRAINING CURVES =========================================
\newpage
\section{{Training Curves}}
\label{{sec:curves}}

{curves_figure}


% ===== SECTION 6: DECK MECHANICS ANALYSIS =================================
\section{{Deck Mechanics Analysis}}
\label{{sec:decks}}

\begin{{table}}[H]
\centering
\caption{{Training deck summary with top mechanics.}}
\label{{tab:decks}}
\begin{{tabular}}{{@{{}}lrrp{{5cm}}@{{}}}}
\toprule
\textbf{{Deck}} & \textbf{{Cards}} & \textbf{{Resolved}} & \textbf{{Top Mechanics}} \\
\midrule
{deck_table_rows_str}
\bottomrule
\end{{tabular}}
\end{{table}}

{centroid_figure}


% ===== SECTION 7: MONITORING & ARTIFACT LINKS =============================
\newpage
\section{{Monitoring \& Artifact Links}}
\label{{sec:monitoring}}

\subsection{{TensorBoard}}

\begin{{tabular}}{{@{{}}ll@{{}}}}
Local URL & \url{{{tb_url}}} \\
Log directory & \texttt{{{tb_log_dir_display}}} \\
\end{{tabular}}

\medskip
\noindent Start locally:
\begin{{verbatim}}
tensorboard --logdir runs/ --port 6006
\end{{verbatim}}

\subsection{{Weights \& Biases}}

Dashboard: \url{{{wandb_url}}}

\subsection{{S3 Checkpoints}}

\begin{{tabular}}{{@{{}}ll@{{}}}}
Bucket & \texttt{{{_tex_escape(s3_path)}}} \\
Latest checkpoint & \texttt{{{ckpt}}} \\
\end{{tabular}}

\medskip
\noindent Download latest checkpoint:
\begin{{verbatim}}
aws s3 cp {s3_path}checkpoints/ checkpoints/ --recursive
\end{{verbatim}}

\subsection{{Quick Commands}}

\begin{{verbatim}}
# View TensorBoard locally
tensorboard --logdir runs/ --port 6006

# Download latest checkpoint from S3
aws s3 cp {s3_path}checkpoints/ checkpoints/ --recursive

# Tail training logs on EC2
./scripts/ssh-instance.sh logs
\end{{verbatim}}


\end{{document}}
"""


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
    """Generate a professional LaTeX training report PDF.

    Falls back to matplotlib PdfPages if pdflatex is not installed.

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
    if metrics is None:
        metrics = _dummy_metrics()
    if timing is None:
        timing = _dummy_timing()

    output_dir = Path("data/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(output_dir / f"training_report_{timestamp}.pdf")

    # Check for pdflatex
    has_pdflatex = shutil.which("pdflatex") is not None
    if not has_pdflatex:
        logger.warning("pdflatex not found -- falling back to matplotlib PDF generation")
        logger.warning("Install LaTeX for professional formatting: brew install --cask mactex-no-gui")
        return _generate_report_matplotlib_fallback(metrics, timing, timing_detailed, output_path, tb_log_dir)

    # Collect project data for the overview sections
    logger.info("Collecting project data...")
    net = _get_network_params()
    vocab = _get_vocab_stats()
    h5 = _get_h5_stats()

    # Generate chart PNGs into a temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Timing chart
        timing_chart_path = os.path.join(tmpdir, "timing_chart.png")
        has_timing_chart = _generate_timing_chart(timing, timing_chart_path)
        if not has_timing_chart:
            timing_chart_path = None

        # Training curves chart
        curves_chart_path = os.path.join(tmpdir, "training_curves.png")
        has_curves = _generate_training_curves(metrics, tb_log_dir, curves_chart_path)
        if not has_curves:
            curves_chart_path = None

        # Deck centroid chart
        centroid_chart_path = os.path.join(tmpdir, "deck_centroid.png")
        has_centroid, deck_summaries = _generate_deck_centroid_chart(centroid_chart_path)
        if not has_centroid:
            centroid_chart_path = None

        # Architecture image
        arch_img = PROJECT_ROOT / "data" / "reports" / "network_architecture.png"
        arch_image_path = str(arch_img) if arch_img.exists() else None

        # Build LaTeX source
        tex_source = _build_latex_document(
            metrics=metrics,
            timing=timing,
            timing_detailed=timing_detailed,
            net=net,
            vocab=vocab,
            h5=h5,
            deck_summaries=deck_summaries,
            timing_chart_path=timing_chart_path,
            curves_chart_path=curves_chart_path,
            centroid_chart_path=centroid_chart_path,
            arch_image_path=arch_image_path,
        )

        # Write .tex and compile
        tex_file = os.path.join(tmpdir, "training_report.tex")
        with open(tex_file, "w") as f:
            f.write(tex_source)

        for pass_num in (1, 2):
            logger.info(f"pdflatex pass {pass_num}/2...")
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, tex_file],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0 and pass_num == 2:
                logger.warning("pdflatex warnings/errors (may still produce PDF):")
                for line in result.stdout.splitlines():
                    if line.startswith("!") or "Error" in line:
                        logger.warning(f"  {line}")

        pdf_tmp = os.path.join(tmpdir, "training_report.pdf")
        if os.path.exists(pdf_tmp):
            shutil.copy2(pdf_tmp, output_path)
            logger.info(f"Created LaTeX training report PDF: {output_path}")
            size_kb = os.path.getsize(output_path) / 1024
            logger.info(f"  Size: {size_kb:.0f} KB")
            return output_path

        # LaTeX compilation failed -- fall back to matplotlib
        logger.warning("pdflatex compilation failed, falling back to matplotlib")
        # Save .tex for debugging
        tex_out = Path(output_path).with_suffix(".tex")
        tex_out.write_text(tex_source)
        logger.info(f"  TeX source saved for debugging: {tex_out}")

    return _generate_report_matplotlib_fallback(metrics, timing, timing_detailed, output_path, tb_log_dir)


def _generate_report_matplotlib_fallback(
    metrics: Dict[str, Any],
    timing: Dict[str, float],
    timing_detailed: Optional[Dict[str, Dict[str, float]]],
    output_path: str,
    tb_log_dir: str,
) -> str:
    """Fallback: generate a matplotlib PdfPages report (original approach)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        logger.error("matplotlib not installed. Install with: uv sync --extra dev")
        sys.exit(1)

    with PdfPages(output_path) as pdf:
        _page_training_summary(pdf, metrics)
        _page_network_architecture(pdf)
        _page_pipeline_timing(pdf, timing, timing_detailed)
        _page_monitoring_links(pdf, metrics)
        _page_training_curves(pdf, metrics, tb_log_dir)
        _page_deck_mechanics(pdf)

    logger.info(f"Created training report PDF (matplotlib fallback): {output_path}")
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
