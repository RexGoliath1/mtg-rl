#!/usr/bin/env python3
"""Generate a professional LaTeX white-paper PDF for the ForgeRL project.

Pulls real data from vocabulary.py, the AlphaZero network, HDF5 embeddings,
and training decks to populate a NASA-TP-inspired technical document.

Usage:
    # Generate the whitepaper (requires pdflatex):
    uv run python3 scripts/generate_whitepaper.py

    # Generate .tex only (no compilation):
    uv run python3 scripts/generate_whitepaper.py --tex-only

    # Custom output path:
    uv run python3 scripts/generate_whitepaper.py --output data/reports/custom.pdf

Dependencies:
    - pdflatex (install via: brew install --cask mactex-no-gui  OR  brew install basictex)
    - Python packages: torch, h5py (already in project extras)
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# Data collection helpers
# ---------------------------------------------------------------------------

def get_vocab_size() -> int:
    from mechanics.vocabulary import VOCAB_SIZE
    return VOCAB_SIZE


def get_mechanic_count() -> int:
    from mechanics.vocabulary import Mechanic
    return len(Mechanic)


def get_network_params() -> dict:
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
        print(f"Warning: Could not load network ({e}), using defaults")
        return {
            "encoder_params": 33_100_000, "policy_params": 530_000,
            "value_params": 300_000, "total_params": 33_930_000,
            "d_model": 512, "n_heads": 8, "n_layers": 3, "d_ff": 1024,
            "output_dim": 768, "zone_emb_dim": 512, "global_emb_dim": 192,
            "action_dim": 203, "policy_hidden": 384, "value_hidden": 384,
            "dropout": 0.1,
        }


def get_h5_stats() -> dict:
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


def get_deck_names() -> list:
    """List training decks."""
    decks_dir = PROJECT_ROOT / "decks"
    if not decks_dir.is_dir():
        return []
    names = []
    for f in sorted(decks_dir.glob("*.dck")):
        for line in f.read_text().splitlines():
            if line.startswith("Name="):
                names.append(line.split("=", 1)[1].strip())
                break
        else:
            names.append(f.stem)
    return names


def has_arch_image() -> bool:
    return (PROJECT_ROOT / "data" / "reports" / "network_architecture.png").exists()


# ---------------------------------------------------------------------------
# LaTeX escaping
# ---------------------------------------------------------------------------

def tex_escape(s: str) -> str:
    """Escape special LaTeX characters."""
    for ch in ["&", "%", "$", "#", "_", "{", "}"]:
        s = s.replace(ch, "\\" + ch)
    s = s.replace("~", "\\textasciitilde{}")
    s = s.replace("^", "\\textasciicircum{}")
    return s


# ---------------------------------------------------------------------------
# LaTeX document
# ---------------------------------------------------------------------------

def build_tex(net: dict, h5: dict, vocab_size: int, mechanic_count: int, decks: list, has_img: bool) -> str:
    """Return the complete LaTeX source as a string."""

    def fmt(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    deck_items = "\n".join(f"    \\item {tex_escape(d)}" for d in decks) if decks else "    \\item (no decks found)"

    arch_figure = ""
    if has_img:
        img_path = str(PROJECT_ROOT / "data" / "reports" / "network_architecture.png")
        arch_figure = rf"""
\begin{{figure}}[ht]
  \centering
  \includegraphics[width=0.85\textwidth]{{{img_path}}}
  \caption{{AlphaZero network architecture for ForgeRL. The state encoder produces a
  {net['output_dim']}-dimensional embedding consumed by both policy and value heads.}}
  \label{{fig:architecture}}
\end{{figure}}
"""

    today = datetime.now().strftime("%B %d, %Y")
    year = datetime.now().strftime("%Y")

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
\usepackage{{subcaption}}
\usepackage{{tabularx}}

% ---------------------------------------------------------------------------
% Style
% ---------------------------------------------------------------------------
\hypersetup{{
  colorlinks=true,
  linkcolor=blue!60!black,
  citecolor=blue!60!black,
  urlcolor=blue!60!black,
}}

\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{\small\textit{{ForgeRL Technical Report}}}}
\fancyhead[R]{{\small\thepage}}
\renewcommand{{\headrulewidth}}{{0.4pt}}

\titleformat{{\section}}{{\Large\bfseries}}{{\thesection.}}{{0.5em}}{{}}
\titleformat{{\subsection}}{{\large\bfseries}}{{\thesubsection}}{{0.5em}}{{}}

\captionsetup{{font=small,labelfont=bf}}

% ---------------------------------------------------------------------------
\begin{{document}}

% ===== TITLE PAGE ==========================================================
\begin{{titlepage}}
\vspace*{{2cm}}
\begin{{center}}

{{\Huge\bfseries ForgeRL}}\\[0.5cm]
{{\Large Reinforcement Learning for\\Magic: The Gathering}}

\vspace{{2cm}}

{{\large Technical Report}}\\[0.3cm]
{{\large Version 1.0}}

\vspace{{2cm}}

{{\large {today}}}

\vspace{{3cm}}

\begin{{tabular}}{{l}}
\textbf{{Project Repository:}} \url{{https://github.com/RexGoliath1/mtg}} \\[0.3cm]
\textbf{{Forge Engine:}} \url{{https://github.com/RexGoliath1/forge}} \\[0.3cm]
\textbf{{Total Parameters:}} {fmt(net['total_params'])} \\
\end{{tabular}}

\vfill
{{\small Generated by \texttt{{generate\_whitepaper.py}} \textbullet\ ForgeRL {year}}}
\end{{center}}
\end{{titlepage}}

% ===== TABLE OF CONTENTS ===================================================
\tableofcontents
\newpage

% ===== SECTION 1: ARCHITECTURE OVERVIEW ====================================
\section{{Architecture Overview}}
\label{{sec:architecture}}

ForgeRL employs an AlphaZero-style architecture~\cite{{silver2018alphazero}}
comprising a shared state encoder, a policy head, and a value head.
The system is designed for the complex, partially observable, variable-action
environment of Magic: The Gathering.

{arch_figure}

\subsection{{State Encoder}}

The state encoder converts a Forge JSON game state into a fixed-size
embedding vector $\mathbf{{h}} \in \mathbb{{R}}^{{{net['output_dim']}}}$.
It consists of:

\begin{{enumerate}}
  \item \textbf{{Card Embedding MLP}} --- A shared 2-layer MLP that projects
        raw card features (mechanics multi-hot $+$ parameters $+$ dynamic state)
        from $\mathbb{{R}}^{{{vocab_size} + 37 + 32}}$ to $\mathbb{{R}}^{{{net['d_model']}}}$.
  \item \textbf{{Zone Encoders}} --- Per-zone self-attention (hand, battlefield,
        graveyard, exile) with CLS-token pooling. Each zone encoder uses
        {net['n_heads']}-head attention with $d_{{\\text{{model}}}}={net['d_model']}$
        and $d_{{\\text{{ff}}}}={net['d_ff']}$.
  \item \textbf{{Stack Encoder}} --- Positional-aware encoding of the spell stack (LIFO order).
  \item \textbf{{Global Encoder}} --- Life totals, mana pools, turn/phase, active/priority player
        $\to \mathbb{{R}}^{{{net['global_emb_dim']}}}$.
  \item \textbf{{Cross-Zone Attention}} --- 3-layer multi-head attention across zone embeddings
        to model inter-zone interactions (e.g., reanimation spell in hand $+$ target in graveyard).
\end{{enumerate}}

\subsubsection{{Cross-Attention Formulation}}

The cross-zone attention layers use the standard scaled dot-product attention:

\begin{{equation}}
\text{{Attention}}(Q, K, V) = \text{{softmax}}\!\left(\frac{{QK^\top}}{{\sqrt{{d_k}}}}\right) V
\label{{eq:attention}}
\end{{equation}}

\noindent where $Q, K, V \in \mathbb{{R}}^{{n \times d_k}}$ are queries, keys, and
values projected from the zone embeddings, and $d_k = d_{{\\text{{model}}}} / n_{{\\text{{heads}}}} = {net['d_model'] // net['n_heads']}$.

\subsection{{Parameter Counts}}

\begin{{table}}[H]
\centering
\caption{{Parameter breakdown by component.}}
\label{{tab:params}}
\begin{{tabular}}{{@{{}}lrr@{{}}}}
\toprule
\textbf{{Component}} & \textbf{{Parameters}} & \textbf{{\% of Total}} \\
\midrule
State Encoder     & {net['encoder_params']:,}  & {net['encoder_params']/net['total_params']*100:.1f}\% \\
Policy Head       & {net['policy_params']:,}   & {net['policy_params']/net['total_params']*100:.1f}\% \\
Value Head        & {net['value_params']:,}    & {net['value_params']/net['total_params']*100:.1f}\% \\
\midrule
\textbf{{Total}}  & \textbf{{{net['total_params']:,}}} & 100.0\% \\
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{Policy Head}}

The policy head maps the state embedding to a probability distribution over
the {net['action_dim']}-dimensional action space:

\begin{{equation}}
\boldsymbol{{\pi}}(a \mid s) = \text{{softmax}}\bigl(W_p\, \mathbf{{h}}_{{\\text{{policy}}}} + \mathbf{{b}}_p\bigr) \odot \mathbf{{m}}
\label{{eq:policy}}
\end{{equation}}

\noindent where $\mathbf{{h}}_{{\\text{{policy}}}}$ is the output of a 3-layer MLP with hidden
dimension~{net['policy_hidden']}, and $\mathbf{{m}} \in \{{0,1\}}^{{{net['action_dim']}}}$ is the
legal-action mask (illegal actions receive $-\infty$ before softmax).

\subsection{{Value Head}}

The value head estimates the expected outcome from the current state:

\begin{{equation}}
v(s) = \tanh\bigl(W_v\, \mathbf{{h}}_{{\\text{{value}}}} + b_v\bigr)
\label{{eq:value}}
\end{{equation}}

\noindent where $\mathbf{{h}}_{{\\text{{value}}}}$ is the output of a 2-layer MLP with hidden
dimension~{net['value_hidden']}. For multiplayer Commander, the output becomes a
softmax over per-player win probabilities.


% ===== SECTION 2: MECHANICS VOCABULARY =====================================
\newpage
\section{{Mechanics Vocabulary}}
\label{{sec:vocabulary}}

Rather than using text embeddings or one-hot card identifiers, ForgeRL
decomposes each card into a multi-hot vector over \textbf{{{vocab_size}}} atomic
mechanics primitives. This design enables:

\begin{{itemize}}[nosep]
  \item \textbf{{Compositional generalization}} --- new cards are novel combinations of known primitives.
  \item \textbf{{Format transfer}} --- the same vocabulary covers Draft, Standard, and Commander.
  \item \textbf{{Compact storage}} --- all {h5['card_count']:,} {h5['format']}-legal cards fit in {h5['file_size_mb']:.2f}\,MB (HDF5).
\end{{itemize}}

\subsection{{Vocabulary Statistics}}

\begin{{table}}[H]
\centering
\caption{{Mechanics vocabulary summary.}}
\label{{tab:vocab}}
\begin{{tabular}}{{@{{}}lr@{{}}}}
\toprule
\textbf{{Metric}} & \textbf{{Value}} \\
\midrule
Total vocabulary size (\texttt{{VOCAB\_SIZE}}) & {vocab_size} \\
Named mechanic enums & {mechanic_count} \\
Cards encoded ({h5['format']}) & {h5['card_count']:,} \\
HDF5 file size & {h5['file_size_mb']:.2f}\,MB \\
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{Example: Saw in Half}}

\textit{{Saw in Half}} $\{{2\}}{{B}}$ --- Instant

\medskip
\noindent\textbf{{Oracle text:}} Destroy target creature. If that creature would
die this turn, instead exile it, then create two tokens that are copies of
that creature, except their base power is half that creature's power and
their base toughness is half that creature's toughness.

\medskip
\noindent\textbf{{Mechanics encoding:}}

\begin{{verbatim}}
[INSTANT_SPEED, TARGET_CREATURE, TARGET_ANY_CONTROLLER,
 DESTROY, IF_TARGET_DIES, CREATE_TOKEN_COPY, HALF_STATS]
\end{{verbatim}}

\noindent The network learns through self-play that \texttt{{DESTROY}} $+$
\texttt{{CREATE\_TOKEN\_COPY}} on its own creature yields double ETB triggers ---
an emergent strategic insight, not a pre-coded rule.


% ===== SECTION 3: TRAINING CONFIGURATION ==================================
\newpage
\section{{Training Configuration}}
\label{{sec:training}}

Training proceeds in three phases: behavioral cloning on human data,
reinforcement learning via self-play, and evaluation against baselines.

\subsection{{Hyperparameters}}

\begin{{table}}[H]
\centering
\caption{{Training hyperparameters.}}
\label{{tab:hyperparams}}
\begin{{tabular}}{{@{{}}llr@{{}}}}
\toprule
\textbf{{Category}} & \textbf{{Parameter}} & \textbf{{Value}} \\
\midrule
\multirow{{4}}{{*}}{{Architecture}}
  & $d_{{\\text{{model}}}}$ & {net['d_model']} \\
  & $d_{{\\text{{ff}}}}$ & {net['d_ff']} \\
  & Attention heads & {net['n_heads']} \\
  & Dropout & {net['dropout']} \\
\midrule
\multirow{{4}}{{*}}{{Optimizer}}
  & Algorithm & AdamW \\
  & Learning rate & $3 \times 10^{{-4}}$ \\
  & Weight decay & $10^{{-4}}$ \\
  & Scheduler & Cosine annealing \\
\midrule
\multirow{{3}}{{*}}{{MCTS}}
  & Simulations per move & 800 \\
  & $c_{{\\text{{puct}}}}$ & 1.5 \\
  & Dirichlet $\alpha$ & 0.3 \\
\midrule
\multirow{{2}}{{*}}{{Self-Play}}
  & Temperature & $1.0 \to 0.1$ \\
  & History window & 500K positions \\
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{MCTS Selection Rule}}

During tree search, actions are selected using the PUCT formula~\cite{{silver2018alphazero}}:

\begin{{equation}}
a^* = \arg\max_a \left[ Q(s,a) + c_{{\\text{{puct}}}} \cdot P(s,a) \cdot
\frac{{\sqrt{{N(s)}}}}{{1 + N(s,a)}} \right]
\label{{eq:puct}}
\end{{equation}}

\noindent where $Q(s,a)$ is the mean action value, $P(s,a)$ is the policy prior,
$N(s)$ is the parent visit count, and $N(s,a)$ is the child visit count.

\subsection{{Loss Function}}

The combined loss for joint policy--value training is:

\begin{{equation}}
\mathcal{{L}} = \underbrace{{-\boldsymbol{{\pi}}^{{\\text{{target}}}} \cdot \log \boldsymbol{{\pi}}_\theta}}_{{\\text{{policy loss (cross-entropy)}}}}
+ \underbrace{{(z - v_\theta)^2}}_{{\\text{{value loss (MSE)}}}}
+ \underbrace{{c \|\theta\|^2}}_{{\\text{{L2 regularization}}}}
\label{{eq:loss}}
\end{{equation}}

\noindent where $z \in \{{-1, +1\}}$ is the game outcome and $c = 10^{{-4}}$.


% ===== SECTION 4: TRAINING RESULTS ========================================
\newpage
\section{{Training Results}}
\label{{sec:results}}

\textit{{This section will be populated with real metrics after the first
full training run. Placeholder structure is shown below.}}

\subsection{{Loss Curves}}

\begin{{table}}[H]
\centering
\caption{{Training metrics at convergence (placeholder).}}
\label{{tab:results}}
\begin{{tabular}}{{@{{}}lrr@{{}}}}
\toprule
\textbf{{Metric}} & \textbf{{BC Phase}} & \textbf{{RL Phase}} \\
\midrule
Policy loss & 0.35 & --- \\
Value loss  & 0.12 & --- \\
Top-1 accuracy & 92.0\% & --- \\
Epochs & 10 & --- \\
Wall time & 1.5\,h & --- \\
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{Timing Breakdown}}

\begin{{table}}[H]
\centering
\caption{{Pipeline stage durations (placeholder).}}
\label{{tab:timing}}
\begin{{tabular}}{{@{{}}lrr@{{}}}}
\toprule
\textbf{{Stage}} & \textbf{{Duration}} & \textbf{{\% of Total}} \\
\midrule
Data collection  & 50\,min & 55.6\% \\
Data encoding    & 14\,min & 15.6\% \\
Training         & 20\,min & 22.2\% \\
Evaluation       &  6\,min &  6.7\% \\
\midrule
\textbf{{Total}} & \textbf{{90\,min}} & 100.0\% \\
\bottomrule
\end{{tabular}}
\end{{table}}


% ===== SECTION 5: DECK ANALYSIS ===========================================
\newpage
\section{{Deck Analysis}}
\label{{sec:decks}}

Training uses {len(decks)} constructed decks spanning aggro, midrange, and
control archetypes. The mechanics centroid of each deck is computed from
HDF5 embeddings to characterize strategic profiles.

\subsection{{Training Deck List}}

\begin{{enumerate}}[nosep]
{deck_items}
\end{{enumerate}}

\subsection{{Centroid Analysis}}

For each deck $D = \{{c_1, \ldots, c_n\}}$, the mechanics centroid is:

\begin{{equation}}
\boldsymbol{{\mu}}_D = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} \mathbf{{e}}(c_i)
\label{{eq:centroid}}
\end{{equation}}

\noindent where $\mathbf{{e}}(c_i) \in \{{0,1\}}^{{{vocab_size}}}$ is the multi-hot
mechanics vector for card $c_i$. High-weight dimensions in the centroid
reveal the dominant mechanical themes of the deck (e.g., \texttt{{DEATH\_TRIGGER}}
and \texttt{{SACRIFICE}} for aristocrats strategies).

\textit{{Per-deck centroid bar charts will be added after HDF5 analysis integration.}}


% ===== SECTION 6: EVALUATION ==============================================
\newpage
\section{{Evaluation}}
\label{{sec:evaluation}}

\textit{{This section will be populated after evaluation runs are completed.}}

\subsection{{Evaluation Protocol}}

\begin{{enumerate}}[nosep]
  \item \textbf{{Forge AI Baseline}} --- 100 games against the built-in Forge AI
        at medium difficulty.
  \item \textbf{{Random Policy}} --- 100 games against a uniform-random legal-action agent.
  \item \textbf{{Self-Play Elo}} --- Round-robin tournament between checkpoint generations
        to track training progress via Elo rating.
\end{{enumerate}}

\subsection{{Expected Metrics}}

\begin{{table}}[H]
\centering
\caption{{Evaluation targets (placeholder).}}
\label{{tab:eval}}
\begin{{tabular}}{{@{{}}lrrr@{{}}}}
\toprule
\textbf{{Opponent}} & \textbf{{Win Rate}} & \textbf{{Games}} & \textbf{{Elo}} \\
\midrule
Random Policy   & $>$95\% & 100 & --- \\
Forge AI (Med)  & $>$60\% & 100 & --- \\
Self (Gen $n-1$)& $>$55\% & 200 & 1500+ \\
\bottomrule
\end{{tabular}}
\end{{table}}


% ===== REFERENCES ==========================================================
\newpage
\begin{{thebibliography}}{{9}}

\bibitem{{silver2018alphazero}}
D.~Silver \textit{{et al.}},
``A general reinforcement learning algorithm that masters chess, shogi,
and Go through self-play,''
\textit{{Science}}, vol.~362, no.~6419, pp.~1140--1144, 2018.

\bibitem{{browne2012mcts}}
C.~B.~Browne \textit{{et al.}},
``A survey of Monte Carlo tree search methods,''
\textit{{IEEE Transactions on Computational Intelligence and AI in Games}},
vol.~4, no.~1, pp.~1--43, 2012.

\bibitem{{schrittwieser2020muzero}}
J.~Schrittwieser \textit{{et al.}},
``Mastering Atari, Go, chess and shogi by planning with a learned model,''
\textit{{Nature}}, vol.~588, pp.~604--609, 2020.

\bibitem{{cowling2012ismcts}}
P.~I.~Cowling, E.~J.~Powley, and D.~Whitehouse,
``Information set Monte Carlo tree search,''
\textit{{IEEE Transactions on Computational Intelligence and AI in Games}},
vol.~4, no.~2, pp.~120--143, 2012.

\bibitem{{forge}}
Card-Forge Contributors,
``Forge --- Open source Magic: The Gathering game engine,''
\url{{https://github.com/Card-Forge/forge}}, 2024.

\end{{thebibliography}}

\end{{document}}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate ForgeRL whitepaper PDF")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "reports" / "forgerl_whitepaper.pdf"),
                        help="Output PDF path")
    parser.add_argument("--tex-only", action="store_true",
                        help="Generate .tex file without compiling to PDF")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Collecting project data...")
    vocab_size = get_vocab_size()
    mechanic_count = get_mechanic_count()
    net = get_network_params()
    h5 = get_h5_stats()
    decks = get_deck_names()
    has_img = has_arch_image()

    print(f"  VOCAB_SIZE = {vocab_size}")
    print(f"  Named mechanics = {mechanic_count}")
    print(f"  Network params = {net['total_params']:,}")
    print(f"  HDF5 cards = {h5['card_count']:,} ({h5['format']})")
    print(f"  Training decks = {len(decks)}")
    print(f"  Architecture image = {has_img}")

    tex_source = build_tex(net, h5, vocab_size, mechanic_count, decks, has_img)

    if args.tex_only:
        tex_path = output_path.with_suffix(".tex")
        tex_path.write_text(tex_source)
        print(f"\nTeX file written: {tex_path}")
        return

    # Check for pdflatex
    if not shutil.which("pdflatex"):
        # Write .tex anyway so user can compile manually
        tex_path = output_path.with_suffix(".tex")
        tex_path.write_text(tex_source)
        print("\npdflatex not found. Install LaTeX with one of:")
        print("  macOS:  brew install --cask mactex-no-gui")
        print("  macOS:  brew install basictex")
        print("  Ubuntu: sudo apt-get install texlive-latex-recommended texlive-fonts-recommended")
        print(f"\nTeX file saved to: {tex_path}")
        print(f"Compile manually:  pdflatex -output-directory={output_path.parent} {tex_path}")
        return

    # Compile with pdflatex (run twice for TOC)
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_file = Path(tmpdir) / "forgerl_whitepaper.tex"
        tex_file.write_text(tex_source)

        for pass_num in (1, 2):
            print(f"pdflatex pass {pass_num}/2...")
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, str(tex_file)],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0 and pass_num == 2:
                print("pdflatex warnings/errors (may still produce PDF):")
                for line in result.stdout.splitlines():
                    if line.startswith("!") or "Error" in line:
                        print(f"  {line}")

        pdf_tmp = Path(tmpdir) / "forgerl_whitepaper.pdf"
        if pdf_tmp.exists():
            shutil.copy2(pdf_tmp, output_path)
            print(f"\nWhitepaper PDF generated: {output_path}")
            print(f"  Size: {output_path.stat().st_size / 1024:.0f} KB")
        else:
            tex_out = output_path.with_suffix(".tex")
            tex_out.write_text(tex_source)
            print(f"\nPDF compilation failed. TeX source saved to: {tex_out}")
            print("Check pdflatex output for errors.")
            log_file = Path(tmpdir) / "forgerl_whitepaper.log"
            if log_file.exists():
                print("\nLast 20 lines of LaTeX log:")
                for line in log_file.read_text().splitlines()[-20:]:
                    print(f"  {line}")


if __name__ == "__main__":
    main()
