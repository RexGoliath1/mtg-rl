#!/usr/bin/env python3
"""Test script for email notification system.

Creates a dummy training report PDF and sends it via EmailNotifier.

Usage:
    # With environment variables:
    FORGERL_NOTIFY_EMAIL=user@example.com uv run python3 scripts/send_test_report.py

    # Override recipient for testing:
    uv run python3 scripts/send_test_report.py --to user@example.com

    # Send existing PDF:
    uv run python3 scripts/send_test_report.py --pdf data/reports/vocab_health_2026-02-07.pdf

    # Custom subject:
    uv run python3 scripts/send_test_report.py --subject "Test Report - $(date)"
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for local testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.email_notifier import EmailNotifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_dummy_pdf() -> str:
    """Create a simple PDF with matplotlib for testing.

    Returns:
        Path to created PDF file
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not installed. Install with: uv sync --extra dev")
        sys.exit(1)

    # Create output directory
    output_dir = Path("data/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"test_report_{timestamp}.pdf"

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    fig.suptitle("ForgeRL Test Training Report", fontsize=16, fontweight="bold")

    # Dummy metrics plot
    epochs = list(range(1, 11))
    loss = [1.5, 1.2, 0.9, 0.7, 0.6, 0.5, 0.45, 0.4, 0.38, 0.35]
    accuracy = [0.6, 0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.89, 0.91, 0.92]

    ax1.plot(epochs, loss, marker="o", label="Loss", color="red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="red")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.grid(True, alpha=0.3)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs, accuracy, marker="s", label="Accuracy", color="blue")
    ax1_twin.set_ylabel("Accuracy", color="blue")
    ax1_twin.tick_params(axis="y", labelcolor="blue")

    ax1.set_title("Training Progress")

    # Summary text
    summary_text = f"""
    Test Training Summary
    ────────────────────────────────────

    Training Configuration:
      • Model: AlphaZero policy/value network
      • Dataset: Synthetic test data
      • Epochs: 10
      • Batch size: 32
      • Learning rate: 0.001

    Final Metrics:
      • Loss: 0.35
      • Accuracy: 92%
      • Training time: 1.5 hours

    Vocabulary Stats:
      • Active enums: 387 / 1387
      • Dead enums: 39
      • Low-fire enums: 4
      • Co-occurrence pairs: 6

    Status: ✓ Training completed successfully

    Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    """

    ax2.axis("off")
    ax2.text(
        0.1, 0.5,
        summary_text,
        fontsize=10,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Created test PDF: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Test email notification system with a dummy training report"
    )
    parser.add_argument(
        "--to",
        help="Recipient email (overrides FORGERL_NOTIFY_EMAIL env var)"
    )
    parser.add_argument(
        "--subject",
        default="ForgeRL Test Report",
        help="Email subject line"
    )
    parser.add_argument(
        "--pdf",
        help="Path to existing PDF to send (if not provided, creates dummy PDF)"
    )
    args = parser.parse_args()

    # Override recipient if specified
    if args.to:
        os.environ["FORGERL_NOTIFY_EMAIL"] = args.to
        logger.info(f"Overriding recipient: {args.to}")

    # Initialize notifier
    try:
        notifier = EmailNotifier()
    except ValueError as e:
        logger.error(str(e))
        logger.info("Set FORGERL_NOTIFY_EMAIL or use --to flag")
        sys.exit(1)

    # Get or create PDF
    if args.pdf:
        pdf_path = args.pdf
        if not Path(pdf_path).exists():
            logger.error(f"PDF not found: {pdf_path}")
            sys.exit(1)
        logger.info(f"Using existing PDF: {pdf_path}")
    else:
        logger.info("Creating dummy test PDF...")
        pdf_path = create_dummy_pdf()

    # Send test email
    body = f"""This is a test email from the ForgeRL training notification system.

Training Metrics:
  final_loss: 0.35
  best_accuracy: 0.92
  total_epochs: 10
  training_time_hours: 1.5

Vocabulary Stats:
  active_enums: 387 / 1387
  dead_enums: 39

Test PDF attached: {Path(pdf_path).name}

If you received this, the email notification system is working correctly!

Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    logger.info("Sending test email...")
    success = notifier.send_report(
        subject=args.subject,
        body_text=body,
        attachments=[pdf_path]
    )

    if success:
        print("\n✓ Test email sent successfully!")
        print(f"  Recipient: {notifier.recipient}")
        print(f"  Subject: {args.subject}")
        print(f"  Attachment: {Path(pdf_path).name}")
    else:
        print("\n✗ Failed to send test email")
        print("  Check logs above for details")
        print("  Verify SMTP credentials in environment variables")
        sys.exit(1)


if __name__ == "__main__":
    main()
