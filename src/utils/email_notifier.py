"""Email notification system for training reports.

Sends emails with optional PDF attachments using SMTP.
All config loaded from environment variables (never hardcoded).
"""

import logging
import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Send training reports via email with optional PDF attachments.

    Configuration via environment variables:
        FORGERL_NOTIFY_EMAIL: Recipient email (REQUIRED)
        FORGERL_SMTP_HOST: SMTP server (default: smtp.gmail.com)
        FORGERL_SMTP_PORT: SMTP port (default: 587)
        FORGERL_SMTP_USER: SMTP username (default: same as notify email)
        FORGERL_SMTP_PASS: SMTP password (REQUIRED if SMTP_USER set)

    Example:
        notifier = EmailNotifier()
        notifier.send_training_complete(
            metrics={"loss": 0.123, "accuracy": 0.95},
            report_path="data/reports/vocab_health_2026-02-07.pdf"
        )
    """

    def __init__(self):
        """Initialize email notifier from environment variables.

        Raises:
            ValueError: If FORGERL_NOTIFY_EMAIL is not set
        """
        self.recipient = os.getenv("FORGERL_NOTIFY_EMAIL")
        if not self.recipient:
            raise ValueError(
                "FORGERL_NOTIFY_EMAIL environment variable is not set. "
                "Cannot send email notifications without recipient address."
            )

        self.smtp_host = os.getenv("FORGERL_SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("FORGERL_SMTP_PORT", "587"))
        self.smtp_user = os.getenv("FORGERL_SMTP_USER", self.recipient)
        self.smtp_pass = os.getenv("FORGERL_SMTP_PASS")

        if not self.smtp_pass and self.smtp_user:
            logger.warning(
                "FORGERL_SMTP_PASS not set. Email sending will likely fail. "
                "Set environment variable or disable email notifications."
            )

        logger.info(
            f"EmailNotifier initialized: {self.smtp_user} -> {self.recipient} "
            f"via {self.smtp_host}:{self.smtp_port}"
        )

    def send_report(
        self,
        subject: str,
        body_text: str,
        attachments: Optional[list[str]] = None,
    ) -> bool:
        """Send an email with optional PDF attachments.

        Args:
            subject: Email subject line
            body_text: Plain text body (brief training summary)
            attachments: List of file paths (PDFs, images) to attach

        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.smtp_pass:
            logger.warning("No SMTP password configured, skipping email send")
            return False

        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.smtp_user
            msg["To"] = self.recipient
            msg["Subject"] = subject

            # Attach body text
            msg.attach(MIMEText(body_text, "plain"))

            # Attach files
            if attachments:
                for filepath in attachments:
                    path = Path(filepath)
                    if not path.exists():
                        logger.warning(f"Attachment not found: {filepath}")
                        continue

                    with open(path, "rb") as f:
                        part = MIMEApplication(f.read(), Name=path.name)
                        part["Content-Disposition"] = f'attachment; filename="{path.name}"'
                        msg.attach(part)
                        logger.info(f"Attached: {path.name}")

            # Send via SMTP
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)

            logger.info(f"Email sent successfully to {self.recipient}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}", exc_info=True)
            return False

    def send_training_complete(
        self,
        metrics: dict,
        report_path: Optional[str] = None,
        timing: Optional[dict] = None,
        timing_detailed: Optional[dict] = None,
        auto_generate_pdf: bool = True,
    ) -> bool:
        """Send a training-completion email with an auto-generated PDF report.

        When *auto_generate_pdf* is True and no *report_path* is given, a
        comprehensive 4-page training report is generated automatically using
        ``scripts/send_test_report.generate_training_report``.  The PDF
        includes a training summary, pipeline timing profile, monitoring
        links, and training curves.

        Args:
            metrics: Training metrics dict.  Recognised keys include
                ``model_name``, ``checkpoint_path``, ``total_time_s``,
                ``epochs``, ``policy_loss``, ``value_loss``, ``accuracy``,
                ``best_checkpoint``, ``games_played``, ``win_rate``, and
                ``history`` (list of per-epoch metric dicts).
            report_path: Explicit path to a PDF to attach.  When provided
                the auto-generation step is skipped.
            timing: ``{stage: total_seconds}`` from ``PipelineTimer.summary()``.
            timing_detailed: ``{stage: {total_s, count, mean_s, pct_of_wall}}``
                from ``PipelineTimer.summary_detailed()``.
            auto_generate_pdf: If True (default) and *report_path* is None,
                automatically generate the training report PDF.

        Returns:
            True if the email was sent successfully, False otherwise.

        Example::

            notifier.send_training_complete(
                metrics={
                    "model_name": "AlphaZero v2",
                    "policy_loss": 0.123,
                    "value_loss": 0.045,
                    "accuracy": 0.95,
                    "epochs": 100,
                    "total_time_s": 7200,
                    "games_played": 500,
                    "win_rate": 68.5,
                },
                timing=pipeline_timer.summary(),
                timing_detailed=pipeline_timer.summary_detailed(),
            )
        """
        # ── Auto-generate PDF if needed ────────────────────────────────
        if report_path is None and auto_generate_pdf:
            try:
                # Import here to avoid circular deps at module level
                import importlib
                mod = importlib.import_module("scripts.send_test_report")
                report_path = mod.generate_training_report(
                    metrics=metrics,
                    timing=timing,
                    timing_detailed=timing_detailed,
                )
                logger.info(f"Auto-generated training report: {report_path}")
            except Exception as exc:
                logger.warning(f"Failed to auto-generate PDF report: {exc}")
                # Continue without attachment

        # ── Build subject line ─────────────────────────────────────────
        model_name = metrics.get("model_name", "model")
        win_rate = metrics.get("win_rate")
        if win_rate is not None:
            subject = f"ForgeRL Training Complete: {model_name} -- {win_rate:.1f}% WR"
        else:
            subject = f"ForgeRL Training Complete: {model_name}"

        # ── Build body text ────────────────────────────────────────────
        body_lines = ["Training run completed successfully.", ""]
        body_lines.append("Metrics:")
        # Show important keys first, then the rest alphabetically
        priority_keys = [
            "model_name", "epochs", "policy_loss", "value_loss",
            "accuracy", "total_time_s", "games_played", "win_rate",
        ]
        shown = set()
        for key in priority_keys:
            if key in metrics:
                shown.add(key)
                value = metrics[key]
                if isinstance(value, float):
                    body_lines.append(f"  {key}: {value:.4f}")
                else:
                    body_lines.append(f"  {key}: {value}")
        for key in sorted(metrics.keys()):
            if key in shown or key == "history":
                continue
            value = metrics[key]
            if isinstance(value, float):
                body_lines.append(f"  {key}: {value:.4f}")
            elif not isinstance(value, (list, dict)):
                body_lines.append(f"  {key}: {value}")

        if report_path:
            body_lines.append("")
            body_lines.append(f"Report attached: {Path(report_path).name}")

        body_text = "\n".join(body_lines)

        # ── Send ───────────────────────────────────────────────────────
        attachments = [report_path] if report_path else None
        return self.send_report(subject, body_text, attachments)
