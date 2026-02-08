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
    ) -> bool:
        """Convenience method for training completion notifications.

        Formats a nice summary from metrics dict and attaches report PDF.

        Args:
            metrics: Dictionary of training metrics (loss, accuracy, etc.)
            report_path: Path to vocab health report PDF or other training report

        Returns:
            True if email sent successfully, False otherwise

        Example:
            notifier.send_training_complete(
                metrics={
                    "final_loss": 0.123,
                    "best_accuracy": 0.95,
                    "total_epochs": 100,
                    "training_time_hours": 2.5,
                },
                report_path="data/reports/vocab_health_2026-02-07.pdf"
            )
        """
        # Format summary
        subject = "ForgeRL Training Complete"

        body_lines = ["Training run completed successfully.", ""]
        body_lines.append("Metrics:")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                body_lines.append(f"  {key}: {value:.4f}")
            else:
                body_lines.append(f"  {key}: {value}")

        if report_path:
            body_lines.append("")
            body_lines.append(f"Report attached: {Path(report_path).name}")

        body_text = "\n".join(body_lines)

        # Send email
        attachments = [report_path] if report_path else None
        return self.send_report(subject, body_text, attachments)
