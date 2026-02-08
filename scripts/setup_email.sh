#!/usr/bin/env bash
# Email notification setup wizard for ForgeRL
# Creates .env file with Gmail App Password configuration

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

echo "==============================================="
echo "ForgeRL Email Notification Setup"
echo "==============================================="
echo ""

# Check if .env already exists
if [ -f "$ENV_FILE" ]; then
    echo "WARNING: .env file already exists at:"
    echo "  ${ENV_FILE}"
    echo ""
    read -p "Overwrite existing configuration? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled. Existing .env file unchanged."
        exit 0
    fi
    echo ""
fi

# Prompt for email
echo "Enter your Gmail address (e.g., user@gmail.com):"
read -r EMAIL
if [[ -z "$EMAIL" ]]; then
    echo "ERROR: Email address cannot be empty"
    exit 1
fi

# Display App Password instructions
echo ""
echo "==============================================="
echo "Gmail App Password Setup"
echo "==============================================="
echo ""
echo "You need a Gmail App Password (NOT your regular password)."
echo ""
echo "Steps to get one:"
echo "  1. Open: https://myaccount.google.com/apppasswords"
echo "  2. Sign in and confirm 2FA if prompted"
echo "  3. Under 'App name', type: ForgeRL"
echo "  4. Click 'Create'"
echo "  5. Copy the 16-character password shown"
echo ""
echo "NOTE: The password will look like: abcd efgh ijkl mnop"
echo "      (spaces are OK, we'll remove them)"
echo ""
echo "Paste your App Password here:"
read -rs APP_PASSWORD
echo ""

if [[ -z "$APP_PASSWORD" ]]; then
    echo "ERROR: App Password cannot be empty"
    exit 1
fi

# Strip spaces from password (Gmail shows it with spaces)
APP_PASSWORD_CLEAN=$(echo "$APP_PASSWORD" | tr -d ' ')

# Write .env file
cat > "$ENV_FILE" <<EOF
# ForgeRL Email Notification Configuration
# Generated: $(date)

# Recipient email address
FORGERL_NOTIFY_EMAIL=${EMAIL}

# SMTP credentials for Gmail
FORGERL_SMTP_HOST=smtp.gmail.com
FORGERL_SMTP_PORT=587
FORGERL_SMTP_USER=${EMAIL}
FORGERL_SMTP_PASS=${APP_PASSWORD_CLEAN}
EOF

chmod 600 "$ENV_FILE"  # Secure permissions

echo "✓ Configuration saved to: ${ENV_FILE}"
echo ""

# Test connection
echo "==============================================="
echo "Testing SMTP Connection"
echo "==============================================="
echo ""
echo "Sending test email to ${EMAIL}..."
echo ""

# Source the .env file and run the test
set -a
source "$ENV_FILE"
set +a

cd "$REPO_ROOT"
if uv run python3 scripts/send_test_report.py --to "$EMAIL" --subject "ForgeRL Email Test" 2>&1; then
    echo ""
    echo "==============================================="
    echo "✓ SUCCESS"
    echo "==============================================="
    echo ""
    echo "Email sent successfully!"
    echo "Check your inbox at: ${EMAIL}"
    echo ""
    echo "Configuration complete. Your .env file is ready to use."
    echo ""
else
    echo ""
    echo "==============================================="
    echo "✗ FAILED"
    echo "==============================================="
    echo ""
    echo "Email send failed. Common issues:"
    echo ""
    echo "1. Incorrect App Password"
    echo "   - Make sure you copied the full 16-character password"
    echo "   - Try generating a new App Password and running setup again"
    echo ""
    echo "2. 2FA not enabled on Gmail"
    echo "   - App Passwords require 2-step verification"
    echo "   - Enable at: https://myaccount.google.com/security"
    echo ""
    echo "3. 'Less secure app access' setting"
    echo "   - This is deprecated; use App Passwords instead"
    echo ""
    echo "To retry setup, run:"
    echo "  ./scripts/setup_email.sh"
    echo ""
    exit 1
fi
