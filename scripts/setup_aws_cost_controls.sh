#!/bin/bash
# =============================================================================
# AWS Cost Controls Setup Script
# =============================================================================
#
# This script sets up cost controls for the MTG RL project:
# - AWS Budget with email alerts
# - CloudWatch billing alarm
#
# Usage:
#   ./scripts/setup_aws_cost_controls.sh your@email.com [monthly_limit]
#
# Example:
#   ./scripts/setup_aws_cost_controls.sh user@example.com 50
#
# Note: You must have AWS CLI configured and appropriate permissions.

set -e

# =============================================================================
# SAFETY: Hard-coded maximum budget - DO NOT MODIFY
# =============================================================================
MAX_BUDGET_LIMIT=100  # $100 absolute maximum - cannot be overridden

# Parse arguments
EMAIL="${1:?Email address required as first argument}"
REQUESTED_LIMIT="${2:-100}"  # Default $100/month
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Enforce hard limit
if [ "$REQUESTED_LIMIT" -gt "$MAX_BUDGET_LIMIT" ]; then
    echo "ERROR: Requested budget \$$REQUESTED_LIMIT exceeds maximum allowed (\$$MAX_BUDGET_LIMIT)"
    echo "This limit is hard-coded for cost protection and cannot be overridden."
    exit 1
fi
MONTHLY_LIMIT="$REQUESTED_LIMIT"

echo "============================================================"
echo "AWS Cost Controls Setup"
echo "============================================================"
echo "Account: $ACCOUNT_ID"
echo "Email: $EMAIL"
echo "Monthly Limit: \$$MONTHLY_LIMIT"
echo "============================================================"

# Create budget JSON
BUDGET_JSON=$(cat <<EOF
{
    "BudgetName": "mtg-rl-monthly-budget",
    "BudgetType": "COST",
    "BudgetLimit": {
        "Amount": "$MONTHLY_LIMIT",
        "Unit": "USD"
    },
    "CostFilters": {},
    "CostTypes": {
        "IncludeTax": true,
        "IncludeSubscription": true,
        "UseBlended": false,
        "IncludeRefund": false,
        "IncludeCredit": false,
        "IncludeUpfront": true,
        "IncludeRecurring": true,
        "IncludeOtherSubscription": true,
        "IncludeSupport": true,
        "IncludeDiscount": true,
        "UseAmortized": false
    },
    "TimeUnit": "MONTHLY"
}
EOF
)

# Create notifications JSON
NOTIFICATIONS_JSON=$(cat <<EOF
[
    {
        "Notification": {
            "NotificationType": "ACTUAL",
            "ComparisonOperator": "GREATER_THAN",
            "Threshold": 50,
            "ThresholdType": "PERCENTAGE"
        },
        "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "$EMAIL"}]
    },
    {
        "Notification": {
            "NotificationType": "ACTUAL",
            "ComparisonOperator": "GREATER_THAN",
            "Threshold": 80,
            "ThresholdType": "PERCENTAGE"
        },
        "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "$EMAIL"}]
    },
    {
        "Notification": {
            "NotificationType": "ACTUAL",
            "ComparisonOperator": "GREATER_THAN",
            "Threshold": 100,
            "ThresholdType": "PERCENTAGE"
        },
        "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "$EMAIL"}]
    },
    {
        "Notification": {
            "NotificationType": "FORECASTED",
            "ComparisonOperator": "GREATER_THAN",
            "Threshold": 100,
            "ThresholdType": "PERCENTAGE"
        },
        "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "$EMAIL"}]
    }
]
EOF
)

# Create budget
echo ""
echo "Creating AWS Budget..."
aws budgets create-budget \
    --account-id "$ACCOUNT_ID" \
    --budget "$BUDGET_JSON" \
    --notifications-with-subscribers "$NOTIFICATIONS_JSON" \
    2>/dev/null || {
        # Budget might already exist, try to update
        echo "Budget may already exist, checking..."
        aws budgets describe-budget \
            --account-id "$ACCOUNT_ID" \
            --budget-name "mtg-rl-monthly-budget" \
            --query 'Budget.BudgetName' \
            --output text 2>/dev/null && echo "Budget already exists" || {
                echo "Failed to create budget. Check permissions."
                exit 1
            }
    }

echo "Budget created/verified: mtg-rl-monthly-budget"

# Create SNS topic for billing alarms (must be in us-east-1)
echo ""
echo "Creating SNS topic for billing alerts (us-east-1)..."
TOPIC_ARN=$(aws sns create-topic \
    --name mtg-rl-billing-alerts \
    --region us-east-1 \
    --query 'TopicArn' \
    --output text 2>/dev/null) || {
        # Topic might already exist
        TOPIC_ARN=$(aws sns list-topics --region us-east-1 --query "Topics[?contains(TopicArn, 'mtg-rl-billing-alerts')].TopicArn" --output text)
    }

echo "SNS Topic: $TOPIC_ARN"

# Subscribe email to topic
echo ""
echo "Subscribing email to billing alerts..."
aws sns subscribe \
    --topic-arn "$TOPIC_ARN" \
    --protocol email \
    --notification-endpoint "$EMAIL" \
    --region us-east-1 \
    2>/dev/null || echo "Email subscription may already exist"

echo ""
echo "============================================================"
echo "IMPORTANT: Check your email ($EMAIL) to confirm the SNS subscription!"
echo "============================================================"

# Create CloudWatch billing alarm
echo ""
echo "Creating CloudWatch billing alarm..."
aws cloudwatch put-metric-alarm \
    --alarm-name "mtg-rl-billing-${MONTHLY_LIMIT}usd" \
    --alarm-description "Billing alarm when charges exceed \$${MONTHLY_LIMIT}" \
    --metric-name EstimatedCharges \
    --namespace AWS/Billing \
    --statistic Maximum \
    --period 21600 \
    --threshold "$MONTHLY_LIMIT" \
    --comparison-operator GreaterThanThreshold \
    --dimensions Name=Currency,Value=USD \
    --evaluation-periods 1 \
    --alarm-actions "$TOPIC_ARN" \
    --region us-east-1 \
    2>/dev/null && echo "Billing alarm created" || echo "Note: Billing alarm requires billing metrics to be enabled"

echo ""
echo "============================================================"
echo "Cost Controls Setup Complete!"
echo "============================================================"
echo ""
echo "Summary:"
echo "  - Budget: \$$MONTHLY_LIMIT/month"
echo "  - Alerts at: 50%, 80%, 100% of budget"
echo "  - Forecast alert if exceeding budget"
echo "  - CloudWatch billing alarm at \$$MONTHLY_LIMIT"
echo ""
echo "Next steps:"
echo "  1. Confirm the SNS subscription email"
echo "  2. Enable billing alerts in AWS Console if not already enabled:"
echo "     Billing Dashboard > Billing Preferences > Receive Billing Alerts"
echo ""
echo "To check current spending:"
echo "  aws ce get-cost-and-usage \\"
echo "    --time-period Start=\$(date -v-30d +%Y-%m-%d),End=\$(date +%Y-%m-%d) \\"
echo "    --granularity MONTHLY \\"
echo "    --metrics UnblendedCost"
echo ""
