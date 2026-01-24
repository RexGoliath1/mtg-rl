# =============================================================================
# AWS Cost Controls for MTG RL Project
# =============================================================================
#
# This module sets up:
# 1. AWS Budget with email alerts at 50%, 80%, 100%, and forecast
# 2. CloudWatch billing alarms
# 3. Service-specific cost anomaly detection
#
# Usage:
#   1. Set your email in terraform.tfvars or via command line
#   2. Run: terraform init && terraform apply -var="alert_email=you@email.com"

variable "alert_email" {
  description = "Email address for budget alerts"
  type        = string
}

variable "monthly_budget" {
  description = "Monthly budget limit in USD"
  type        = number
  default     = 50
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

# =============================================================================
# AWS Budget
# =============================================================================

resource "aws_budgets_budget" "monthly" {
  name              = "mtg-rl-${var.environment}-monthly"
  budget_type       = "COST"
  limit_amount      = var.monthly_budget
  limit_unit        = "USD"
  time_unit         = "MONTHLY"
  time_period_start = "2025-01-01_00:00"

  cost_types {
    include_credit             = false
    include_discount           = true
    include_other_subscription = true
    include_recurring          = true
    include_refund             = false
    include_subscription       = true
    include_support            = true
    include_tax                = true
    include_upfront            = true
    use_amortized              = false
    use_blended                = false
  }

  # Alert at 50% of budget
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 50
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
  }

  # Alert at 80% of budget
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
  }

  # Alert at 100% of budget
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
  }

  # Alert if forecast exceeds budget
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.alert_email]
  }
}

# =============================================================================
# CloudWatch Billing Alarm (us-east-1 only for billing metrics)
# =============================================================================

provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
}

resource "aws_cloudwatch_metric_alarm" "billing_alarm" {
  provider            = aws.us_east_1
  alarm_name          = "mtg-rl-billing-alarm-${var.monthly_budget}usd"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = 21600  # 6 hours
  statistic           = "Maximum"
  threshold           = var.monthly_budget
  alarm_description   = "Billing alarm when charges exceed $${var.monthly_budget}"

  dimensions = {
    Currency = "USD"
  }

  alarm_actions = [aws_sns_topic.billing_alerts.arn]
  ok_actions    = [aws_sns_topic.billing_alerts.arn]
}

resource "aws_sns_topic" "billing_alerts" {
  provider = aws.us_east_1
  name     = "mtg-rl-billing-alerts"
}

resource "aws_sns_topic_subscription" "billing_email" {
  provider  = aws.us_east_1
  topic_arn = aws_sns_topic.billing_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# =============================================================================
# Cost Anomaly Detection
# =============================================================================

resource "aws_ce_anomaly_monitor" "service_monitor" {
  name              = "mtg-rl-service-anomaly-monitor"
  monitor_type      = "DIMENSIONAL"
  monitor_dimension = "SERVICE"
}

resource "aws_ce_anomaly_subscription" "alerts" {
  name      = "mtg-rl-anomaly-alerts"
  frequency = "DAILY"

  monitor_arn_list = [
    aws_ce_anomaly_monitor.service_monitor.arn
  ]

  subscriber {
    type    = "EMAIL"
    address = var.alert_email
  }

  threshold_expression {
    dimension {
      key           = "ANOMALY_TOTAL_IMPACT_ABSOLUTE"
      values        = ["10"]  # Alert if anomaly impact > $10
      match_options = ["GREATER_THAN_OR_EQUAL"]
    }
  }
}

# =============================================================================
# Outputs
# =============================================================================

output "budget_name" {
  value       = aws_budgets_budget.monthly.name
  description = "Name of the AWS Budget"
}

output "sns_topic_arn" {
  value       = aws_sns_topic.billing_alerts.arn
  description = "SNS topic ARN for billing alerts"
}

output "cost_controls_summary" {
  value = <<-EOT
    Cost Controls Summary:
    - Monthly Budget: $${var.monthly_budget}
    - Alert Email: ${var.alert_email}
    - Alerts at: 50%, 80%, 100% actual, 100% forecast
    - Billing alarm: $${var.monthly_budget} threshold
    - Anomaly detection: $10+ impact
  EOT
}
