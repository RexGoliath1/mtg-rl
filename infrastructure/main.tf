# =============================================================================
# MTG RL Infrastructure - Terraform Configuration
# =============================================================================
#
# This configuration creates:
# - S3 bucket for model checkpoints
# - ECR repository for Docker images
# - IAM roles for training instances
# - (Optional) Spot instance configuration for training
#
# Usage:
#   terraform init
#   terraform plan
#   terraform apply
#
# Scale training by changing variables in terraform.tfvars
# =============================================================================

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# =============================================================================
# Variables
# =============================================================================

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "mtg-rl"
}

# Training scaling variables
variable "enable_training_instance" {
  description = "Whether to create training EC2 instance"
  type        = bool
  default     = false  # Set to true when ready to train
}

variable "training_instance_type" {
  description = "EC2 instance type for training (g4dn.xlarge recommended)"
  type        = string
  default     = "g4dn.xlarge"
}

variable "use_spot_instances" {
  description = "Use spot instances for training (70% cheaper)"
  type        = bool
  default     = true
}

variable "spot_max_price" {
  description = "Maximum spot price (leave empty for on-demand price cap)"
  type        = string
  default     = "0.20"  # ~$0.16 typical for g4dn.xlarge spot
}

# =============================================================================
# Provider Configuration
# =============================================================================

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# Provider for us-east-1 (required for billing metrics)
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# =============================================================================
# Data Sources
# =============================================================================

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Get latest Deep Learning AMI
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Default VPC
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# =============================================================================
# S3 Bucket for Model Checkpoints
# =============================================================================

resource "aws_s3_bucket" "checkpoints" {
  bucket_prefix = "${var.project_name}-checkpoints-"

  # Prevent accidental deletion
  lifecycle {
    prevent_destroy = false  # Set to true in production
  }
}

resource "aws_s3_bucket_versioning" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Lifecycle rule to manage costs
resource "aws_s3_bucket_lifecycle_configuration" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id

  rule {
    id     = "cleanup-old-checkpoints"
    status = "Enabled"

    # Move to Infrequent Access after 30 days
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    # Delete after 90 days (keep recent checkpoints)
    expiration {
      days = 90
    }

    # Only apply to non-best checkpoints
    filter {
      prefix = "checkpoints/"
    }
  }

  rule {
    id     = "keep-best-models"
    status = "Enabled"

    # Best models never expire, just move to cheaper storage
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    filter {
      prefix = "best/"
    }
  }
}

# =============================================================================
# ECR Repository for Docker Images
# =============================================================================

resource "aws_ecr_repository" "training" {
  name                 = "${var.project_name}-training"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Lifecycle policy to limit stored images
resource "aws_ecr_lifecycle_policy" "training" {
  repository = aws_ecr_repository.training.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 5 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 5
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# =============================================================================
# IAM Role for Training Instances
# =============================================================================

resource "aws_iam_role" "training" {
  name = "${var.project_name}-training-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "training_s3" {
  name = "${var.project_name}-s3-access"
  role = aws_iam_role.training.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Resource = [
          aws_s3_bucket.checkpoints.arn,
          "${aws_s3_bucket.checkpoints.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy" "training_ecr" {
  name = "${var.project_name}-ecr-access"
  role = aws_iam_role.training.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "training" {
  name = "${var.project_name}-training-profile"
  role = aws_iam_role.training.name
}

# =============================================================================
# Security Group for Training Instances
# =============================================================================

resource "aws_security_group" "training" {
  name_prefix = "${var.project_name}-training-"
  description = "Security group for MTG RL training instances"
  vpc_id      = data.aws_vpc.default.id

  # SSH access (restrict to your IP in production)
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # TODO: Restrict to your IP
    description = "SSH access"
  }

  # TensorBoard
  ingress {
    from_port   = 6006
    to_port     = 6006
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # TODO: Restrict to your IP
    description = "TensorBoard"
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# =============================================================================
# Training Instance (Optional - enable when ready)
# =============================================================================

resource "aws_spot_instance_request" "training" {
  count = var.enable_training_instance && var.use_spot_instances ? 1 : 0

  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.training_instance_type
  spot_price             = var.spot_max_price
  wait_for_fulfillment   = true
  spot_type              = "one-time"
  iam_instance_profile   = aws_iam_instance_profile.training.name
  vpc_security_group_ids = [aws_security_group.training.id]
  subnet_id              = data.aws_subnets.default.ids[0]

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  user_data = base64encode(<<-EOF
    #!/bin/bash
    set -e

    # Update system
    apt-get update -y

    # Install Docker
    apt-get install -y docker.io
    systemctl start docker
    usermod -aG docker ubuntu

    # Install AWS CLI v2
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    ./aws/install

    # Clone repo
    cd /home/ubuntu
    git clone https://github.com/RexGoliath1/mtg-rl.git
    chown -R ubuntu:ubuntu mtg-rl

    # Log startup complete
    echo "Training instance ready" > /home/ubuntu/startup-complete.txt
  EOF
  )

  tags = {
    Name = "${var.project_name}-training-spot"
  }
}

resource "aws_instance" "training" {
  count = var.enable_training_instance && !var.use_spot_instances ? 1 : 0

  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.training_instance_type
  iam_instance_profile   = aws_iam_instance_profile.training.name
  vpc_security_group_ids = [aws_security_group.training.id]
  subnet_id              = data.aws_subnets.default.ids[0]

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  tags = {
    Name = "${var.project_name}-training-ondemand"
  }
}

# =============================================================================
# Outputs
# =============================================================================

output "s3_bucket_name" {
  description = "S3 bucket for model checkpoints"
  value       = aws_s3_bucket.checkpoints.bucket
}

output "ecr_repository_url" {
  description = "ECR repository URL for training images"
  value       = aws_ecr_repository.training.repository_url
}

output "training_role_arn" {
  description = "IAM role ARN for training instances"
  value       = aws_iam_role.training.arn
}

output "deep_learning_ami" {
  description = "Deep Learning AMI ID"
  value       = data.aws_ami.deep_learning.id
}

output "deployment_summary" {
  description = "Summary of deployed resources"
  value       = <<-EOT
    MTG RL Infrastructure Deployed:
    ================================
    Region: ${var.aws_region}
    Environment: ${var.environment}

    S3 Bucket: ${aws_s3_bucket.checkpoints.bucket}
    ECR Repository: ${aws_ecr_repository.training.repository_url}

    Training Instance: ${var.enable_training_instance ? "ENABLED" : "DISABLED"}
    Instance Type: ${var.training_instance_type}
    Spot Instances: ${var.use_spot_instances ? "YES" : "NO"}

    To enable training instance:
      terraform apply -var="enable_training_instance=true"

    To scale up:
      terraform apply -var="training_instance_type=g4dn.2xlarge"
  EOT
}
