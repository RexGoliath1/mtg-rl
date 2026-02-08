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
  default     = "us-east-1"
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

# Training configuration
variable "training_sets" {
  description = "MTG sets to train on"
  type        = list(string)
  default     = ["FDN", "DSK", "BLB", "TLA"]
}

variable "training_epochs" {
  description = "Number of training epochs"
  type        = number
  default     = 50
}

variable "training_batch_size" {
  description = "Training batch size"
  type        = number
  default     = 256
}

variable "training_max_samples" {
  description = "Max samples per set (0 = all)"
  type        = number
  default     = 0
}

variable "encoder_type" {
  description = "Card encoder type: 'keyword' (v1) or 'hybrid' (v2 with text embeddings)"
  type        = string
  default     = "hybrid"
}

variable "training_mode" {
  description = "Training mode: 'bc' for behavioral cloning, 'imitation' for data collection, 'imitation_train' for model training"
  type        = string
  default     = "bc"
}

variable "imitation_games" {
  description = "Number of games for imitation learning data collection"
  type        = number
  default     = 10000
}

variable "imitation_workers" {
  description = "Parallel workers for imitation learning (max 10)"
  type        = number
  default     = 8
}

variable "imitation_train_epochs" {
  description = "Epochs for imitation model training"
  type        = number
  default     = 50
}

variable "imitation_train_hidden_dim" {
  description = "Hidden dimension for imitation policy network"
  type        = number
  default     = 256
}

variable "auto_shutdown" {
  description = "Auto-shutdown instance after training completes (saves cost)"
  type        = bool
  default     = true
}

variable "ssh_key_name" {
  description = "SSH key pair name for EC2 access"
  type        = string
  default     = ""
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
    # Suspended: reproducible training artifacts don't need versioning.
    # Previously Enabled, which silently accumulated hidden versions.
    # Note: "Suspended" stops creating new versions but preserves existing ones.
    status = "Suspended"
  }
}

# Lifecycle rules to manage S3 costs
resource "aws_s3_bucket_lifecycle_configuration" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id

  # ---- Experiment checkpoints: IA after 30d, delete after 90d ----
  # Applies to experiment checkpoint files (epoch_*.pt, latest.pt, etc.)
  # but NOT to best.pt which is promoted to models/ prefix.
  rule {
    id     = "cleanup-experiment-checkpoints"
    status = "Enabled"

    filter {
      prefix = "experiments/"
    }

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    expiration {
      days = 90
    }
  }

  # ---- Legacy checkpoints/ prefix: same policy ----
  # Existing data under checkpoints/ from prior runs.
  # Excludes best.pt via separate promoted-models rule.
  rule {
    id     = "cleanup-legacy-checkpoints"
    status = "Enabled"

    filter {
      prefix = "checkpoints/"
    }

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    expiration {
      days = 90
    }
  }

  # ---- Promoted best models: never expire, cheaper storage ----
  # Best models are copied to models/ when promoted.
  # Store as IA after 30d since they're rarely re-downloaded.
  rule {
    id     = "keep-promoted-models"
    status = "Enabled"

    filter {
      prefix = "models/"
    }

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    # No expiration - best models are kept indefinitely
  }

  # ---- Imitation learning data: One Zone-IA after 30d, Glacier IR after 90d ----
  # Imitation data is reproducible (re-collect from Forge), so One Zone-IA
  # is safe and ~20% cheaper than standard IA.
  rule {
    id     = "imitation-data-tiering"
    status = "Enabled"

    filter {
      prefix = "imitation_data/"
    }

    transition {
      days          = 30
      storage_class = "ONEZONE_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER_IR"
    }
  }

  # ---- TensorBoard logs: delete after 60d ----
  rule {
    id     = "cleanup-tensorboard-logs"
    status = "Enabled"

    filter {
      prefix = "tensorboard-logs/"
    }

    expiration {
      days = 60
    }
  }

  # ---- Abort incomplete multipart uploads after 7d ----
  # Prevents accumulation of orphaned partial uploads.
  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"

    filter {
      prefix = ""
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  # ---- Clean up old noncurrent versions (from when versioning was Enabled) ----
  # Versioning is now Suspended, but existing noncurrent versions remain.
  # This rule deletes them after 7 days to reclaim storage.
  rule {
    id     = "expire-noncurrent-versions"
    status = "Enabled"

    filter {
      prefix = ""
    }

    noncurrent_version_expiration {
      noncurrent_days = 7
    }
  }
}

# =============================================================================
# ECR Repositories for Docker Images
# =============================================================================

resource "aws_ecr_repository" "training" {
  name                 = "${var.project_name}-training"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "daemon" {
  name                 = "${var.project_name}-daemon"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "collection" {
  name                 = "${var.project_name}-collection"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Lifecycle policies to limit stored images
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

resource "aws_ecr_lifecycle_policy" "daemon" {
  repository = aws_ecr_repository.daemon.name

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

resource "aws_ecr_lifecycle_policy" "collection" {
  repository = aws_ecr_repository.collection.name

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

# SSM policy for keyless SSH via Session Manager
resource "aws_iam_role_policy_attachment" "training_ssm" {
  role       = aws_iam_role.training.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "training_secrets" {
  name = "${var.project_name}-training-secrets"
  role = aws_iam_role.training.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = "arn:aws:secretsmanager:${var.aws_region}:*:secret:mtg-rl/*"
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
  key_name               = var.ssh_key_name != "" ? var.ssh_key_name : null

  # Force new instance when userdata changes
  user_data_replace_on_change = true

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  user_data = base64encode(
    var.training_mode == "imitation" ? templatefile("${path.module}/imitation_userdata.sh.tpl", {
      s3_bucket     = aws_s3_bucket.checkpoints.bucket
      ecr_repo      = aws_ecr_repository.daemon.repository_url
      num_games     = var.imitation_games
      workers       = var.imitation_workers
      auto_shutdown = var.auto_shutdown ? "true" : "false"
    }) : var.training_mode == "imitation_train" ? templatefile("${path.module}/imitation_train_userdata.sh.tpl", {
      s3_bucket     = aws_s3_bucket.checkpoints.bucket
      epochs        = var.imitation_train_epochs
      batch_size    = var.training_batch_size
      hidden_dim    = var.imitation_train_hidden_dim
      auto_shutdown = var.auto_shutdown ? "true" : "false"
    }) : templatefile("${path.module}/training_userdata.sh.tpl", {
      s3_bucket     = aws_s3_bucket.checkpoints.bucket
      sets          = join(" ", var.training_sets)
      epochs        = var.training_epochs
      batch_size    = var.training_batch_size
      max_samples   = var.training_max_samples > 0 ? var.training_max_samples : ""
      encoder_type  = var.encoder_type
      auto_shutdown = var.auto_shutdown ? "true" : "false"
    })
  )

  tags = {
    Name = "${var.project_name}-${var.training_mode}-spot"
  }
}

resource "aws_instance" "training" {
  count = var.enable_training_instance && !var.use_spot_instances ? 1 : 0

  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.training_instance_type
  iam_instance_profile   = aws_iam_instance_profile.training.name
  vpc_security_group_ids = [aws_security_group.training.id]
  subnet_id              = data.aws_subnets.default.ids[0]
  key_name               = var.ssh_key_name != "" ? var.ssh_key_name : null

  # Force new instance when userdata changes
  user_data_replace_on_change = true

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  user_data = base64encode(
    var.training_mode == "imitation" ? templatefile("${path.module}/imitation_userdata.sh.tpl", {
      s3_bucket     = aws_s3_bucket.checkpoints.bucket
      ecr_repo      = aws_ecr_repository.daemon.repository_url
      num_games     = var.imitation_games
      workers       = var.imitation_workers
      auto_shutdown = var.auto_shutdown ? "true" : "false"
    }) : var.training_mode == "imitation_train" ? templatefile("${path.module}/imitation_train_userdata.sh.tpl", {
      s3_bucket     = aws_s3_bucket.checkpoints.bucket
      epochs        = var.imitation_train_epochs
      batch_size    = var.training_batch_size
      hidden_dim    = var.imitation_train_hidden_dim
      auto_shutdown = var.auto_shutdown ? "true" : "false"
    }) : templatefile("${path.module}/training_userdata.sh.tpl", {
      s3_bucket     = aws_s3_bucket.checkpoints.bucket
      sets          = join(" ", var.training_sets)
      epochs        = var.training_epochs
      batch_size    = var.training_batch_size
      max_samples   = var.training_max_samples > 0 ? var.training_max_samples : ""
      encoder_type  = var.encoder_type
      auto_shutdown = var.auto_shutdown ? "true" : "false"
    })
  )

  tags = {
    Name = "${var.project_name}-${var.training_mode}-ondemand"
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

output "ecr_daemon_repository_url" {
  description = "ECR repository URL for Forge daemon images"
  value       = aws_ecr_repository.daemon.repository_url
}

output "ecr_collection_repository_url" {
  description = "ECR repository URL for collection images"
  value       = aws_ecr_repository.collection.repository_url
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
    ECR Repositories:
      Training:   ${aws_ecr_repository.training.repository_url}
      Daemon:     ${aws_ecr_repository.daemon.repository_url}
      Collection: ${aws_ecr_repository.collection.repository_url}

    Training Instance: ${var.enable_training_instance ? "ENABLED" : "DISABLED"}
    Instance Type: ${var.training_instance_type}
    Spot Instances: ${var.use_spot_instances ? "YES" : "NO"}

    Training Config:
      Sets: ${join(", ", var.training_sets)}
      Epochs: ${var.training_epochs}
      Batch Size: ${var.training_batch_size}

    To enable training instance:
      terraform apply -var="enable_training_instance=true"

    To scale up:
      terraform apply -var="training_instance_type=g4dn.2xlarge"
  EOT
}

output "training_instance_ip" {
  description = "Public IP of training instance (if enabled)"
  value = var.enable_training_instance ? (
    var.use_spot_instances ?
      try(aws_spot_instance_request.training[0].public_ip, "pending") :
      try(aws_instance.training[0].public_ip, "pending")
  ) : "not enabled"
}

output "training_monitor_commands" {
  description = "Commands to monitor training"
  value = var.enable_training_instance ? join("\n", [
    "# SSH to training instance (replace INSTANCE_IP):",
    "ssh -i ~/.ssh/your-key.pem ubuntu@INSTANCE_IP",
    "",
    "# Forward TensorBoard:",
    "ssh -L 6006:localhost:6006 -i ~/.ssh/your-key.pem ubuntu@INSTANCE_IP",
    "# Then open http://localhost:6006",
    "",
    "# Check training progress:",
    "ssh ubuntu@INSTANCE_IP 'tail -20 /home/ubuntu/mtg-rl/training.log'",
    "",
    "# Download latest checkpoint:",
    "aws s3 cp s3://${aws_s3_bucket.checkpoints.bucket}/checkpoints/latest.pt checkpoints/"
  ]) : "Training not enabled"
}
