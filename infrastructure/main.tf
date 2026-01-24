# MTG RL Infrastructure - Terraform Configuration
# Run: terraform init && terraform apply

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  default = "us-west-2"
}

# ECR Repository
resource "aws_ecr_repository" "mtg_rl" {
  name                 = "mtg-rl-inference"
  image_tag_mutability = "MUTABLE"
}

# S3 Bucket for models
resource "aws_s3_bucket" "models" {
  bucket_prefix = "mtg-rl-models-"
}

# Outputs
output "ecr_repository_url" {
  value = aws_ecr_repository.mtg_rl.repository_url
}

output "s3_bucket" {
  value = aws_s3_bucket.models.bucket
}
