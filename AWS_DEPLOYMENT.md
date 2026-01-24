# AWS Deployment Guide for MTG RL

## Quick Recommendation

**For this project, I recommend: AWS ECS Fargate + API Gateway + WAF**

- **Compute**: ECS Fargate (serverless containers) - ~$0.04/vCPU-hour
- **GPU Training**: EC2 p3.2xlarge or g5.xlarge for training runs
- **Rate Limiting**: API Gateway + WAF - built-in throttling
- **Estimated Cost**: $50-100/month for inference, $20-50 for training runs

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AWS Architecture                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │  Client  │───►│ API Gateway  │───►│   AWS WAF    │───►│   ALB    │  │
│  └──────────┘    │ (throttling) │    │ (DDoS/rules) │    └────┬─────┘  │
│                  └──────────────┘    └──────────────┘         │        │
│                                                               ▼        │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                     ECS Fargate Cluster                           │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │ │
│  │  │ Inference Svc   │  │ Inference Svc   │  │ Inference Svc   │   │ │
│  │  │ (draft-api)     │  │ (draft-api)     │  │ (draft-api)     │   │ │
│  │  │ 2 vCPU, 4GB     │  │ 2 vCPU, 4GB     │  │ 2 vCPU, 4GB     │   │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘   │ │
│  │           │                    │                    │             │ │
│  │           └────────────────────┴────────────────────┘             │ │
│  │                              │                                    │ │
│  └──────────────────────────────┼────────────────────────────────────┘ │
│                                 ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                        S3 Bucket                                  │  │
│  │  ├── models/                     (trained model weights)         │  │
│  │  ├── data/17lands/               (training data)                 │  │
│  │  └── checkpoints/                (training checkpoints)          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              EC2 GPU Instance (for training)                      │  │
│  │  Instance: p3.2xlarge (V100) or g5.xlarge (A10G)                 │  │
│  │  Spot pricing: ~$1-2/hour                                         │  │
│  │  Launched on-demand for training runs                            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Rate Limiting Strategy

### Layer 1: API Gateway Throttling (Primary)

```yaml
# API Gateway throttling settings
ThrottlingBurstLimit: 100    # Max concurrent requests
ThrottlingRateLimit: 50      # Requests per second

# Per-client throttling (with API keys)
UsagePlan:
  Throttle:
    BurstLimit: 10           # Per-client burst
    RateLimit: 5             # Per-client rate
  Quota:
    Limit: 10000             # Daily limit
    Period: DAY
```

### Layer 2: AWS WAF Rules (DDoS Protection)

```yaml
WAFRules:
  # Rate-based rule
  - Name: RateLimitRule
    Priority: 1
    Statement:
      RateBasedStatement:
        Limit: 2000          # Requests per 5-minute window
        AggregateKeyType: IP
    Action: Block

  # Geographic restrictions (optional)
  - Name: GeoBlock
    Priority: 2
    Statement:
      NotStatement:
        GeoMatchStatement:
          CountryCodes: [US, CA, EU]
    Action: Block

  # Bot detection
  - Name: BotControl
    Priority: 3
    Statement:
      ManagedRuleGroupStatement:
        VendorName: AWS
        Name: AWSManagedRulesBotControlRuleSet
```

### Layer 3: Application-Level Rate Limiting

```python
# In your FastAPI application
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/draft/pick")
@limiter.limit("10/minute")  # 10 requests per minute per IP
async def get_pick_recommendation(request: PickRequest):
    ...

@app.post("/api/v1/draft/start")
@limiter.limit("5/minute")   # 5 new drafts per minute per IP
async def start_draft(request: DraftRequest):
    ...
```

---

## Infrastructure as Code (Terraform)

### `main.tf`

```hcl
# Provider configuration
provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  default = "us-west-2"
}

variable "environment" {
  default = "production"
}

# VPC
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "mtg-rl-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["us-west-2a", "us-west-2b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true
}

# ECR Repository
resource "aws_ecr_repository" "mtg_rl" {
  name                 = "mtg-rl-inference"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "mtg-rl-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "inference" {
  family                   = "mtg-rl-inference"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 2048   # 2 vCPU
  memory                   = 4096   # 4 GB

  container_definitions = jsonencode([
    {
      name  = "inference"
      image = "${aws_ecr_repository.mtg_rl.repository_url}:latest"

      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "MODEL_PATH"
          value = "s3://mtg-rl-models/production/model.pt"
        },
        {
          name  = "MAX_BATCH_SIZE"
          value = "32"
        },
        {
          name  = "RATE_LIMIT_PER_SECOND"
          value = "100"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/mtg-rl-inference"
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}

# ECS Service with Auto Scaling
resource "aws_ecs_service" "inference" {
  name            = "mtg-rl-inference"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.inference.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = module.vpc.private_subnets
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.inference.arn
    container_name   = "inference"
    container_port   = 8000
  }
}

# Auto Scaling
resource "aws_appautoscaling_target" "ecs" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.inference.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "cpu" {
  name               = "cpu-autoscaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70.0
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "mtg-rl-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = module.vpc.public_subnets
}

resource "aws_lb_target_group" "inference" {
  name        = "mtg-rl-inference"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = module.vpc.vpc_id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 3
  }
}

# API Gateway
resource "aws_apigatewayv2_api" "main" {
  name          = "mtg-rl-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_stage" "prod" {
  api_id      = aws_apigatewayv2_api.main.id
  name        = "prod"
  auto_deploy = true

  default_route_settings {
    throttling_burst_limit = 100
    throttling_rate_limit  = 50
  }
}

# WAF Web ACL
resource "aws_wafv2_web_acl" "main" {
  name        = "mtg-rl-waf"
  scope       = "REGIONAL"
  description = "WAF for MTG RL API"

  default_action {
    allow {}
  }

  # Rate limiting rule
  rule {
    name     = "RateLimitRule"
    priority = 1

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "RateLimitRule"
      sampled_requests_enabled   = true
    }
  }

  # AWS Managed Rules
  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 2

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        vendor_name = "AWS"
        name        = "AWSManagedRulesCommonRuleSet"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "AWSManagedRulesCommonRuleSet"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "mtg-rl-waf"
    sampled_requests_enabled   = true
  }
}

# S3 Bucket for models
resource "aws_s3_bucket" "models" {
  bucket = "mtg-rl-models-${var.aws_region}"
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Security Groups
resource "aws_security_group" "alb" {
  name   = "mtg-rl-alb-sg"
  vpc_id = module.vpc.vpc_id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "ecs" {
  name   = "mtg-rl-ecs-sg"
  vpc_id = module.vpc.vpc_id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Outputs
output "api_endpoint" {
  value = aws_apigatewayv2_stage.prod.invoke_url
}

output "alb_dns" {
  value = aws_lb.main.dns_name
}

output "ecr_repository_url" {
  value = aws_ecr_repository.mtg_rl.repository_url
}
```

---

## Docker Configuration

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Download model from S3 on startup
ENV MODEL_PATH=/app/models/model.pt
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Run
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `requirements.txt`

```
torch>=2.0.0
numpy>=1.24.0
fastapi>=0.100.0
uvicorn>=0.23.0
slowapi>=0.1.8
boto3>=1.28.0
pydantic>=2.0.0
```

---

## Inference API

### `api/main.py`

```python
"""
MTG RL Inference API

FastAPI application for serving draft recommendations.
Includes rate limiting and request validation.
"""

import os
import time
import boto3
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from shared_card_encoder import SharedCardEncoder, CardEncoderConfig, CardFeatureExtractor
from training_pipeline import DraftModel, TrainingConfig

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI
app = FastAPI(
    title="MTG RL Draft API",
    description="AI-powered draft recommendations",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pt")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
RATE_LIMIT = os.getenv("RATE_LIMIT", "100/minute")

# Global model (loaded on startup)
model = None
feature_extractor = None


class Card(BaseModel):
    """Card representation."""
    name: str
    mana_cost: str = ""
    type: str = ""
    power: Optional[int] = None
    toughness: Optional[int] = None
    keywords: List[str] = Field(default_factory=list)
    rarity: str = "common"


class PickRequest(BaseModel):
    """Request for a pick recommendation."""
    pack: List[Card] = Field(..., min_length=1, max_length=15)
    pool: List[Card] = Field(default_factory=list, max_length=45)


class PickResponse(BaseModel):
    """Response with pick recommendation."""
    recommended_pick: int
    pick_probabilities: List[float]
    confidence: float
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model, feature_extractor

    # Initialize model
    config = TrainingConfig()
    model = DraftModel(config)

    # Load weights
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {MODEL_PATH}")
    elif MODEL_PATH.startswith("s3://"):
        # Download from S3
        s3 = boto3.client("s3")
        bucket = MODEL_PATH.split("/")[2]
        key = "/".join(MODEL_PATH.split("/")[3:])
        local_path = "/tmp/model.pt"
        s3.download_file(bucket, key, local_path)
        checkpoint = torch.load(local_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from S3: {MODEL_PATH}")
    else:
        print("Warning: No model weights found, using random initialization")

    model.eval()

    # Initialize feature extractor
    feature_extractor = CardFeatureExtractor(CardEncoderConfig())

    print("Model loaded and ready for inference")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0",
    )


@app.post("/api/v1/draft/pick", response_model=PickResponse)
@limiter.limit(RATE_LIMIT)
async def get_pick_recommendation(request: Request, pick_request: PickRequest):
    """
    Get a pick recommendation for the current pack.

    Rate limited to prevent abuse.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # Extract features
        pack_features = []
        for card in pick_request.pack:
            features = feature_extractor.extract(card.model_dump())
            pack_features.append(features)

        pool_features = []
        for card in pick_request.pool:
            features = feature_extractor.extract(card.model_dump())
            pool_features.append(features)

        # Pad to expected sizes
        while len(pack_features) < 15:
            pack_features.append(np.zeros_like(pack_features[0]))
        while len(pool_features) < 45:
            if pool_features:
                pool_features.append(np.zeros_like(pool_features[0]))
            else:
                pool_features.append(np.zeros(feature_extractor.config.input_dim))

        # Convert to tensors
        import numpy as np
        pack_tensor = torch.tensor(np.stack(pack_features)).unsqueeze(0).float()
        pool_tensor = torch.tensor(np.stack(pool_features)).unsqueeze(0).float()
        pack_mask = torch.zeros(1, 15)
        pack_mask[:, :len(pick_request.pack)] = 1
        pool_mask = torch.zeros(1, 45)
        pool_mask[:, :len(pick_request.pool)] = 1

        # Inference
        with torch.no_grad():
            logits, _ = model(pack_tensor, pool_tensor, pack_mask, pool_mask)

        # Apply mask and softmax
        logits = logits.masked_fill(pack_mask == 0, float("-inf"))
        probs = torch.softmax(logits, dim=-1)[0].tolist()

        # Get recommendation
        recommended = int(torch.argmax(logits[0]).item())
        confidence = probs[recommended]

        latency = (time.time() - start_time) * 1000

        return PickResponse(
            recommended_pick=recommended,
            pick_probabilities=probs[:len(pick_request.pack)],
            confidence=confidence,
            latency_ms=latency,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats")
@limiter.limit("10/minute")
async def get_stats(request: Request):
    """Get API statistics."""
    return {
        "total_requests": 0,  # Would track with Redis/DynamoDB
        "avg_latency_ms": 0,
        "model_version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## CLI Deployment Commands

### Setup AWS CLI

```bash
# Configure AWS CLI (one-time setup)
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-west-2), Output format (json)

# Verify configuration
aws sts get-caller-identity
```

### Deploy Infrastructure

```bash
# Initialize Terraform
cd infrastructure
terraform init

# Preview changes
terraform plan

# Deploy
terraform apply -auto-approve

# Get outputs
terraform output
```

### Build and Push Docker Image

```bash
# Get ECR login
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-west-2.amazonaws.com

# Build image
docker build -t mtg-rl-inference .

# Tag for ECR
docker tag mtg-rl-inference:latest <account>.dkr.ecr.us-west-2.amazonaws.com/mtg-rl-inference:latest

# Push
docker push <account>.dkr.ecr.us-west-2.amazonaws.com/mtg-rl-inference:latest

# Force ECS to use new image
aws ecs update-service --cluster mtg-rl-cluster --service mtg-rl-inference --force-new-deployment
```

### Upload Model to S3

```bash
# Upload trained model
aws s3 cp checkpoints/bc_best.pt s3://mtg-rl-models-us-west-2/production/model.pt

# Verify
aws s3 ls s3://mtg-rl-models-us-west-2/production/
```

### Monitor

```bash
# View ECS service status
aws ecs describe-services --cluster mtg-rl-cluster --services mtg-rl-inference

# View CloudWatch logs
aws logs tail /ecs/mtg-rl-inference --follow

# View WAF metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/WAFV2 \
  --metric-name BlockedRequests \
  --dimensions Name=WebACL,Value=mtg-rl-waf \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --period 300 \
  --statistics Sum
```

---

## Cost Estimates

### Inference (Production)

| Resource | Configuration | Cost/Month |
|----------|---------------|------------|
| ECS Fargate | 2 tasks × 2 vCPU × 4GB | ~$60 |
| ALB | 1 load balancer | ~$20 |
| API Gateway | 10M requests | ~$35 |
| WAF | 1 Web ACL + rules | ~$10 |
| S3 | 10GB storage | ~$1 |
| CloudWatch | Logs + metrics | ~$10 |
| **Total** | | **~$136/month** |

### Training (On-Demand)

| Instance Type | GPU | Spot Price/Hour | Time | Cost |
|---------------|-----|-----------------|------|------|
| g5.xlarge | A10G 24GB | ~$0.50 | 20h | ~$10 |
| p3.2xlarge | V100 16GB | ~$0.90 | 15h | ~$14 |
| p3.8xlarge | 4× V100 | ~$3.60 | 5h | ~$18 |

### Scaling Estimates

| Traffic Level | Tasks | API Gateway | Total/Month |
|---------------|-------|-------------|-------------|
| Low (100K req) | 2 | ~$5 | ~$100 |
| Medium (1M req) | 4 | ~$35 | ~$180 |
| High (10M req) | 10 | ~$350 | ~$500 |

---

## Rate Limit Summary

| Layer | Limit | Scope | Action |
|-------|-------|-------|--------|
| WAF | 2000 req/5min | Per IP | Block |
| API Gateway | 50 req/sec | Global | Throttle |
| API Gateway | 100 burst | Global | Queue |
| Usage Plan | 10K req/day | Per API Key | Deny |
| Application | 100 req/min | Per IP | 429 |

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/RexGoliath1/mtg-rl.git
cd mtg-rl

# 2. Train model locally first
python training_pipeline.py --mode bc --sets FDN --epochs 10

# 3. Deploy to AWS
cd infrastructure
terraform init && terraform apply

# 4. Build and push Docker image
./scripts/deploy.sh

# 5. Test API
curl -X POST https://your-api-endpoint/api/v1/draft/pick \
  -H "Content-Type: application/json" \
  -d '{"pack": [{"name": "Lightning Bolt", "mana_cost": "{R}"}], "pool": []}'
```

---

## What You Need

1. **AWS Account** with appropriate permissions
2. **AWS CLI** configured locally
3. **Terraform** installed (v1.0+)
4. **Docker** for building images
5. **Trained model** (from training pipeline)

The rate limiting is built into multiple layers (WAF, API Gateway, application) so you're protected before deployment. Start with the Terraform setup and the default limits, then adjust based on your actual traffic patterns.
