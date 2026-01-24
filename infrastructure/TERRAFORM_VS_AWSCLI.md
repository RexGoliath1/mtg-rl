# Terraform vs AWS CLI for MTG RL Infrastructure

## Our Choice: Terraform

We use **Terraform** for this project. Here's why:

## Comparison

| Aspect | Terraform | AWS CLI |
|--------|-----------|---------|
| **Scalability** | Excellent - change variables to scale | Manual - rerun commands with new params |
| **Reproducibility** | Perfect - same config = same infra | Requires scripting discipline |
| **State Management** | Built-in state tracking | None - must track manually |
| **Rollback** | `terraform destroy` or revert code | Manual cleanup |
| **Multi-region** | Native provider aliasing | Separate scripts per region |
| **Team Collaboration** | Version control + state locking | Error-prone without tooling |
| **Learning Curve** | Moderate (HCL syntax) | Lower (familiar CLI) |
| **Debugging** | `terraform plan` shows changes | Trial and error |

## When to Use Each

### Use Terraform When:
- Infrastructure will evolve over time
- Multiple environments (dev/staging/prod)
- Team collaboration
- Need audit trail of changes
- Complex dependencies between resources

### Use AWS CLI When:
- One-off tasks (checking costs, quick queries)
- Simple scripts that won't change
- Interactive exploration
- CI/CD pipeline steps (alongside Terraform)

## Scaling with Terraform

To scale training resources, just change variables:

```hcl
# Scale from 1 to 4 training instances
variable "training_instance_count" {
  default = 1  # Change to 4 for more parallelism
}

# Scale instance size
variable "instance_type" {
  default = "g4dn.xlarge"  # Change to g4dn.2xlarge for more CPU
}
```

Then apply:
```bash
terraform apply -var="training_instance_count=4"
```

## File Structure

```
infrastructure/
├── main.tf              # Main resources
├── cost_controls.tf     # Budgets and alarms
├── variables.tf         # Input variables
├── outputs.tf           # Output values
└── terraform.tfvars     # Your specific values (gitignored)
```

## Quick Reference

```bash
# Initialize (first time)
terraform init

# Preview changes
terraform plan

# Apply changes
terraform apply

# Destroy all resources
terraform destroy

# Format code
terraform fmt

# Validate configuration
terraform validate
```
