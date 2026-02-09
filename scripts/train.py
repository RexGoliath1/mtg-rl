#!/usr/bin/env python3
"""
ForgeRL Unified Training CLI

Replaces 9+ shell scripts with one Python command for all training operations:
  collect          - Launch data collection on AWS
  train            - Launch GPU training on AWS
  run              - Full pipeline (collect -> train -> notify)
  local            - Local smoke test (no AWS)
  status           - Check running instances and S3 data
  connect          - Connect to running instance
  kill             - Terminate all training instances
  docker-push      - Build and push Docker images to GHCR
  ssh              - SSH into running instance via EC2 Instance Connect
  deploy-training  - Launch EC2 spot instance for GPU training (Terraform)

Usage:
    python3 scripts/train.py local --epochs 5
    python3 scripts/train.py collect --games 10000 --dry-run
    python3 scripts/train.py train --epochs 300 --wandb --dry-run
    python3 scripts/train.py run --games 10000 --epochs 300 --notify --dry-run
    python3 scripts/train.py status
    python3 scripts/train.py connect --tensorboard --jupyter
    python3 scripts/train.py kill
    python3 scripts/train.py docker-push --all
    python3 scripts/train.py ssh logs
    python3 scripts/train.py deploy-training --instance-type g4dn.xlarge
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional deploy module (created by parallel agent -- may not exist yet)
# ---------------------------------------------------------------------------
try:
    from src.deploy.config import CollectionConfig, TrainingConfig
    from src.deploy.aws import (
        validate_credentials,
        launch_spot_instance,
        get_running_instances,
        terminate_instances,
    )
    from src.deploy.cost import estimate_costs, check_budget, get_current_month_spend
    from src.deploy.package import (
        create_code_tarball,
        create_forge_tarball,
        upload_to_s3,
    )
    from src.deploy.userdata import (
        generate_collection_userdata,
        generate_docker_collection_userdata,
        generate_training_userdata,
    )
    from src.deploy.monitor import poll_s3_completion

    DEPLOY_AVAILABLE = True
except ImportError:
    DEPLOY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
S3_BUCKET = "mtg-rl-checkpoints-20260124190118616600000001"
REGION = os.environ.get("AWS_REGION", "us-east-1")
MONTHLY_BUDGET = 100  # Hard cap in USD
LOCAL_MAX_ITERATIONS = 50  # Cloud-first rule

# ANSI colours
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _ok(msg: str) -> str:
    return f"{_GREEN}[OK]{_RESET} {msg}"


def _warn(msg: str) -> str:
    return f"{_YELLOW}[WARN]{_RESET} {msg}"


def _err(msg: str) -> str:
    return f"{_RED}[ERROR]{_RESET} {msg}"


def _info(msg: str) -> str:
    return f"{_CYAN}[INFO]{_RESET} {msg}"


def _heading(msg: str) -> str:
    bar = "=" * 60
    return f"\n{_BOLD}{bar}\n{msg}\n{bar}{_RESET}"


# ---------------------------------------------------------------------------
# .env loader (stdlib only -- no python-dotenv dependency)
# ---------------------------------------------------------------------------
def _load_dotenv() -> None:
    """Read .env file from project root and inject into os.environ."""
    dotenv = PROJECT_DIR / ".env"
    if not dotenv.exists():
        return
    for line in dotenv.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# Guard: require deploy module for AWS subcommands
# ---------------------------------------------------------------------------
def _require_deploy(subcommand: str) -> None:
    if DEPLOY_AVAILABLE:
        return
    print(_err(f"'{subcommand}' requires the src.deploy package which is not installed."))
    print("Install it with:  uv sync --all-extras")
    print("Or check that src/deploy/ exists and contains __init__.py.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand: local
# ---------------------------------------------------------------------------
def cmd_local(args: argparse.Namespace) -> None:
    """Run a local smoke test -- no AWS, no deploy module needed."""
    epochs = min(args.epochs, LOCAL_MAX_ITERATIONS)
    if args.epochs > LOCAL_MAX_ITERATIONS:
        print(_warn(
            f"Clamped epochs from {args.epochs} to {LOCAL_MAX_ITERATIONS} "
            "(cloud-first rule: >50 iterations must run on AWS)"
        ))

    mode = args.mode
    print(_heading(f"LOCAL SMOKE TEST -- {mode.upper()}"))
    print(f"  Mode:   {mode}")
    print(f"  Epochs: {epochs}")
    print()

    if mode == "bc":
        cmd = [
            sys.executable, "-m", "src.training.forge_imitation",
            "--train-only",
            "--epochs", str(epochs),
        ]
    elif mode == "selfplay":
        cmd = [
            sys.executable,
            str(PROJECT_DIR / "scripts" / "training_pipeline.py"),
            "--mode", "rl",
            "--episodes", str(epochs),
        ]
    else:
        print(_err(f"Unknown local mode: {mode}"))
        sys.exit(1)

    print(f"  Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Subcommand: collect
# ---------------------------------------------------------------------------
def cmd_collect(args: argparse.Namespace) -> None:
    _require_deploy("collect")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"collection_{ts}"

    cfg = CollectionConfig(
        num_games=args.games,
        num_workers=args.workers,
        instance_type=args.instance,
    )

    use_docker = not getattr(args, "legacy", False)
    method = "Docker" if use_docker else "Legacy (S3 tarballs)"

    print(_heading("DATA COLLECTION"))
    print(f"  Method:   {method}")
    print(f"  Games:    {cfg.num_games}")
    print(f"  Workers:  {cfg.num_workers}")
    print(f"  Instance: {cfg.instance_type}")
    print(f"  Output:   s3://{S3_BUCKET}/imitation_data/{run_id}/")
    print()

    # Validate + cost
    validate_credentials()
    print(_ok("AWS credentials"))

    cost_est = estimate_costs(collection_config=cfg)
    spend = get_current_month_spend()
    remaining = MONTHLY_BUDGET - spend

    print(f"\n  Estimated cost:    ${cost_est.total:.2f}")
    print(f"  Month-to-date:     ${spend:.2f}")
    print(f"  Budget remaining:  ${remaining:.2f} of ${MONTHLY_BUDGET}")

    if not check_budget(cost_est.total, MONTHLY_BUDGET):
        print(_err(f"Estimated cost (${cost_est.total:.2f}) exceeds remaining budget (${remaining:.2f})"))
        sys.exit(1)
    print(_ok("Budget check passed"))

    if args.dry_run:
        _print_dry_run_plan("collect", cfg, cost_est.total)
        return

    # Import additional deploy helpers
    from src.deploy.aws import find_ami, get_vpc_info, ensure_security_group
    from src.deploy.config import DeployConfig

    image_registry = "ghcr.io/rexgoliath1"
    deploy_cfg = DeployConfig(
        region=REGION,
        s3_bucket=S3_BUCKET,
        image_registry=image_registry,
    )

    if use_docker:
        # --- Docker-based deployment (recommended) ---
        print(_info("Using Docker images from GHCR"))
        print(f"  Image registry: {image_registry}")

        # Generate Docker-based userdata
        userdata = generate_docker_collection_userdata(
            deploy_config=deploy_cfg,
            collection_config=cfg,
            run_id=run_id,
            timestamp=ts,
            image_registry=image_registry,
        )
    else:
        # --- Legacy tarball-based deployment ---
        print(_info("Using legacy S3 tarball deployment"))

        # Package & upload
        print(_info("Packaging code..."))
        code_tar = create_code_tarball(PROJECT_DIR)
        forge_tar = create_forge_tarball(PROJECT_DIR)
        code_filename = f"{run_id}_code.tar.gz"
        forge_filename = f"{run_id}_forge.tar.gz"
        upload_to_s3(code_tar, S3_BUCKET, f"test_packages/{code_filename}")
        if forge_tar:
            upload_to_s3(forge_tar, S3_BUCKET, f"test_packages/{forge_filename}")
        else:
            import boto3
            s3 = boto3.client("s3", region_name=REGION)
            resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="test_packages/forge_jar_")
            forge_jars = sorted(
                [obj["Key"] for obj in resp.get("Contents", [])],
                reverse=True,
            )
            if forge_jars:
                latest_jar = forge_jars[0].split("/")[-1]
                forge_filename = latest_jar
                print(_ok(f"Reusing pre-built Forge JAR: {latest_jar}"))
            else:
                print(_warn("No Forge JAR available -- instance will build from source (~7 min)"))
                forge_filename = ""
        print(_ok("Packages uploaded"))

        # Generate legacy userdata
        userdata = generate_collection_userdata(
            deploy_config=deploy_cfg,
            collection_config=cfg,
            run_id=run_id,
            timestamp=ts,
            code_package=code_filename,
            forge_package=forge_filename if forge_tar else "",
        )

    # Network setup
    print(_info("Setting up network..."))
    ami_id = find_ami(REGION)
    vpc_info = get_vpc_info(REGION)
    sg_id = ensure_security_group(REGION, vpc_info["vpc_id"])
    print(_ok(f"AMI: {ami_id}"))

    # Launch (Docker needs less disk -- 30GB vs 50GB)
    volume_size = 30 if use_docker else cfg.volume_size_gb

    result = launch_spot_instance(
        config=deploy_cfg,
        userdata=userdata,
        instance_type=cfg.instance_type,
        ami_id=ami_id,
        subnet_id=vpc_info["subnet_id"],
        security_group_id=sg_id,
        spot_price=str(cfg.spot_price),
        volume_size_gb=volume_size,
        tags={"Name": f"forgerl-{run_id}", "Project": "forgerl"},
    )
    instance_id = result["instance_id"]
    print(_ok(f"Spot instance launched: {instance_id} ({result['request_type']})"))

    # Poll -- with new logging, check S3 live log every 2 min
    print(_info("Polling for completion (timeout 3h)..."))
    print(_info(f"Live logs: aws s3 cp s3://{S3_BUCKET}/imitation_data/{run_id}/live_log.txt - | tail -50"))
    success = poll_s3_completion(
        S3_BUCKET,
        f"imitation_data/{run_id}/collection_complete.json",
        interval=120,
        timeout=10800,  # 3 hours
    )
    if success:
        print(_ok(f"Collection complete!  s3://{S3_BUCKET}/imitation_data/{run_id}/"))
    else:
        print(_err("Collection timed out after 3 hours. Check the instance manually."))
        print(_info(f"Instance: {instance_id} -- terminate with: python3 scripts/train.py kill"))
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> None:
    _require_deploy("train")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"training_{ts}"

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    wandb_entity = os.environ.get("WANDB_ENTITY", "")
    wandb_project = os.environ.get("WANDB_PROJECT", "forgerl")

    cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        instance_type=args.instance,
        region=REGION,
        s3_bucket=S3_BUCKET,
        run_id=run_id,
        data_path=args.data_path,
        wandb_enabled=args.wandb and bool(wandb_key),
        wandb_key=wandb_key,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
    )

    print(_heading("GPU TRAINING"))
    print(f"  Epochs:     {cfg.epochs}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  LR:         {cfg.learning_rate}")
    print(f"  Instance:   {cfg.instance_type}")
    print(f"  Data path:  {cfg.data_path or '(auto-detect latest)'}")
    print(f"  W&B:        {'enabled' if cfg.wandb_enabled else 'disabled'}")
    print()

    validate_credentials()
    print(_ok("AWS credentials"))

    cost = estimate_costs("train", cfg)
    spend = get_current_month_spend()
    remaining = MONTHLY_BUDGET - spend

    print(f"\n  Estimated cost:    ${cost:.2f}")
    print(f"  Month-to-date:     ${spend:.2f}")
    print(f"  Budget remaining:  ${remaining:.2f} of ${MONTHLY_BUDGET}")

    if not check_budget(cost, MONTHLY_BUDGET):
        print(_err(f"Estimated cost (${cost:.2f}) exceeds remaining budget (${remaining:.2f})"))
        sys.exit(1)
    print(_ok("Budget check passed"))

    if args.dry_run:
        _print_dry_run_plan("train", cfg, cost)
        return

    # Package & upload
    print(_info("Packaging code..."))
    code_tar = create_code_tarball(PROJECT_DIR)
    upload_to_s3(code_tar, S3_BUCKET, f"packages/{run_id}/code.tar.gz")
    print(_ok("Code uploaded"))

    # Launch
    userdata = generate_training_userdata(cfg)
    instance_id = launch_spot_instance(cfg, userdata)
    print(_ok(f"GPU instance launched: {instance_id}"))
    print()
    print(f"  Monitor: aws s3 ls s3://{S3_BUCKET}/training_runs/{run_id}/training_complete.json")
    print(f"  Results: aws s3 sync s3://{S3_BUCKET}/training_runs/{run_id}/ training_output/{run_id}/")


# ---------------------------------------------------------------------------
# Subcommand: run  (collect -> train -> notify)
# ---------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> None:
    _require_deploy("run")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    collect_id = f"collection_{ts}"
    train_id = f"training_{ts}"

    print(_heading("FULL PIPELINE"))

    # --- Phase 1: collect ---
    collect_ns = argparse.Namespace(
        games=args.games,
        workers=args.workers,
        instance=args.collect_instance,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        # Build lightweight configs to show the plan
        ccfg = CollectionConfig(
            games=args.games, workers=args.workers,
            instance_type=args.collect_instance,
            region=REGION, s3_bucket=S3_BUCKET, run_id=collect_id,
        )
        tcfg = TrainingConfig(
            epochs=args.epochs, batch_size=256, learning_rate=0.001,
            instance_type=args.train_instance,
            region=REGION, s3_bucket=S3_BUCKET, run_id=train_id,
        )
        validate_credentials()
        c_cost = estimate_costs("collect", ccfg)
        t_cost = estimate_costs("train", tcfg)
        total = c_cost + t_cost
        print(f"\n  Phase 1 (Collection): ${c_cost:.2f}")
        print(f"  Phase 2 (Training):   ${t_cost:.2f}")
        print(f"  TOTAL:                ${total:.2f}")
        _print_dry_run_plan("run", {"collect": ccfg, "train": tcfg}, total)
        return

    # Real execution
    print("\n--- Phase 1: Data Collection ---")
    cmd_collect(collect_ns)

    print("\n--- Phase 2: GPU Training ---")
    train_ns = argparse.Namespace(
        epochs=args.epochs,
        batch_size=256,
        lr=0.001,
        instance=args.train_instance,
        wandb=args.wandb,
        data_path=f"s3://{S3_BUCKET}/imitation_data/{collect_id}/",
        dry_run=False,
    )
    cmd_train(train_ns)

    # Optional email notification
    notify_email = os.environ.get("FORGERL_NOTIFY_EMAIL", "")
    if args.notify and notify_email:
        _send_notification(notify_email, train_id)


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------
def cmd_status(args: argparse.Namespace) -> None:
    _require_deploy("status")

    print(_heading("FORGERL STATUS"))

    validate_credentials()

    # Running instances
    instances = get_running_instances(REGION)
    if instances:
        print(f"\n{_BOLD}Running instances:{_RESET}")
        for inst in instances:
            name = inst.get("name", "unnamed")
            iid = inst["instance_id"]
            itype = inst["instance_type"]
            launch = inst.get("launch_time", "?")
            ip = inst.get("public_ip", "N/A")
            print(f"  {name:30s}  {iid}  {itype:15s}  IP={ip}  launched={launch}")
    else:
        print("\n  No running mtg-rl instances.")

    # Current month spend
    spend = get_current_month_spend()
    remaining = MONTHLY_BUDGET - spend
    print(f"\n{_BOLD}Budget:{_RESET}")
    print(f"  Month-to-date: ${spend:.2f}  |  Remaining: ${remaining:.2f} of ${MONTHLY_BUDGET}")

    # Latest S3 data
    print(f"\n{_BOLD}Latest S3 collections:{_RESET}")
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", f"s3://{S3_BUCKET}/imitation_data/", "--region", REGION],
            capture_output=True, text=True, timeout=15,
        )
        lines = [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]
        for line in lines[-5:]:
            print(f"  {line}")
        if not lines:
            print("  (none)")
    except Exception:
        print("  (could not list S3)")


# ---------------------------------------------------------------------------
# Subcommand: connect
# ---------------------------------------------------------------------------
def cmd_connect(args: argparse.Namespace) -> None:
    _require_deploy("connect")

    validate_credentials()
    instances = get_running_instances(REGION)
    if not instances:
        print(_err("No running mtg-rl instances found."))
        sys.exit(1)

    inst = instances[0]  # Most recent
    instance_id = inst["instance_id"]
    ip = inst.get("public_ip", "")

    # Build list of port forwards
    port_forwards = []
    if args.tensorboard:
        tb_port = args.tensorboard
        port_forwards.append((tb_port, 6006, "TensorBoard"))
    if args.jupyter:
        jp_port = args.jupyter
        port_forwards.append((jp_port, 8888, "Jupyter"))

    if port_forwards:
        # Port forwarding mode (SSH with -L flags)
        services = ", ".join(name for _, _, name in port_forwards)
        print(_heading(f"PORT FORWARD: {services}"))
        print(f"  Instance: {instance_id}")
        for local_port, remote_port, name in port_forwards:
            print(f"  {name:12s} http://localhost:{local_port} -> :{remote_port}")
        print("  Press Ctrl+C to stop\n")

        key_file = Path.home() / ".ssh" / "mtg-rl-training.pem"
        if key_file.exists() and ip:
            ssh_cmd = ["ssh", "-i", str(key_file), "-N", "-o", "StrictHostKeyChecking=no"]
            for local_port, remote_port, _ in port_forwards:
                ssh_cmd.extend(["-L", f"{local_port}:localhost:{remote_port}"])
            ssh_cmd.append(f"ubuntu@{ip}")

            for local_port, _, name in port_forwards:
                print(f"{name} available at http://localhost:{local_port}")

            subprocess.run(ssh_cmd)
        else:
            # Fall back to SSM for single-port forward (SSM only supports one port)
            if len(port_forwards) > 1:
                print(_err("SSM port forwarding supports only one port at a time."))
                print("  Set up SSH key (~/.ssh/mtg-rl-training.pem) for multi-port forwarding.")
                sys.exit(1)
            local_port, remote_port, name = port_forwards[0]
            print(f"{name} available at http://localhost:{local_port}")
            try:
                subprocess.run([
                    "aws", "ssm", "start-session",
                    "--target", instance_id,
                    "--document-name", "AWS-StartPortForwardingSession",
                    "--parameters", f'{{"portNumber":["{remote_port}"],"localPortNumber":["{local_port}"]}}',
                    "--region", REGION,
                ])
            except FileNotFoundError:
                print(_err("SSM plugin not found and no SSH key available."))
                sys.exit(1)
    elif args.ssh:
        key_file = Path.home() / ".ssh" / "mtg-rl-training.pem"
        if not ip:
            print(_err("Instance has no public IP. Use SSM instead."))
            sys.exit(1)
        if not key_file.exists():
            print(_err(f"SSH key not found: {key_file}"))
            print("  Use SSM instead or create key with ./scripts/setup_ssh.sh")
            sys.exit(1)
        print(f"Connecting via SSH to {ip}...")
        subprocess.run([
            "ssh", "-i", str(key_file),
            "-o", "StrictHostKeyChecking=no",
            f"ubuntu@{ip}",
        ])
    else:
        # Default: SSM
        print(f"Connecting via SSM to {instance_id}...")
        subprocess.run([
            "aws", "ssm", "start-session",
            "--target", instance_id,
            "--region", REGION,
        ])


# ---------------------------------------------------------------------------
# Subcommand: kill
# ---------------------------------------------------------------------------
def cmd_kill(args: argparse.Namespace) -> None:
    _require_deploy("kill")

    validate_credentials()
    instances = get_running_instances(REGION)

    if not instances:
        print("No running mtg-rl instances to terminate.")
        return

    print(_heading("TERMINATE INSTANCES"))
    for inst in instances:
        name = inst.get("name", "unnamed")
        iid = inst["instance_id"]
        itype = inst["instance_type"]
        print(f"  {name:30s}  {iid}  {itype}")

    if not args.force:
        confirm = input(f"\nTerminate {len(instances)} instance(s)? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    ids = [inst["instance_id"] for inst in instances]
    terminate_instances(ids, REGION)
    print(_ok(f"Terminated {len(ids)} instance(s)."))


# ---------------------------------------------------------------------------
# Subcommand: docker-push  (GHCR images are now pushed by CI automatically)
# ---------------------------------------------------------------------------
def cmd_docker_push(args: argparse.Namespace) -> None:
    """Build and push Docker images to GHCR."""
    print(_heading("DOCKER BUILD & PUSH"))
    print()
    print("  Docker images are now built and pushed to GHCR automatically by CI")
    print("  on every merge to main. No manual push needed.")
    print()
    print("  To build locally for testing:")
    print("    docker build -t mtg-daemon:latest -f infrastructure/docker/Dockerfile.daemon .")
    print("    docker build -t mtg-training:latest -f infrastructure/docker/Dockerfile.training .")
    print()
    print("  Images are at: ghcr.io/rexgoliath1/mtg-rl-{daemon,collection,training}:latest")


# ---------------------------------------------------------------------------
# Subcommand: ssh  (wraps ssh-instance.sh)
# ---------------------------------------------------------------------------
def cmd_ssh(args: argparse.Namespace) -> None:
    """SSH into running instance via EC2 Instance Connect."""
    script = PROJECT_DIR / "scripts" / "ssh-instance.sh"
    if not script.exists():
        print(_err(f"Script not found: {script}"))
        sys.exit(1)

    cmd = [str(script)]
    if args.mode:
        cmd.append(args.mode)

    print(_heading("SSH INTO INSTANCE"))
    print(f"  Script:  {script}")
    print(f"  Mode:    {args.mode or 'ssh (interactive)'}")
    print()

    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Subcommand: deploy-training  (wraps deploy_training.sh)
# ---------------------------------------------------------------------------
def cmd_deploy_training(args: argparse.Namespace) -> None:
    """Launch EC2 spot instance for GPU training via Terraform."""
    script = PROJECT_DIR / "scripts" / "deploy_training.sh"
    if not script.exists():
        print(_err(f"Script not found: {script}"))
        sys.exit(1)

    cmd = [str(script)]
    if args.instance_type:
        cmd.extend(["--instance-type", args.instance_type])
    if args.spot_price:
        cmd.extend(["--spot-price", args.spot_price])
    if args.s3_bucket:
        cmd.extend(["--s3-bucket", args.s3_bucket])
    if args.dry_run:
        cmd.append("--dry-run")

    print(_heading("DEPLOY TRAINING (Terraform)"))
    print(f"  Script:        {script}")
    print(f"  Instance type: {args.instance_type}")
    print(f"  Spot price:    ${args.spot_price}")
    print(f"  S3 bucket:     {args.s3_bucket or '(auto)'}")
    print(f"  Dry run:       {args.dry_run}")
    print()

    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _print_dry_run_plan(phase: str, cfg, cost: float) -> None:
    print(_heading("[DRY RUN] EXECUTION PLAN"))

    if phase == "collect":
        print(f"  - Launch {cfg.instance_type} spot instance")
        print(f"  - Collect {cfg.num_games} games with {cfg.num_workers} workers")
        print(f"  - Upload to s3://{S3_BUCKET}/imitation_data/")
    elif phase == "train":
        print(f"  - Launch {cfg.instance_type} spot instance (GPU)")
        print(f"  - Train for {cfg.epochs} epochs (batch={cfg.batch_size})")
        print(f"  - LR: {cfg.learning_rate}, AMP enabled")
        data = cfg.data_path or "(auto-detect latest collection)"
        print(f"  - Data: {data}")
        print(f"  - Save to s3://{S3_BUCKET}/training_runs/{cfg.run_id}/")
        if cfg.wandb_enabled:
            print(f"  - W&B: {cfg.wandb_entity}/{cfg.wandb_project}")
    elif phase == "run":
        print("  Phase 1: Data Collection (see 'collect' plan)")
        print("  Phase 2: GPU Training (see 'train' plan)")
        print("  Phase 3: Email notification (if FORGERL_NOTIFY_EMAIL set)")

    print(f"\n  Estimated cost: ${cost:.2f}")
    print("\n  To launch for real, remove --dry-run flag.")


def _send_notification(email: str, run_id: str) -> None:
    """Best-effort email notification."""
    try:
        from src.utils.email_notifier import EmailNotifier
        notifier = EmailNotifier()
        notifier.send_training_complete(
            metrics={"run_id": run_id, "model_name": f"ForgeRL {run_id}"},
        )
        print(_ok(f"Notification sent to {email}"))
    except Exception as exc:
        print(_warn(f"Email notification failed (non-fatal): {exc}"))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="ForgeRL Unified Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 scripts/train.py local --epochs 5\n"
            "  python3 scripts/train.py collect --games 10000 --dry-run\n"
            "  python3 scripts/train.py train --epochs 300 --wandb --dry-run\n"
            "  python3 scripts/train.py status\n"
            "  python3 scripts/train.py kill\n"
            "  python3 scripts/train.py docker-push --all\n"
            "  python3 scripts/train.py docker-push --build --push\n"
            "  python3 scripts/train.py ssh\n"
            "  python3 scripts/train.py ssh logs\n"
            "  python3 scripts/train.py deploy-training --dry-run\n"
            "  python3 scripts/train.py deploy-training --instance-type g4dn.xlarge\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- local ---
    p_local = sub.add_parser("local", help="Local smoke test (no AWS)")
    p_local.add_argument("--epochs", type=int, default=5, help="Number of epochs (max 50)")
    p_local.add_argument("--mode", choices=["bc", "selfplay"], default="bc", help="Training mode")
    p_local.set_defaults(func=cmd_local)

    # --- collect ---
    p_collect = sub.add_parser("collect", help="Launch data collection on AWS")
    p_collect.add_argument("--games", type=int, required=True, help="Number of games to collect")
    p_collect.add_argument("--workers", type=int, default=8, help="Parallel workers")
    p_collect.add_argument("--instance", type=str, default="c5.2xlarge", help="EC2 instance type")
    p_collect.add_argument("--legacy", action="store_true", help="Use legacy S3 tarball deployment instead of Docker")
    p_collect.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    p_collect.set_defaults(func=cmd_collect)

    # --- train ---
    p_train = sub.add_parser("train", help="Launch GPU training on AWS")
    p_train.add_argument("--epochs", type=int, required=True, help="Training epochs")
    p_train.add_argument("--batch-size", type=int, default=256, help="Batch size")
    p_train.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    p_train.add_argument("--instance", type=str, default="g4dn.xlarge", help="GPU instance type")
    p_train.add_argument("--wandb", action="store_true", help="Enable W&B tracking")
    p_train.add_argument("--data-path", type=str, default=None, help="S3 path to training data")
    p_train.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    p_train.set_defaults(func=cmd_train)

    # --- run ---
    p_run = sub.add_parser("run", help="Full pipeline: collect -> train -> notify")
    p_run.add_argument("--games", type=int, required=True, help="Number of games to collect")
    p_run.add_argument("--epochs", type=int, required=True, help="Training epochs")
    p_run.add_argument("--workers", type=int, default=8, help="Parallel workers for collection")
    p_run.add_argument("--collect-instance", type=str, default="c5.2xlarge", help="Collection instance")
    p_run.add_argument("--train-instance", type=str, default="g4dn.xlarge", help="Training instance")
    p_run.add_argument("--wandb", action="store_true", help="Enable W&B tracking")
    p_run.add_argument("--notify", action="store_true", help="Send email on completion")
    p_run.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    p_run.set_defaults(func=cmd_run)

    # --- status ---
    p_status = sub.add_parser("status", help="Check running instances and S3 data")
    p_status.set_defaults(func=cmd_status)

    # --- connect ---
    p_connect = sub.add_parser("connect", help="Connect to running instance")
    p_connect.add_argument("--ssh", action="store_true", help="Use SSH instead of SSM")
    p_connect.add_argument("--tensorboard", type=int, nargs="?", const=6006, default=None, help="Port-forward TensorBoard (default 6006)")
    p_connect.add_argument("--jupyter", type=int, nargs="?", const=8888, default=None, help="Port-forward Jupyter (default 8888)")
    p_connect.set_defaults(func=cmd_connect)

    # --- kill ---
    p_kill = sub.add_parser("kill", help="Terminate all training instances")
    p_kill.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    p_kill.set_defaults(func=cmd_kill)

    # --- docker-push (info only — CI handles GHCR pushes automatically) ---
    p_docker = sub.add_parser("docker-push", help="Show Docker image build/push info (CI handles this)")
    p_docker.set_defaults(func=cmd_docker_push)

    # --- ssh (wraps ssh-instance.sh) ---
    p_ssh = sub.add_parser("ssh", help="SSH into running instance via EC2 Instance Connect")
    p_ssh.add_argument("mode", nargs="?", choices=["logs", "status"], default=None,
                        help="Optional mode: 'logs' to tail logs, 'status' to check daemon")
    p_ssh.set_defaults(func=cmd_ssh)

    # --- deploy-training (wraps deploy_training.sh) ---
    p_deploy = sub.add_parser("deploy-training", help="Deploy GPU training via Terraform")
    p_deploy.add_argument("--instance-type", type=str, default="g4dn.xlarge", help="EC2 instance type")
    p_deploy.add_argument("--spot-price", type=str, default="0.20", help="Max spot price in USD")
    p_deploy.add_argument("--s3-bucket", type=str, default=None, help="S3 bucket for checkpoints")
    p_deploy.add_argument("--dry-run", action="store_true", help="Show Terraform plan without applying")
    p_deploy.set_defaults(func=cmd_deploy_training)

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
