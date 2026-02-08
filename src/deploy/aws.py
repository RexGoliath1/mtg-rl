"""
AWS EC2 operations for ForgeRL deployment.

Wraps boto3 calls for AMI lookup, VPC discovery, spot instance
launching, and instance lifecycle management.
"""

from __future__ import annotations

import base64
import time
from typing import Any, Optional

from src.deploy.config import DeployConfig


def _get_ec2_client(region: str):
    """Create a boto3 EC2 client for the given region."""
    import boto3
    return boto3.client("ec2", region_name=region)


def _get_sts_client(region: str):
    """Create a boto3 STS client for the given region."""
    import boto3
    return boto3.client("sts", region_name=region)


def validate_credentials(region: str = "us-east-1") -> str:
    """
    Validate AWS credentials and return the account ID.

    Raises:
        RuntimeError: If credentials are not configured or invalid.
    """
    try:
        sts = _get_sts_client(region)
        identity = sts.get_caller_identity()
        account_id = identity["Account"]
        print(f"  [OK] AWS credentials (account: {account_id})")
        return account_id
    except Exception as e:
        raise RuntimeError(f"AWS credentials not configured: {e}") from e


def find_ami(
    region: str,
    name_filter: str = "ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*",
    owners: Optional[list[str]] = None,
) -> str:
    """
    Find the latest AMI matching the given name filter.

    Args:
        region: AWS region.
        name_filter: AMI name glob pattern.
        owners: AMI owner IDs. Defaults to Canonical (Ubuntu).

    Returns:
        AMI ID string.

    Raises:
        RuntimeError: If no matching AMI is found.
    """
    if owners is None:
        # Canonical (Ubuntu official)
        owners = ["099720109477"]

    ec2 = _get_ec2_client(region)
    response = ec2.describe_images(
        Owners=owners,
        Filters=[
            {"Name": "name", "Values": [name_filter]},
            {"Name": "state", "Values": ["available"]},
        ],
    )

    images = response.get("Images", [])
    if not images:
        raise RuntimeError(f"No AMI found matching '{name_filter}' in {region}")

    # Sort by creation date, pick newest
    images.sort(key=lambda img: img.get("CreationDate", ""), reverse=True)
    ami_id = images[0]["ImageId"]
    ami_name = images[0].get("Name", "unknown")
    print(f"  [OK] AMI: {ami_id} ({ami_name})")
    return ami_id


def find_deep_learning_ami(region: str) -> str:
    """Find the latest AWS Deep Learning AMI (for GPU training instances)."""
    return find_ami(
        region=region,
        name_filter="Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *",
        owners=["amazon"],
    )


def get_vpc_info(region: str) -> dict[str, str]:
    """
    Get default VPC, first subnet, and create-or-find security group.

    Returns:
        Dict with keys: vpc_id, subnet_id, security_group_id.

    Raises:
        RuntimeError: If no default VPC is found.
    """
    ec2 = _get_ec2_client(region)

    # Default VPC
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    vpc_list = vpcs.get("Vpcs", [])
    if not vpc_list:
        raise RuntimeError(f"No default VPC found in {region}")
    vpc_id = vpc_list[0]["VpcId"]

    # First subnet
    subnets = ec2.describe_subnets(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
    )
    subnet_list = subnets.get("Subnets", [])
    if not subnet_list:
        raise RuntimeError(f"No subnets found in VPC {vpc_id}")
    subnet_id = subnet_list[0]["SubnetId"]

    print(f"  [OK] VPC: {vpc_id}, Subnet: {subnet_id}")
    return {
        "vpc_id": vpc_id,
        "subnet_id": subnet_id,
    }


def ensure_security_group(
    region: str,
    vpc_id: str,
    group_name: str = "mtg-rl-training-sg",
) -> str:
    """
    Find or create a security group with SSH access.

    Returns:
        Security group ID.
    """
    ec2 = _get_ec2_client(region)

    try:
        sgs = ec2.describe_security_groups(
            Filters=[
                {"Name": "group-name", "Values": [group_name]},
                {"Name": "vpc-id", "Values": [vpc_id]},
            ]
        )
        sg_list = sgs.get("SecurityGroups", [])
        if sg_list:
            sg_id = sg_list[0]["GroupId"]
            print(f"  [OK] Security group: {sg_id} ({group_name})")
            return sg_id
    except Exception:
        pass

    # Create new security group
    response = ec2.create_security_group(
        GroupName=group_name,
        Description="ForgeRL training and data collection instances",
        VpcId=vpc_id,
    )
    sg_id = response["GroupId"]

    # Allow SSH
    try:
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[{
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH access"}],
            }],
        )
    except Exception:
        pass  # Rule may already exist

    print(f"  [OK] Created security group: {sg_id} ({group_name})")
    return sg_id


def launch_spot_instance(
    config: DeployConfig,
    userdata: str,
    instance_type: str,
    ami_id: str,
    subnet_id: str,
    security_group_id: str,
    spot_price: str,
    volume_size_gb: int = 50,
    tags: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """
    Launch a spot instance with fallback to on-demand.

    Args:
        config: Deployment configuration.
        userdata: Bash userdata script content.
        instance_type: EC2 instance type.
        ami_id: AMI ID to launch.
        subnet_id: Subnet to launch in.
        security_group_id: Security group ID.
        spot_price: Maximum spot price as string.
        volume_size_gb: Root volume size.
        tags: Additional tags to apply.

    Returns:
        Dict with keys: instance_id, request_type ("spot" or "on-demand").
    """
    ec2 = _get_ec2_client(config.region)

    userdata_b64 = base64.b64encode(userdata.encode("utf-8")).decode("ascii")

    launch_spec: dict[str, Any] = {
        "ImageId": ami_id,
        "InstanceType": instance_type,
        "SecurityGroupIds": [security_group_id],
        "SubnetId": subnet_id,
        "IamInstanceProfile": {"Name": config.iam_instance_profile},
        "UserData": userdata_b64,
        "BlockDeviceMappings": [{
            "DeviceName": "/dev/sda1",
            "Ebs": {"VolumeSize": volume_size_gb, "VolumeType": "gp3"},
        }],
    }

    if config.key_name:
        launch_spec["KeyName"] = config.key_name

    # Try spot first
    instance_id = _try_spot_launch(ec2, config.region, launch_spec, spot_price)
    request_type = "spot"

    # Fallback to on-demand
    if not instance_id:
        print("  Spot not available, launching on-demand...")
        # On-demand uses raw userdata, not base64
        run_params: dict[str, Any] = {
            "ImageId": ami_id,
            "InstanceType": instance_type,
            "SecurityGroupIds": [security_group_id],
            "SubnetId": subnet_id,
            "IamInstanceProfile": {"Name": config.iam_instance_profile},
            "UserData": userdata,
            "MinCount": 1,
            "MaxCount": 1,
            "BlockDeviceMappings": [{
                "DeviceName": "/dev/sda1",
                "Ebs": {"VolumeSize": volume_size_gb, "VolumeType": "gp3"},
            }],
        }
        if config.key_name:
            run_params["KeyName"] = config.key_name

        response = ec2.run_instances(**run_params)
        instance_id = response["Instances"][0]["InstanceId"]
        request_type = "on-demand"

    # Apply tags
    all_tags = {"Project": "mtg-rl"}
    if tags:
        all_tags.update(tags)

    ec2.create_tags(
        Resources=[instance_id],
        Tags=[{"Key": k, "Value": v} for k, v in all_tags.items()],
    )

    print(f"  [OK] Launched {request_type} instance: {instance_id}")
    return {"instance_id": instance_id, "request_type": request_type}


def _try_spot_launch(
    ec2,
    region: str,
    launch_spec: dict[str, Any],
    spot_price: str,
    max_wait_checks: int = 30,
    wait_interval: int = 10,
) -> Optional[str]:
    """
    Attempt to launch a spot instance. Returns instance_id or None.
    """
    try:
        response = ec2.request_spot_instances(
            InstanceCount=1,
            Type="one-time",
            SpotPrice=spot_price,
            LaunchSpecification=launch_spec,
        )
        spot_request_id = response["SpotInstanceRequests"][0]["SpotInstanceRequestId"]
        print(f"  Spot request: {spot_request_id}")
    except Exception as e:
        print(f"  Spot request failed: {e}")
        return None

    # Poll for fulfillment
    for i in range(1, max_wait_checks + 1):
        time.sleep(wait_interval)
        try:
            desc = ec2.describe_spot_instance_requests(
                SpotInstanceRequestIds=[spot_request_id]
            )
            instance_id = desc["SpotInstanceRequests"][0].get("InstanceId")
            if instance_id:
                return instance_id
        except Exception:
            pass
        print(f"  Waiting for spot fulfillment... ({i}/{max_wait_checks})")

    # Cancel unfulfilled request
    try:
        ec2.cancel_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id])
    except Exception:
        pass
    print("  Spot request not fulfilled, cancelling.")
    return None


def wait_for_instance(
    instance_id: str,
    region: str,
    timeout: int = 300,
) -> str:
    """
    Wait for an instance to reach 'running' state and return its public IP.

    Args:
        instance_id: EC2 instance ID.
        region: AWS region.
        timeout: Maximum seconds to wait.

    Returns:
        Public IP address string.

    Raises:
        RuntimeError: If instance does not reach running state in time.
    """
    ec2 = _get_ec2_client(region)

    waiter = ec2.get_waiter("instance_running")
    try:
        waiter.wait(
            InstanceIds=[instance_id],
            WaiterConfig={"Delay": 10, "MaxAttempts": timeout // 10},
        )
    except Exception as e:
        raise RuntimeError(
            f"Instance {instance_id} did not reach running state within {timeout}s: {e}"
        ) from e

    response = ec2.describe_instances(InstanceIds=[instance_id])
    public_ip = (
        response["Reservations"][0]["Instances"][0]
        .get("PublicIpAddress", "N/A")
    )
    print(f"  [OK] Instance {instance_id} running at {public_ip}")
    return public_ip


def get_running_instances(
    region: str,
    tag_filter: str = "mtg-rl",
) -> list[dict[str, str]]:
    """
    List running instances tagged with the given project tag.

    Returns:
        List of dicts with keys: instance_id, instance_type, public_ip, name, launch_time.
    """
    ec2 = _get_ec2_client(region)

    response = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Project", "Values": [tag_filter]},
            {"Name": "instance-state-name", "Values": ["running", "pending"]},
        ]
    )

    instances = []
    for reservation in response.get("Reservations", []):
        for inst in reservation.get("Instances", []):
            name = ""
            for tag in inst.get("Tags", []):
                if tag["Key"] == "Name":
                    name = tag["Value"]
                    break
            instances.append({
                "instance_id": inst["InstanceId"],
                "instance_type": inst["InstanceType"],
                "public_ip": inst.get("PublicIpAddress", "N/A"),
                "name": name,
                "launch_time": str(inst.get("LaunchTime", "")),
                "state": inst["State"]["Name"],
            })

    return instances


def terminate_instances(instance_ids: list[str], region: str) -> None:
    """
    Terminate one or more EC2 instances.

    Args:
        instance_ids: List of instance IDs to terminate.
        region: AWS region.
    """
    if not instance_ids:
        print("  No instances to terminate.")
        return

    ec2 = _get_ec2_client(region)
    ec2.terminate_instances(InstanceIds=instance_ids)
    print(f"  [OK] Terminated {len(instance_ids)} instance(s): {', '.join(instance_ids)}")
