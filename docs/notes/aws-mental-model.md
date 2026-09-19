# AWS mental model (what we set up, and how the pieces relate)

Written on day 1 after creating the account, an IAM user and the EC2 lab. Revisit in Stage 5.

## 1. The account — the container

Everything lives inside one **AWS account**, identified by a 12-digit number (`772530337172`). It has one bill, one owner identity (**root**, the email used at sign-up), and one budget watching every service inside it.

Root can do anything, including close the account, so the rule is: **MFA on root, use it only for billing and account settings, then leave it alone.**

## 2. IAM — the door

**IAM (Identity and Access Management)** answers "who is asking, and are they allowed?" Nothing in AWS happens without an identity and a permission check.

| IAM object | What it is | Ours |
|---|---|---|
| **User** | A person or program with long-lived credentials | `ali-admin` |
| **Policy** | A JSON document listing allowed actions on resources | `AdministratorAccess` attached to `ali-admin` |
| **Credentials** | Password (+ MFA) for the console; access key for the CLI | passkey MFA; CLI profile `orbiteye` |
| **Role** | An identity that a machine or service *assumes* temporarily; no stored password | Stage 5: EC2 instance profile, GitHub Actions via OIDC, EKS pods via IRSA |

Roles are how professionals avoid keys on laptops and in CI. Users are for humans; roles are for everything else.

## 3. EC2 — renting a computer

**EC2 (Elastic Compute Cloud)** is a virtual machine on AWS hardware. Our lab is a `t3.micro` running Ubuntu 24.04 in the **Sydney region** (`ap-southeast-2`). We reach it over SSH. It bills by the second while running, so it is stopped between sessions.

Around every instance sit a few networking objects:

| Object | Role | Ours |
|---|---|---|
| **Region** | A geographic cluster of data centres | Sydney |
| **Availability Zone** | One data centre within a region | `ap-southeast-2b` |
| **VPC** | Your private network inside the region | the default VPC |
| **Subnet** | A slice of the VPC in one AZ | default public subnet |
| **Security group** | The firewall for an instance (allow rules only) | `orbiteye-lab-ssh`: port 22 from my home IP only |
| **Key pair** | The SSH key the instance trusts | `orbiteye-lab` |

**Lesson learned on day 1:** the console's *Launch instance* button creates a security group that opens SSH to `0.0.0.0/0` (the whole internet). Scanners find such instances within minutes. This is why infrastructure comes from scripts or Terraform, never from clicking.

## 4. The other services in this plan

Each is a building block with its own price and its own IAM permissions.

| Need | AWS service | Stage |
|---|---|---|
| Store files (data, models, inference logs) | **S3** | 2 (DVC remote), 5+ |
| Store Docker images | **ECR** | 4–5 |
| Run Kubernetes | **EKS** | 6 |
| Metrics, logs, alarms | **CloudWatch** | 5, 7 |
| Identity for CI and pods without secrets | **IAM roles + OIDC / IRSA** | 5, 6 |
| Create all of the above from code | **Terraform** (not an AWS service; the tool that drives the AWS API) | 5 |

## 5. SageMaker — the managed ML platform

**SageMaker** is AWS's all-in-one ML service: notebooks, training jobs on rented CPUs/GPUs, a model registry, and hosted inference endpoints. You hand it a script and a container and it runs the servers. It is the **PaaS** way to do ML. Azure's equivalent is **Azure ML** (touched in Stage 5).

This plan deliberately does **not** use SageMaker for the core work. We build the equivalent ourselves:

| SageMaker feature | What we build instead |
|---|---|
| Experiments & model registry | MLflow (Stage 2) |
| Training jobs | Docker image on EC2 / CI (Stages 3–5) |
| Hosted endpoints with autoscaling | FastAPI + ONNX Runtime on EKS with HPA (Stage 6) |
| Model monitor | Prometheus + Grafana + Evidently (Stage 7) |

**Why:** it teaches what the managed service hides, and the skills transfer to Azure ML, GCP, or on-premise. The honest interview answer to "why not SageMaker?": for a small team it is often the right choice, and I understand exactly what it does under the hood because I built each piece once.

## 6. The three tiers

| Tier | Meaning | Examples here |
|---|---|---|
| **IaaS** | Raw parts: machines, disks, networks | EC2, S3, VPC |
| **Container / orchestration** | You bring containers, the platform schedules them | EKS, Azure Container Apps |
| **PaaS (managed ML)** | Assembled machines for one job | SageMaker, Azure ML |

This plan mostly lives at the IaaS and Kubernetes layers and visits PaaS in Stage 5 so the comparison is first-hand.

## Interview one-liners to remember

- **Root vs IAM user:** root owns the account; an IAM user is an identity inside it with only the permissions its policies grant.
- **User vs role:** a user has long-lived credentials for a human; a role is assumed temporarily by a machine or service and has none.
- **Security group:** a stateful, allow-only firewall attached to an instance's network interface.
- **Region vs AZ:** a region is a geography with several isolated data centres (AZs); spread across AZs for resilience.
- **Stop vs terminate:** stop pauses the VM and keeps the disk (no compute charge); terminate deletes it.
