# Linux lab (EC2) — how to use it

The Stage 1 Linux lab is a free-tier EC2 instance in AWS Sydney (`ap-southeast-2`).

| Item | Value |
|---|---|
| Instance ID | `i-06ed5cf6bd058e8fe` |
| Type / OS | `t3.micro` · Ubuntu 24.04 LTS · x86_64 · 20 GB gp3 disk |
| Login user | `ubuntu` |
| SSH key | `~/.ssh/orbiteye-lab.pem` (never commit this; it is outside the repo) |
| Security group | `sg-0f6fdf53910772be4` — port 22 open **only from my home IP** |
| AWS CLI profile | `orbiteye` (IAM user `ali-admin`; root is not used for daily work) |

## Resume a session (start here each day)

The instance is normally **stopped** between sessions. Stopped = no compute charges; only the 20 GB disk remains, and it is inside the free tier.

From the repo root on the Mac:

```bash
scripts/lab.sh start    # starts the instance, waits, writes the new public IP into ~/.ssh/config
ssh orbiteye-lab        # log in
```

or in one step: `scripts/lab.sh ssh`.

When you finish for the day:

```bash
exit                    # leave the Linux box (Ctrl-b d first if you want to keep a tmux session)
scripts/lab.sh stop     # stop the instance
```

Check state at any time with `scripts/lab.sh status`. The public IP changes on every start; the script handles that, so you never edit `~/.ssh/config` by hand.

## Log in

Open Terminal on the Mac and run:

```bash
ssh orbiteye-lab
```

`orbiteye-lab` is an alias defined in `~/.ssh/config`:

```
Host orbiteye-lab
    HostName <current public IP>
    User ubuntu
    IdentityFile ~/.ssh/orbiteye-lab.pem
    IdentitiesOnly yes
```

Long form, if the alias is ever missing:

```bash
ssh -i ~/.ssh/orbiteye-lab.pem ubuntu@<public IP>
```

The first connection asks to trust the host fingerprint; answer `yes`. The prompt then looks like `ubuntu@ip-172-31-41-1:~$`.

## Work inside tmux

Start (or re-attach to) a session immediately after logging in so a dropped Wi-Fi connection does not kill your work:

```bash
tmux new -As lab      # create session "lab", or attach if it already exists
```

| Keys | Action |
|---|---|
| `Ctrl-b d` | detach (session keeps running) |
| `Ctrl-b c` | new window |
| `Ctrl-b n` / `Ctrl-b p` | next / previous window |
| `Ctrl-b %` / `Ctrl-b "` | split pane vertically / horizontally |
| `tmux ls` | list sessions (from a normal shell) |

## Log out

```bash
exit          # or Ctrl-d
```

If you are inside tmux, `exit` closes only that tmux window. Detach with `Ctrl-b d` first if you want the session to survive, then `exit` the SSH shell.

## Stop and start (cost control)

A stopped instance costs nothing for compute; the 20 GB disk is inside the free tier. **Stop it at the end of every session.**

```bash
export AWS_PROFILE=orbiteye
aws ec2 stop-instances  --instance-ids i-06ed5cf6bd058e8fe
aws ec2 start-instances --instance-ids i-06ed5cf6bd058e8fe
aws ec2 describe-instances --instance-ids i-06ed5cf6bd058e8fe \
  --query 'Reservations[0].Instances[0].[State.Name,PublicIpAddress]' --output text
```

The **public IP changes after every stop/start**. Update the `HostName` line in `~/.ssh/config` with the new IP, or run:

```bash
NEWIP=$(aws ec2 describe-instances --instance-ids i-06ed5cf6bd058e8fe \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
sed -i '' "/^Host orbiteye-lab/,/^Host /{s/^\( *HostName \).*/\1$NEWIP/;}" ~/.ssh/config
```

## If SSH times out

Usually your home IP has changed and the security group no longer matches. Replace the rule:

```bash
export AWS_PROFILE=orbiteye
SG=sg-0f6fdf53910772be4
OLD=$(aws ec2 describe-security-groups --group-ids $SG \
  --query 'SecurityGroups[0].IpPermissions[0].IpRanges[0].CidrIp' --output text)
NEW=$(curl -s https://checkip.amazonaws.com)/32
aws ec2 revoke-security-group-ingress    --group-id $SG --protocol tcp --port 22 --cidr $OLD
aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 22 --cidr $NEW
```

## Copy files

```bash
scp file.txt orbiteye-lab:~/            # Mac → lab
scp orbiteye-lab:~/result.log .         # lab → Mac
rsync -avz ./dir/ orbiteye-lab:~/dir/   # sync a folder
```

## Tear down (end of Stage 1, or if you want a clean box)

```bash
export AWS_PROFILE=orbiteye
aws ec2 terminate-instances --instance-ids i-06ed5cf6bd058e8fe
aws ec2 wait instance-terminated --instance-ids i-06ed5cf6bd058e8fe
aws ec2 delete-security-group --group-id sg-0f6fdf53910772be4
aws ec2 delete-key-pair --key-name orbiteye-lab
```
