#!/usr/bin/env bash
# Manage the Stage 1 EC2 Linux lab from the Mac.
#   scripts/lab.sh start   # start instance, wait, refresh ~/.ssh/config with the new IP
#   scripts/lab.sh stop    # stop instance (no compute charges while stopped)
#   scripts/lab.sh status  # state + public IP
#   scripts/lab.sh ssh     # start if needed, then open an SSH session
set -euo pipefail
export AWS_PROFILE="${AWS_PROFILE:-orbiteye}"
ID="i-06ed5cf6bd058e8fe"
HOST="orbiteye-lab"

state() { aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].State.Name' --output text; }
ip()    { aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text; }

refresh_ssh_config() {
  local newip; newip=$(ip)
  # replace the HostName line inside the "Host orbiteye-lab" block
  sed -i '' "/^Host $HOST\$/,/^Host /{s/^\( *HostName \).*/\1$newip/;}" ~/.ssh/config
  echo "ssh config updated: $HOST -> $newip"
}

case "${1:-status}" in
  start)
    if [[ $(state) != "running" ]]; then
      aws ec2 start-instances --instance-ids "$ID" >/dev/null
      echo "starting..."; aws ec2 wait instance-running --instance-ids "$ID"
    fi
    refresh_ssh_config
    echo "ready: ssh $HOST" ;;
  stop)
    aws ec2 stop-instances --instance-ids "$ID" >/dev/null
    echo "stopping..."; aws ec2 wait instance-stopped --instance-ids "$ID"
    echo "stopped (no compute charges)" ;;
  status)
    echo "state: $(state)   ip: $(ip)" ;;
  ssh)
    "$0" start; sleep 5; exec ssh "$HOST" ;;
  *) echo "usage: $0 {start|stop|status|ssh}"; exit 1 ;;
esac
