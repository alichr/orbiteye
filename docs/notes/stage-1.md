# Stage 1 notes — Linux + Git

Rules for this file: every concept gets 2–4 sentences **in my own words**, written after trying it on the lab box or in a scratch repo. If I cannot explain it, I read the linked resource until I can. No copy-paste from docs.

Resources: *The Linux Command Line* (Shotts) ch. 1–10, 14–17, 24–26 · MIT Missing Semester · *Pro Git* ch. 1–3, 5, 7.6 · `man` pages on the lab box.

---

## 1. Linux

### Filesystem hierarchy (`/etc`, `/var`, `/opt`, `/proc`, `/home`, `/tmp`, `/usr`)
<!-- try: ls /; ls /etc | head; ls /proc | head; cat /proc/cpuinfo | head -->
-

### Users, groups, permissions, `sudo`
<!-- try: id; ls -l /etc/passwd; chmod, chown, sudo -l -->
-

### Processes and signals (`SIGTERM` vs `SIGKILL`, `SIGHUP`, `SIGINT`)
<!-- try: sleep 300 &; ps aux | grep sleep; kill -TERM <pid>; kill -9 <pid> -->
-

### stdin / stdout / stderr and redirection (`>`, `>>`, `2>`, `2>&1`, `|`, `<`)
<!-- try: ls /nope 2> err.txt; ls / > out.txt 2>&1; cat out.txt | wc -l -->
-

### Exit codes (`$?`, `&&`, `||`, `set -e`)
<!-- try: true; echo $?; false; echo $?; ls /nope || echo failed -->
-

### Environment variables vs shell variables, `export`, `env`, `.bashrc` vs `.profile`
<!-- try: FOO=1; bash -c 'echo $FOO'; export FOO; bash -c 'echo $FOO' -->
-

### `PATH` and how the shell finds commands
<!-- try: echo $PATH; which python3; type ls; command -v uv -->
-

### Symlinks vs hard links
<!-- try: ln -s /etc/hostname link; ls -l link; ln /etc/hostname hard (may fail: why?) -->
-

### What a daemon is
-

### `systemd` units and `journalctl`
<!-- try: systemctl status ssh; systemctl list-units --type=service | head; journalctl -u ssh -n 20 -->
-

### SSH keys and `~/.ssh/config`, `authorized_keys`, `known_hosts`
<!-- look at: cat ~/.ssh/config on the Mac; cat ~/.ssh/authorized_keys on the lab box -->
-

### Package managers (`apt update` vs `apt upgrade` vs `apt install`)
<!-- try: sudo apt update; apt list --upgradable | head -->
-

### Disk usage (`df -h`, `du -sh`, `du -h --max-depth=1 | sort -h`)
-

### Networking basics (`ip a`, `ss -tlnp`, `curl -I`, ports, `/etc/hosts`, DNS)
<!-- try: ip a; ss -tlnp; curl -I https://example.com; cat /etc/hosts -->
-

### cron syntax (`* * * * *` fields, `crontab -e`, where output goes)
-

---

## 2. Git

### The three areas: working tree, index (staging area), HEAD
<!-- try: touch f; git status; git add f; git status; git diff --staged -->
-

### Commits are snapshots, not diffs
-

### Branches are pointers; what `HEAD` points to
<!-- try: cat .git/HEAD; cat .git/refs/heads/main -->
-

### Fast-forward vs merge commit vs rebase
-

### Detached HEAD
<!-- try: git checkout <sha>; git status; git switch - -->
-

### `reflog`: what it records and how it rescues you
-

### Remotes and tracking branches (`origin/main` vs `main`)
-

### Tags: lightweight vs annotated
-

### `.gitignore` semantics (patterns, negation, why an already-tracked file is not ignored)
-

### Why large binaries do not belong in git (and what DVC does instead)
-

### What a pull request actually is
-

---

## 3. Python packaging

### `pyproject.toml`: what sections exist and who reads them
-

### Editable installs (`uv pip install -e .`) vs normal installs
-

### Lockfiles: why `requirements.txt` pinning is not enough
-

### Virtual environments: what `.venv` contains and what `activate` changes
<!-- try: cat .venv/bin/activate | grep PATH; which python before/after -->
-

### Entry points / console scripts
-

---

## Interview questions (answer aloud, then write the answer in one paragraph each)

1. `git merge` vs `git rebase`, and when would you forbid rebase?
2. A process ignores Ctrl-C. What do you do, step by step?
3. How does SSH key authentication work? What is in `authorized_keys`?
4. What happens at boot on a systemd system? How do you make a service start on boot?
5. Why does a lockfile matter if `requirements.txt` already pins versions?

---

## What I learned (10 lines, written at the end of the stage)

1.
