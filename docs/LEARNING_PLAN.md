<div align="center">

# 🛰️ OrbitEye — End-to-End MLOps Learning Plan

### Satellite land-cover classification, from a Git repo skeleton to a Kubernetes service and an offline ARM64 edge device

[![Stages](https://img.shields.io/badge/stages-9-blue)](#-master-task-tracker-start-here)
[![Duration](https://img.shields.io/badge/duration-10–12_weeks-informational)](#-weekly-rhythm-810-h)
[![Effort](https://img.shields.io/badge/effort-8–10_h%2Fweek-informational)](#-weekly-rhythm-810-h)
[![Cloud budget](https://img.shields.io/badge/cloud_budget-US%2440–70-success)](#-appendix-b--cost-control-checklist)
[![Progress](https://img.shields.io/badge/progress-0%2F9-lightgrey)](#-master-task-tracker-start-here)

**[Tracker](#-master-task-tracker-start-here) · [Project](#project) · [Setup](#setup) · [Stage 1](#stage-1) · [Stage 2](#stage-2) · [Stage 3](#stage-3) · [Stage 4](#stage-4) · [Stage 5](#stage-5) · [Stage 6](#stage-6) · [Stage 7](#stage-7) · [Stage 8](#stage-8) · [Stage 9](#stage-9) · [Appendices](#appendix-a)**

</div>

---

> [!IMPORTANT]
> **Goal.** Finish one small ML project that legitimately covers every skill on this CV line:
>
> *MLOps / Infrastructure: Docker, Kubernetes, Git, CI/CD, model monitoring; AWS, Azure; Linux; edge deployment (embedded ARM / Raspberry Pi, onboard satellite systems)*
>
> Distributed multi-GPU training, HPC clusters and NVIDIA Jetson are deliberately left out: they need hardware you do not have. The edge target is a Raspberry Pi, fully exercised via ARM64 emulation.

> [!IMPORTANT]
> **Rule of the plan.** One stage = one skill, learned deeply, with a concrete artefact in the repo that proves it. You do not move to the next stage until the "You can claim it when…" checklist is done.

## 🗺️ Contents

| | Section | | Section |
|---|---|---|---|
| 📋 | [Master task tracker](#-master-task-tracker-start-here) | ☁️ | [Stage 5 — Cloud: AWS & Azure](#stage-5) |
| 🎯 | [The project](#project) | ☸️ | [Stage 6 — Serving & Kubernetes](#stage-6) |
| 🧰 | [Before you start: setup & how to study](#setup) | 📈 | [Stage 7 — Monitoring](#stage-7) |
| 🐧 | [Stage 1 — Linux + Git](#stage-1) | 🛰️ | [Stage 8 — Edge deployment](#stage-8) |
| 🧬 | [Stage 2 — ML design, DVC & MLflow](#stage-2) | 🎓 | [Stage 9 — Capstone](#stage-9) |
| 🐳 | [Stage 3 — Docker](#stage-3) | 📚 | [Primary resources](#resources) |
| 🔁 | [Stage 4 — CI/CD](#stage-4) | 🗂️ | [Appendices A–D](#appendix-a) |

---

## 📋 Master task tracker (start here)

Every stage below opens with a **"Your tasks"** checklist: every single thing you must do for that stage, in order, as tick-boxes. Tick them in this file as you go and commit. The detail sections after each checklist (Learn / Build / Exercises / Drills / Interview / Resources) explain *how*; the checklist is *what*.

| # | Stage | Weeks | Jump to tasks | Status |
|---|---|---|---|---|
| 0 | 🧰 Prerequisites & Mac setup | Day 1 | [Stage 0 tasks](#-your-tasks--stage-0-setup) | ⬜ |
| 1 | 🐧 Linux + Git repo skeleton | 1 | [Stage 1 tasks](#-your-tasks--stage-1) | ⬜ |
| 2 | 🧬 ML design, DVC, MLflow | 2–3 | [Stage 2 tasks](#-your-tasks--stage-2) | ⬜ |
| 3 | 🐳 Docker | 4 | [Stage 3 tasks](#-your-tasks--stage-3) | ⬜ |
| 4 | 🔁 CI/CD | 5 | [Stage 4 tasks](#-your-tasks--stage-4) | ⬜ |
| 5 | ☁️ Cloud: AWS (+ Azure) | 6–7 | [Stage 5 tasks](#-your-tasks--stage-5) | ⬜ |
| 6 | ☸️ Serving + Kubernetes | 8–9 | [Stage 6 tasks](#-your-tasks--stage-6) | ⬜ |
| 7 | 📈 Monitoring | 10 | [Stage 7 tasks](#-your-tasks--stage-7) | ⬜ |
| 8 | 🛰️ Edge deployment | 11–12 | [Stage 8 tasks](#-your-tasks--stage-8) | ⬜ |
| 9 | 🎓 Capstone | final | [Stage 9 tasks](#-your-tasks--stage-9) | ⬜ |

Legend: ⬜ not started · 🟨 in progress · ✅ complete. Update this table **and** the status table in the root `README.md` when a stage is done.

**The same six task groups repeat in every stage** (they mirror "How to study each stage" below):

| Group | What it means | When |
|---|---|---|
| **A. Setup** | Tools/accounts you need before you start | first hour |
| **B. Concepts & notes** | Write `docs/notes/stage-N.md` explaining every core concept in your own words | Mon |
| **C. Build** | The artefacts that must exist in the repo | Tue–Thu |
| **D. Exercises** | The numbered hands-on exercises | Tue–Thu |
| **E. Break-it drills** | Deliberately break things and fix them | Fri |
| **F. Interview & close-out** | Answer questions aloud, tick the claim checklist, update README, commit | Sat |

---

<a id="project"></a>

## 🎯 The project (keep it deliberately simple)

| Item | Choice | Why |
|---|---|---|
| Task | Classify 64×64 satellite image patches into 10 land-cover classes | Simple, well understood, but genuinely "satellite" so the edge/onboard story is real |
| Dataset | EuroSAT RGB (27,000 Sentinel-2 patches, ~90 MB) | Small enough to train on a laptop CPU/Apple GPU in minutes |
| Model | ResNet-18 (torchvision) fine-tuned, later a tiny MobileNetV3 for edge | Trains in minutes; exports cleanly to ONNX and quantises well for CPU |
| Framework | PyTorch | Industry default; clean ONNX export |
| Serving | FastAPI + ONNX Runtime | Framework-agnostic, fast, runs identically in cloud and on a Raspberry Pi |

The ML is intentionally boring. All the learning is in the lifecycle around it.

### 🏗️ End-state architecture (what you will have built)

```
 dev laptop (Mac) ──git push──► GitHub ──Actions CI──► tests, lint, docker build ──► image registry (GHCR / ECR)
        │                                     │
        │ dvc push                            └─CD on tag──► Kubernetes cluster (kind locally → EKS on AWS)
        ▼                                                    ├── inference Deployment (FastAPI + ONNX)
   S3 bucket (data + models)                                 ├── Prometheus + Grafana (latency, throughput)
        ▲                                                    └── drift monitor (Evidently) + alerts
        │ artifacts
 training: Mac (CPU/MPS) or a small cloud VM ──► MLflow tracking server
        │
        └── export ONNX → INT8 ──► signed bundle in S3 ──► Raspberry Pi (ARM64, emulated on Mac) offline "onboard" service
```

> [!NOTE]
> **Hardware reality check (you are on a Mac)**
> - Stages 1–4, 6, 7 and all of 8 run entirely on your Mac (CPU or Apple MPS). No GPU and no physical edge device is needed anywhere in this plan.
> - Edge (stage 8) targets a Raspberry Pi but is built and tested under ARM64 emulation (Docker buildx + QEMU) on your Mac. If you later buy a Pi (~A$120) the same image flashes and runs unchanged.
> - Total cloud budget for the whole plan: roughly US$40–70 if you shut things down.

**Timeline:** 9 stages, ~10–12 weeks at 8–10 h/week. Faster if you already know Git/Linux.

---

<a id="setup"></a>

## 🧰 Before you start: prerequisites, setup, and how to study

> [!NOTE]
> **Assumed:** you can write Python comfortably, know what a CNN is, and have trained a model in a notebook before. Everything else is taught here.

### ✅ Your tasks — Stage 0 (setup)

**🧰 A. Setup**
- [ ] Install Homebrew tools: `git gh uv pyenv tmux htop jq yq tree wget` and `kind kubectl helm k6 terraform awscli trivy qemu`.
- [ ] Install Docker Desktop; enable "Use Rosetta"; set RAM to 8 GB.
- [ ] Install `azure-cli` (can wait until stage 5).
- [ ] `uv python install 3.12`; create `.venv`; install the package list below.
- [ ] Run the sanity line: `docker run --rm hello-world && kind version && kubectl version --client && terraform -version`.
- [ ] Create accounts: AWS (set a **US$50 budget alarm immediately**) and Azure (US$200 credits).
- [ ] Create `docs/notes/` folder in the repo (one `stage-N.md` file per stage will live here).
- [ ] Read "How to study each stage" and "Weekly rhythm" below and block the hours in your calendar.

### 💻 One-time Mac setup (do this on day 1)

```bash
# package manager + core tools
brew install git gh uv pyenv tmux htop jq yq tree wget
brew install --cask docker            # Docker Desktop (enable "Use Rosetta" + increase RAM to 8 GB in Settings)
brew install kind kubectl helm k6 terraform awscli trivy
brew install azure-cli
brew install qemu                     # needed for stage 8 ARM64 emulation

# python toolchain for the repo
uv python install 3.12
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch torchvision mlflow dvc[s3] hydra-core pytest ruff pre-commit fastapi uvicorn onnx onnxruntime pillow scikit-learn

# sanity
docker run --rm hello-world && kind version && kubectl version --client && terraform -version
```

Create free accounts now so they are ready when needed: GitHub (done), AWS (free tier, set a **US$50 budget alarm immediately**), Azure (US$200 credits).

### 🔄 How to study each stage (repeat every stage)

1. **Read the concepts list first** ("Core concepts you must be able to explain"). For each item write 2–4 sentences in your own words in `docs/notes/stage-N.md`. If you cannot, read the linked official doc until you can. This is the most important habit in the plan.
2. **Build the artefacts** in the "Build" list. Type the core pieces yourself; use an AI assistant to unblock, then rewrite the piece from memory once.
3. **Do the exercises**. They are designed to be slightly annoying, because that is where understanding forms.
4. **Break it** (see "Common pitfalls & break-it drills"). Interview questions are almost always "what happens when X goes wrong".
5. **Answer the interview questions out loud**, without notes, and record yourself once. Fix the gaps.
6. **Tick the checklist**, update the status table in the root `README.md`, and write a 10-line "what I learned" at the end of `docs/notes/stage-N.md`. Commit.

### 📆 Weekly rhythm (8–10 h)

| Day | Time | What |
|---|---|---|
| Mon | 2 h | Concepts + notes |
| Tue–Thu | 5–6 h | Build + exercises |
| Fri | 1 h | Break-it drills |
| Sat | 1 h | Interview questions aloud, checklist, README status, commit |

---

<a id="stage-1"></a>

## 🐧 Stage 1 — Linux + Git: the professional repo skeleton

| | |
|---|---|
| 📅 **When** | Week 1 |
| 🎯 **Skill claimed** | Git, Linux |
| ⏭️ **Next** | [Stage 2 — ML design, DVC & MLflow](#stage-2) |

### ✅ Your tasks — Stage 1

**🧰 A. Setup**
- [ ] Python 3.12 venv active; `ruff`, `pre-commit`, `pytest`, `dvc` installed via `uv`.
- [ ] Launch a free-tier `t3.micro` Ubuntu EC2 instance (key-pair SSH). This is your Linux lab for the whole stage.
- [ ] Confirm `main` on GitHub is protected: PRs required, no direct pushes.

**📝 B. Concepts & notes** → `docs/notes/stage-1.md`
- [ ] Linux: filesystem hierarchy, users/groups/permissions/sudo, processes & signals (SIGTERM vs SIGKILL), stdin/stdout/stderr & redirection, exit codes, env vs shell variables, `PATH`, symlinks, daemons, systemd units & `journalctl`, SSH keys & `~/.ssh/config`, `apt`, `df`/`du`, `ip`/`ss`/`curl`/ports/`/etc/hosts`, cron syntax.
- [ ] Git: working tree / index / HEAD, commits as snapshots, branches as pointers, fast-forward vs merge vs rebase, detached HEAD, reflog, remotes & tracking branches, lightweight vs annotated tags, `.gitignore` semantics, why big binaries don't belong in git, what a PR is.
- [ ] Python packaging: `pyproject.toml`, editable installs, lockfiles, virtual envs, entry points.

**🔨 C. Build** (on a feature branch, merged via a self-reviewed PR)
- [ ] Folder skeleton: `.github/workflows/`, `configs/`, `src/orbiteye/{data,model,train,evaluate,export}.py`, `serving/`, `deploy/`, `infra/`, `edge/`, `tests/`.
- [ ] `pyproject.toml` (project metadata, deps, ruff + pytest config) and a `uv.lock`.
- [ ] `Makefile` with at least `setup`, `lint`, `test` targets.
- [ ] `.pre-commit-config.yaml`: `ruff`, `ruff-format`, `end-of-file-fixer`, `check-added-large-files`, plus a local `pytest` hook.
- [ ] Extend `.gitignore` for ML: `data/`, `mlruns/`, `*.pt`, `*.ckpt`, `*.onnx`, `outputs/`.
- [ ] One trivial test in `tests/` so `pytest` and the pre-commit hook are green.
- [ ] Every commit uses Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`…).

**🧪 D. Exercises**
- [ ] Ex 1 (EC2): user `orbiteye`, passwordless `sudo` for one command only, SSH key-only auth, verify with `sshd -T`.
- [ ] Ex 2 (EC2): systemd unit running a Python script that logs the time every 10 s; enable on boot; `journalctl -u`; `Restart=on-failure`; `kill -9` → it comes back.
- [ ] Ex 3 (EC2): one-liner printing the 5 largest files under `/var` in MB.
- [ ] Ex 4 (Git): messy branch of 6 commits → interactive squash to 2 → recover the original 6 via `git reflog`. Do it in your own terminal.
- [ ] Ex 5 (Git): commit a 50 MB file, push, remove from history with `git filter-repo`, force-push.
- [ ] Ex 6: pre-commit installed and running on every commit (done in C, verify by committing a badly formatted file).

**💥 E. Break-it drills**
- [ ] Break `sshd_config` on purpose with a second session open; recover.
- [ ] Practise `git push --force-with-lease` and explain why not `--force`.
- [ ] Fill the EC2 disk with `fallocate` until things fail; find the culprit with `du`; clean up.

**🎤 F. Interview & close-out**
- [ ] Answer the 5 interview questions aloud without notes; record yourself once; fix gaps.
- [ ] Tick all 3 "You can claim it when" boxes at the end of this stage.
- [ ] Write the 10-line "what I learned" at the end of `docs/notes/stage-1.md`.
- [ ] Set Stage 1 to ✅ in the tracker above and in `README.md`; commit; **stop the EC2 instance**.

> [!NOTE]
> **Why first:** everything later (CI, Docker, K8s, cloud) is driven from a Linux shell and a Git history. A recruiter opening your repo judges you in 10 seconds by this skeleton.

### 📚 Learn — deeply
- Shell: `ssh`, `scp`/`rsync`, `tmux`, `htop`, `chmod`/`chown`, environment variables, `systemd` basics, `journalctl`, pipes, `grep`/`awk`/`sed`, cron. Do all of this on a real Linux box: start a free-tier `t3.micro` Ubuntu EC2 instance (this also plants the AWS seed for stage 5).
- Git: branching model (main + feature branches), rebase vs merge, `git bisect`, tags, conventional commits, pull requests with reviews (review your own PRs), `.gitignore` for ML (no data, no checkpoints), pre-commit hooks.
- Python project hygiene: `pyproject.toml`, `uv` or `pip-tools` lockfile, `ruff`, `pytest`, `Makefile`.

### 🔨 Build
```
orbiteye/
├── .github/workflows/        (empty for now)
├── configs/                  (yaml experiment configs)
├── src/orbiteye/
│   ├── data.py               (dataset + transforms)
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── export.py
├── serving/                  (stage 6)
├── deploy/                   (k8s manifests, stage 6)
├── infra/                    (terraform, stage 5)
├── edge/                     (stage 8)
├── tests/
├── Makefile
├── pyproject.toml
├── .pre-commit-config.yaml
└── README.md
```

### 🧠 Core concepts you must be able to explain
- Linux: filesystem hierarchy (`/etc`, `/var`, `/opt`, `/proc`), users/groups/permissions and `sudo`, processes and signals (`SIGTERM` vs `SIGKILL`), stdin/stdout/stderr and redirection, exit codes, environment variables vs shell variables, `PATH`, symlinks, what a daemon is, `systemd` units and `journalctl`, SSH keys and `~/.ssh/config`, package managers (`apt`), disk usage (`df`, `du`), networking basics (`ip`, `ss`, `curl`, ports, `/etc/hosts`), cron syntax.
- Git: the three areas (working tree, index, HEAD), commits as snapshots, branches as pointers, fast-forward vs merge commit vs rebase, detached HEAD, `reflog`, remotes and tracking branches, tags (lightweight vs annotated), `.gitignore` semantics, why large binaries do not belong in git, what a PR actually is.
- Python packaging: `pyproject.toml`, editable installs, lockfiles and why reproducibility needs them, virtual environments, entry points.

### 🧪 Hands-on exercises
1. On the EC2 box, create a user `orbiteye`, give it passwordless `sudo` for one command only, and lock SSH to key-only auth. Verify with `sshd -T`.
2. Write a `systemd` unit that runs a Python script logging the time every 10 s; make it start on boot; read its logs with `journalctl -u`; make it restart on failure; kill it with `kill -9` and watch it come back.
3. Write a one-line pipeline that finds the 5 largest files under `/var` and prints them in MB.
4. Git: create a messy branch with 6 commits, interactively squash into 2 clean commits, then use `git reflog` to recover the original 6 after "losing" them. (Interactive rebase must be done in your own terminal, not through an AI tool.)
5. Deliberately commit a 50 MB file, push, then remove it from history with `git filter-repo` and force-push. Now you know why data goes in DVC.
6. Set up `pre-commit` with `ruff`, `ruff-format`, `end-of-file-fixer`, `check-added-large-files`.

### 💥 Common pitfalls & break-it drills
- Locking yourself out of SSH by breaking `sshd_config`: always test in a second session before closing the first.
- `git push --force` on a shared branch: learn `--force-with-lease`.
- Running things as root inside containers/VMs "because it works".
- Drill: fill the disk on the VM with `fallocate` until things break; find the culprit with `du`; clean up.

### 🎤 Interview questions
- What is the difference between `git merge` and `git rebase`, and when would you forbid rebase?
- A process ignores `Ctrl-C`. What do you do, step by step?
- How does SSH key authentication work? What is in `authorized_keys`?
- What happens at boot on a `systemd` Linux system? How do you make a service start on boot?
- Why does a lockfile matter if `requirements.txt` already pins versions?

<details>
<summary><b>📖 Resources</b> — click to expand</summary>

- *The Linux Command Line*, W. Shotts — free: https://linuxcommand.org/tlcl.php (ch. 1–10, 14–17, 24–26)
- MIT "The Missing Semester": https://missing.csail.mit.edu/ (shell, editors, data wrangling, command-line env, git)
- *Pro Git* book, free: https://git-scm.com/book/en/v2 (ch. 1–3, 5, 7.6 rewriting history)
- systemd for administrators: https://www.freedesktop.org/wiki/Software/systemd/ and `man systemd.service`
- Conventional Commits: https://www.conventionalcommits.org/
- `uv` docs: https://docs.astral.sh/uv/  ·  pre-commit: https://pre-commit.com/

</details>

### 🏁 You can claim it when
- [ ] Repo on GitHub, protected `main`, every change via PR, ≥ 20 meaningful commits with conventional messages.
- [ ] Pre-commit runs `ruff` + `pytest` locally.
- [ ] You can SSH into a Linux VM, set up Python, run a background job under `tmux`, inspect it with `htop`, and schedule a cron job, without looking anything up.

<p align="right"><a href="#-master-task-tracker-start-here">⬆ Back to tracker</a></p>

---

<a id="stage-2"></a>

## 🧬 Stage 2 — ML design, data versioning and experiment tracking

| | |
|---|---|
| 📅 **When** | Week 2–3 |
| 🎯 **Skill claimed** | "design" half of "design and deploy end-to-end ML models"; data/model versioning |
| ⏭️ **Next** | [Stage 3 — Docker](#stage-3) |

### ✅ Your tasks — Stage 2

**🧰 A. Setup**
- [ ] `dvc`, `mlflow`, `hydra-core`, `torch`, `torchvision`, `scikit-learn` in the venv.
- [ ] Download EuroSAT RGB (~90 MB) into `data/raw/` (git-ignored).
- [ ] `docs/notes/stage-2.md` created.

**📝 B. Concepts & notes** → `docs/notes/stage-2.md`
- [ ] Problem framing: business vs model vs proxy metric; macro-F1 vs accuracy; offline vs online eval; frozen evaluation set.
- [ ] Data: leakage (spatial leakage in satellite patches), stratified splits, data cards, dataset vs code versioning, content-addressable storage.
- [ ] DVC: `.dvc` files, `dvc.yaml` deps/outs/params/metrics, cache, remotes, `dvc repro` graph, `dvc exp`.
- [ ] MLflow: runs/experiments, tracking server vs backend store vs artefact store, registry & stage transitions, `pyfunc`, signatures.
- [ ] Training engineering: seeds/nondeterminism, config-as-code, checkpointing, early stopping, LR schedules, freeze vs fine-tune.
- [ ] Testing ML: shape/dtype, 2-batch overfit, invariance, data-contract, metric-vs-sklearn tests.

**🔨 C. Build**
- [ ] `docs/DESIGN.md` (Problem, Users, Metric, Data, Baseline, Risks incl. haze/seasonal/new-sensor, Evaluation plan, Failure modes) + data card + model card.
- [ ] `src/orbiteye/data.py`: EuroSAT dataset, transforms, stratified 70/15/15 split with a documented split rule.
- [ ] `src/orbiteye/model.py`: ResNet-18 fine-tune head.
- [ ] `src/orbiteye/train.py`: trains on Mac (CPU/MPS), logs params/metrics/artefacts to MLflow, saves `model.pt` + `metrics.json`; Hydra/YAML config in `configs/baseline.yaml`; seeded.
- [ ] `src/orbiteye/evaluate.py`: accuracy + macro-F1 + confusion matrix on the frozen test split.
- [ ] `dvc init`; `dvc add data/`; local-folder remote; `dvc.yaml` with `prepare → train → evaluate`; `params.yaml`.
- [ ] Baseline run reaching ~95 % accuracy; numbers into the README results table (from script output, not by hand).
- [ ] Six tests in `tests/`: shape, overfit, invariance, data-contract, metric-vs-sklearn, config load.

**🧪 D. Exercises**
- [ ] Ex 1: `DESIGN.md` written (see C).
- [ ] Ex 2: prove `dvc repro` skips unchanged stages; change one param → only `train`+`evaluate` rerun.
- [ ] Ex 3: MLflow sweep of 8 runs (LR × augmentation); parallel-coordinates plot; register best; promote to Production with a note.
- [ ] Ex 4: `infer_signature` added; load `models:/orbiteye/Production` back in a fresh process.
- [ ] Ex 5: make the overfit test fail on purpose (LR = 0), then restore.
- [ ] Ex 6: reproduce an old run from its commit hash only: `git checkout <sha> && dvc pull && dvc repro`.

**💥 E. Break-it drills**
- [ ] Delete `.dvc/cache` → `dvc pull` restores it.
- [ ] Corrupt one cached file → DVC detects it.
- [ ] Try `dvc pull` on a fresh clone *without* having pushed; understand the failure; then `dvc push`.
- [ ] Check EuroSAT filenames for adjacent-patch leakage; document the split rule in `DESIGN.md`.
- [ ] Understand `--default-artifact-root` by pointing MLflow at a path a container can't see.

**🎤 F. Interview & close-out**
- [ ] Answer the 5 interview questions aloud; record once.
- [ ] Tick all 3 "You can claim it when" boxes.
- [ ] 10-line "what I learned" in `docs/notes/stage-2.md`.
- [ ] Stage 2 ✅ in tracker + `README.md`; commit via PR.

### 📚 Learn
- Problem framing document: business goal → ML metric (macro-F1) → offline/online evaluation plan → failure modes (cloud cover, seasonal shift → this becomes your drift story in stage 7).
- Data versioning with **DVC**: `dvc init`, `dvc add data/`, remote = local folder now, S3 in stage 5. Reproducible pipeline `dvc.yaml` with stages `prepare → train → evaluate`.
- Experiment tracking with **MLflow**: params, metrics, artefacts, model registry (Staging → Production). Run the MLflow server locally first.
- Config management with **Hydra** or plain YAML + dataclasses. Seeds and determinism.
- Test your ML code: shape tests, a 2-batch overfit test, a data-contract test (class names, image size).

### 🔨 Build
- `train.py` that trains ResNet-18 on EuroSAT on your Mac (CPU/MPS, ~10 min for a few epochs), logs to MLflow, saves `model.pt` + `metrics.json`.
- `dvc repro` reproduces the whole pipeline from raw data.
- A baseline reaching ~95% accuracy (EuroSAT is easy; that's fine).

### 🧠 Core concepts you must be able to explain
- Problem framing: business metric vs model metric vs proxy metric; why macro-F1 over accuracy for imbalanced classes; offline vs online evaluation; what a "frozen evaluation set" is and why it must never be touched.
- Data: train/val/test leakage (especially spatial leakage in satellite imagery: neighbouring patches), stratified splits, data cards, dataset versioning vs code versioning, content-addressable storage (how DVC stores files by hash).
- DVC: `.dvc` files, `dvc.yaml` stages with `deps`/`outs`/`params`/`metrics`, the cache, remotes, `dvc repro` dependency graph, `dvc exp`.
- MLflow: runs, experiments, params/metrics/artefacts, the tracking server vs backend store vs artefact store, the model registry and stage transitions, `mlflow.pyfunc`, model signatures.
- Training engineering: seeds and nondeterminism (cuDNN, dataloader workers), config-as-code, checkpointing, early stopping, learning-rate schedules, transfer learning (freeze vs fine-tune).
- Testing ML code: shape/dtype tests, a 2-batch overfit test (loss → ~0), invariance tests (flip augmentation shouldn't change class), data-contract tests, evaluation-metric tests against sklearn.

### 🧪 Hands-on exercises
1. Write `DESIGN.md`: one page, sections *Problem, Users, Metric, Data, Baseline, Risks, Evaluation plan, Failure modes*. Explicitly list "haze / seasonal shift / new sensor" as risks (used in stage 7).
2. Build the DVC pipeline with three stages and prove `dvc repro` skips unchanged stages; change one hyper-parameter in `params.yaml` and show only `train`+`evaluate` rerun.
3. Run an MLflow sweep of 8 runs over LR × augmentation; make a parallel-coordinates plot in the UI; register the best model and promote it to Production with a note.
4. Add `mlflow.models.infer_signature` and load the registered model back with `mlflow.pyfunc.load_model("models:/orbiteye/Production")` in a fresh process.
5. Write the six tests listed above; make the overfit test fail on purpose by zeroing the learning rate.
6. Reproduce a run from two weeks ago (or from another branch) using only its commit hash.

### 💥 Common pitfalls & break-it drills
- Committing `data/` to git; forgetting `dvc push`; a teammate (future you) doing `dvc pull` and getting nothing.
- Random split that puts adjacent patches in train and test → inflated accuracy. Check EuroSAT filenames and document your split rule.
- MLflow artefacts written to a local path that a container cannot see. Understand `--default-artifact-root`.
- Drill: delete `.dvc/cache`, then `dvc pull`; corrupt one file in the cache and see DVC detect it.

### 🎤 Interview questions
- How do you version a 50 GB dataset alongside code? What does DVC store in git?
- Your model's offline F1 is 0.95 but users complain. List five reasons.
- What is the difference between MLflow's tracking server, backend store and artefact store?
- How do you make a training run reproducible? What can you not control?
- Why is accuracy a bad metric here, and what would you monitor in production instead?

<details>
<summary><b>📖 Resources</b> — click to expand</summary>

- DVC Get Started (do all of it): https://dvc.org/doc/start  ·  DVC pipelines: https://dvc.org/doc/user-guide/pipelines
- MLflow Tracking: https://mlflow.org/docs/latest/tracking.html  ·  Model Registry: https://mlflow.org/docs/latest/model-registry.html
- Hydra tutorial: https://hydra.cc/docs/tutorials/intro/
- *Designing Machine Learning Systems*, C. Huyen — ch. 2 (framing), 4 (data), 6 (evaluation)
- Made With ML (MLOps course, free): https://madewithml.com/
- Google "Rules of ML": https://developers.google.com/machine-learning/guides/rules-of-ml
- Model cards paper: https://arxiv.org/abs/1810.03993
- PyTorch reproducibility: https://pytorch.org/docs/stable/notes/randomness.html

</details>

### 🏁 You can claim it when
- [ ] `git checkout <commit> && dvc pull && dvc repro` rebuilds any past result.
- [ ] MLflow UI shows ≥ 10 tracked runs and a registered model with a version in "Production".
- [ ] `DESIGN.md` in the repo with framing, metrics, data card, model card.

<p align="right"><a href="#-master-task-tracker-start-here">⬆ Back to tracker</a></p>

---

<a id="stage-3"></a>

## 🐳 Stage 3 — Docker

| | |
|---|---|
| 📅 **When** | Week 4 |
| 🎯 **Skill claimed** | Docker |
| ⏭️ **Next** | [Stage 4 — CI/CD](#stage-4) |

### ✅ Your tasks — Stage 3

**🧰 A. Setup**
- [ ] Docker Desktop running; `trivy` installed; `docker buildx ls` shows a multi-platform builder.
- [ ] `docs/notes/stage-3.md` created.

**📝 B. Concepts & notes** → `docs/notes/stage-3.md`
- [ ] Container = namespaces + cgroups + union FS; container vs VM.
- [ ] Layers, build cache & invalidation, `COPY` order, `.dockerignore`, multi-stage, `ARG` vs `ENV`, `ENTRYPOINT` vs `CMD` (exec vs shell form, PID 1, signals), `USER`, `HEALTHCHECK`, `WORKDIR`.
- [ ] Volumes vs bind mounts vs tmpfs; data lifetime.
- [ ] Bridge networking, port publishing, compose DNS.
- [ ] Registries, tags vs digests, why `latest` is dangerous, scanning/provenance.
- [ ] Docker Desktop on Mac = Linux VM; `--platform` implications.
- [ ] Compose: services, `depends_on` + healthchecks, `.env`, profiles.
- [ ] Read-only: what `nvidia/cuda` images, NVIDIA Container Toolkit and `--gpus all` are.

**🔨 C. Build**
- [ ] `docker/Dockerfile.train`: multi-stage, CPU PyTorch wheels, non-root, pinned base digest, lockfile installed.
- [ ] `docker/Dockerfile.serve`: slim CPU base, < 1 GB, non-root (used again in stage 6).
- [ ] `.dockerignore` excluding `data/`, `.venv/`, `mlruns/`, `.git/`.
- [ ] `docker-compose.yml`: MLflow server + Postgres + MinIO with healthchecks.
- [ ] `make train-docker` runs training in the container with `data/` mounted as a volume.
- [ ] `docker/README.md` with the image-size table and layer-ordering explanation.

**🧪 D. Exercises**
- [ ] Ex 1: build the serving image 3 ways (`python:3.12` single-stage / `slim` multi-stage / `slim` + `uv` no cache); record size + build time in `docker/README.md`.
- [ ] Ex 2: change one Python line, rebuild; reorder until only the last layer rebuilds.
- [ ] Ex 3: `Ctrl-C` test; fix with exec-form `ENTRYPOINT` or `tini`; explain.
- [ ] Ex 4: run serving container non-root with `--read-only --tmpfs /tmp`.
- [ ] Ex 5: `docker compose up` the tracking stack; point `train.py` at it; verify artefacts in MinIO.
- [ ] Ex 6: `trivy image`; fix ≥ 1 finding by bumping base; pin base by digest.
- [ ] Ex 7: `buildx` multi-arch (`linux/amd64,linux/arm64`); inspect the manifest list.

**💥 E. Break-it drills**
- [ ] Copy the whole repo into the context once; watch context size/build time; fix with `.dockerignore`.
- [ ] Install CUDA torch wheels into the CPU image once; note the +2 GB; switch to CPU index URL.
- [ ] Bake a fake secret via `ENV`; find it with `docker history`; move to build secrets / runtime env.
- [ ] `docker system df` → `docker system prune`; explain what was safe.

**🎤 F. Interview & close-out**
- [ ] Answer the 5 interview questions aloud; record once.
- [ ] Tick all 3 "You can claim it when" boxes.
- [ ] 10-line "what I learned" in `docs/notes/stage-3.md`.
- [ ] Stage 3 ✅ in tracker + `README.md`; commit via PR.

### 📚 Learn — deeply
- Images vs containers, layers and caching, multi-stage builds, `.dockerignore`, non-root users, `ENTRYPOINT` vs `CMD`, volumes vs bind mounts, networks, `docker compose`, image tagging strategy (git SHA + semver), registry push/pull, image scanning (`trivy`), size reduction (a 6 GB "works on my machine" image vs a 900 MB one).
- GPU containers (read only, no hardware needed): know what `nvidia/cuda` base images and the NVIDIA Container Toolkit are and how `--gpus all` works, so you can answer the interview question. You will not need them in this plan.
- Reproducibility: pinned base image digest, lockfile installed inside the image.

### 🔨 Build
- `docker/Dockerfile.train` (multi-stage, CPU PyTorch) and `docker/Dockerfile.serve` (slim CPU base, for stage 6).
- `docker-compose.yml` that starts MLflow server + Postgres + MinIO (S3-compatible) locally so your tracking stack is containerised too.
- `make train-docker` runs training inside the container with data mounted as a volume.

### 🧠 Core concepts you must be able to explain
- What a container actually is (namespaces + cgroups + a union filesystem), and how that differs from a VM.
- Image layers, the build cache and cache invalidation rules, `COPY` ordering, `.dockerignore`, multi-stage builds, `ARG` vs `ENV`, `ENTRYPOINT` vs `CMD` (exec form vs shell form, and PID 1 / signal handling), `USER`, `HEALTHCHECK`, `WORKDIR`.
- Storage: volumes vs bind mounts vs tmpfs; what happens to data when a container is removed.
- Networking: bridge network, port publishing, container DNS in `docker compose`.
- Registries, tags vs digests, why `latest` is dangerous, image provenance and scanning.
- Docker Desktop on Mac runs a Linux VM; implications for performance, file sharing and `--platform`.
- `docker compose`: services, depends_on + healthchecks, `.env`, profiles.

### 🧪 Hands-on exercises
1. Build the serving image three ways and record size + build time: (a) `python:3.12` single stage, (b) `python:3.12-slim` multi-stage, (c) same with `uv` and no cache dirs. Put the table in `docker/README.md`.
2. Change one line of Python and rebuild: which layers rebuild? Reorder the Dockerfile until only the last layer does.
3. Run the container with `docker run` and press `Ctrl-C`: does it stop instantly? Fix it with exec-form `ENTRYPOINT` (or `tini`), and explain why.
4. Run the serving container as a non-root user with a read-only root filesystem (`--read-only --tmpfs /tmp`).
5. `docker compose up` the MLflow + Postgres + MinIO stack, point `train.py` at it, and verify artefacts land in MinIO's bucket.
6. Scan with `trivy image`, fix at least one finding by bumping a base image, and pin the base image by digest.
7. Build a multi-arch image (`linux/amd64,linux/arm64`) with `buildx` and inspect the manifest list. (Preview of stage 8.)

### 💥 Common pitfalls & break-it drills
- Copying the whole repo (including `data/` and `.venv/`) into the build context: 5 GB context, 10-minute builds.
- Installing PyTorch with CUDA wheels into a CPU serving image: +2 GB for nothing. Use the CPU index URL.
- Secrets baked into layers via `ENV` or `COPY .env`. Use build secrets or runtime env.
- Drill: `docker system df`, then `docker system prune`; explain what was safe to delete.

### 🎤 Interview questions
- Walk me through what happens when you run `docker run -p 8000:8000 image`.
- Why is your image 900 MB and not 3 GB? What would you do to get it under 500 MB?
- What is the difference between `CMD` and `ENTRYPOINT`? What is PID 1 and why does it matter?
- How do you get a secret into a container securely at build time and at run time?
- Why would you pin by digest rather than tag?

<details>
<summary><b>📖 Resources</b> — click to expand</summary>

- Docker docs "Get started" + "Build with Docker" guide: https://docs.docker.com/get-started/ and https://docs.docker.com/build/
- Dockerfile best practices: https://docs.docker.com/build/building/best-practices/
- Multi-stage builds: https://docs.docker.com/build/building/multi-stage/  ·  buildx multi-platform: https://docs.docker.com/build/building/multi-platform/
- *Docker Deep Dive*, N. Poulton (any recent edition)
- Trivy: https://aquasecurity.github.io/trivy/
- Julia Evans, "How containers work" zine (excellent on namespaces/cgroups): https://wizardzines.com/zines/containers/
- PyTorch CPU wheels: https://pytorch.org/get-started/locally/

</details>

### 🏁 You can claim it when
- [ ] Serving image < 1 GB, trains/serves from a clean machine with only Docker installed.
- [ ] You can explain (and have written in the README) why each layer is ordered as it is and what the cache hit rate is on a code-only change.
- [ ] `trivy` scan has no critical CVEs.

<p align="right"><a href="#-master-task-tracker-start-here">⬆ Back to tracker</a></p>

---

<a id="stage-4"></a>

## 🔁 Stage 4 — CI/CD

| | |
|---|---|
| 📅 **When** | Week 5 |
| 🎯 **Skill claimed** | CI/CD |
| ⏭️ **Next** | [Stage 5 — Cloud: AWS & Azure](#stage-5) |

### ✅ Your tasks — Stage 4

**🧰 A. Setup**
- [ ] GHCR write permission for `GITHUB_TOKEN` understood (`packages: write`).
- [ ] Frozen tiny eval subset committed to DVC and pushed to the remote.
- [ ] `docs/notes/stage-4.md` created.

**📝 B. Concepts & notes** → `docs/notes/stage-4.md`
- [ ] CI vs CD vs continuous deployment; pipeline as quality gates; shift left.
- [ ] GitHub Actions model: workflows, triggers, jobs, runners, steps, `needs`, matrices, `concurrency`, caching, artefacts, secrets vs variables, environments/required reviewers, `GITHUB_TOKEN` permissions, reusable workflows, composite actions.
- [ ] ML gates: data contracts, smoke training, quality gate vs frozen set, metric diff vs `main`, model card generation.
- [ ] Container CI: `build-push-action`, `type=gha` cache, semver from tags, OCI labels.
- [ ] Continuous training: scheduled vs triggered retraining; why humans approve promotion.
- [ ] Security: least privilege, pin actions by SHA, OIDC (stage 5), Dependabot; `pull_request_target` risks.

**🔨 C. Build**
- [ ] `.github/workflows/ci.yml` (on PR): `lint → test → smoke-train → quality-gate → build`.
- [ ] `.github/workflows/release.yml` (on `v*.*.*` tag): multi-arch build, tag version + SHA, push to GHCR, GitHub Release with `metrics.json`.
- [ ] `.github/workflows/nightly-train.yml` (`schedule` + `workflow_dispatch`): retrain, log to MLflow, open PR if better than Production.
- [ ] PR comment step posting metrics + confusion matrix (CML or `gh api` script).
- [ ] Branch protection: required checks, no direct pushes, linear history.
- [ ] Dependabot config for actions + pip; every third-party action pinned to a commit SHA.
- [ ] Enable the CI / Release badges in `README.md`.

**🧪 D. Exercises**
- [ ] Ex 1: `ci.yml` quality gate downloads frozen eval set from DVC remote, evaluates smoke model, fails under threshold; uses `needs:` + job outputs.
- [ ] Ex 2: add `uv`/pip + Docker layer caching; record pipeline time before/after in notes.
- [ ] Ex 3: PR comment with metrics + confusion matrix image.
- [ ] Ex 4: `release.yml` produces a versioned image + Release with zero manual steps.
- [ ] Ex 5: `nightly-train.yml` runs and opens a PR.
- [ ] Ex 6: try pushing directly to `main`; get rejected.
- [ ] Ex 7: SHA-pin all actions; Dependabot enabled.

**💥 E. Break-it drills**
- [ ] Open a PR with a deliberately bad config; watch the quality gate block it; fix; merge. **Keep the link** (Appendix A evidence).
- [ ] Make a test flaky via nondeterminism; fix with seeds + tiny fixed subset + range asserts.
- [ ] Keep PR CI under 10 min; push heavy work to nightly.

**🎤 F. Interview & close-out**
- [ ] Answer the 5 interview questions aloud; record once.
- [ ] Tick all 3 "You can claim it when" boxes.
- [ ] 10-line "what I learned" in `docs/notes/stage-4.md`.
- [ ] Stage 4 ✅ in tracker + `README.md`; commit via PR.

### 📚 Learn
- GitHub Actions: triggers, jobs, matrices, caching (`pip`, Docker layers via `buildx`), secrets, environments with manual approval, reusable workflows, artefacts.
- The ML-specific CI pyramid: lint → unit tests → data-contract tests → 1-epoch smoke training on a tiny subset → model quality gate (fail if macro-F1 < threshold on a fixed eval set) → build & push image.
- CD: on a git tag `v*`, build the serving image, push to GHCR (and to ECR in stage 5), and deploy to the cluster (stage 6 wires the last step).
- Branch protection with required checks.

### 🔨 Build
- `.github/workflows/ci.yml` (on PR), `release.yml` (on tag), `nightly-train.yml` (scheduled small retrain + MLflow log; this feeds "continuous training").
- A `CML`-style PR comment that posts the confusion matrix and metrics diff versus `main`.

### 🧠 Core concepts you must be able to explain
- CI vs CD vs continuous deployment; the deployment pipeline as a series of quality gates; "shift left".
- GitHub Actions model: workflows, events/triggers (`push`, `pull_request`, `schedule`, `workflow_dispatch`, `tags`), jobs, runners, steps, actions vs `run`, `needs`, matrices, `concurrency`, caching, artefacts, secrets vs variables, environments and required reviewers, permissions (`GITHUB_TOKEN` scopes), reusable workflows and composite actions.
- ML-specific gates: data contracts, smoke training, quality gate against a frozen eval set, comparing metrics to `main`, model card generation.
- Container CI: `docker/build-push-action`, layer caching with `cache-from/to: type=gha`, semantic versioning from tags, image labels (OCI annotations) linking image → commit.
- Continuous training: scheduled retraining, triggered retraining (from monitoring in stage 7), and why humans still approve promotion.
- Security: least-privilege tokens, pinning actions by SHA, OIDC to cloud (stage 5), Dependabot.

### 🧪 Hands-on exercises
1. Write `ci.yml` with jobs `lint → test → smoke-train → quality-gate → build`. Make `quality-gate` download the frozen eval set from the DVC remote, evaluate the model trained in `smoke-train` on a tiny subset, and fail below a threshold you set. Use `needs:` and job outputs.
2. Cache `uv`/pip and Docker layers; measure pipeline time before/after and write it down.
3. Add a PR comment step that posts metrics + confusion matrix image (use `cml` or a 20-line script with `gh api`).
4. Write `release.yml` triggered on tags `v*.*.*`: build multi-arch image, tag with version + SHA, push to GHCR, create a GitHub Release with the metrics JSON attached.
5. Write `nightly-train.yml` on `schedule` + `workflow_dispatch` that retrains on full data (CPU, few epochs), logs to MLflow, and opens a PR if the new model beats Production.
6. Turn on branch protection: required checks, no direct pushes, linear history. Try to push to `main` directly and get rejected.
7. Pin every third-party action to a commit SHA; enable Dependabot for actions and pip.

### 💥 Common pitfalls & break-it drills
- `pull_request` from forks cannot see secrets; understand `pull_request_target` risks.
- Flaky tests caused by nondeterministic training: seed, use tiny fixed subsets, assert ranges not exact values.
- A 40-minute pipeline nobody waits for; keep PR CI under 10 minutes, push heavy work to nightly.
- Drill: intentionally break the quality gate with a bad config PR; watch it block; fix; merge.

### 🎤 Interview questions
- What does your CI pipeline do for an ML PR that it wouldn't for a normal Python PR?
- How do you stop a model that is worse than the current one from being deployed?
- How do you pass an artefact between jobs? Between workflows?
- How would you roll back a release produced by your pipeline?
- Why pin actions by SHA?

<details>
<summary><b>📖 Resources</b> — click to expand</summary>

- GitHub Actions docs (read "Writing workflows" fully): https://docs.github.com/en/actions
- Workflow syntax reference: https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions
- Security hardening for Actions: https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions
- docker/build-push-action: https://github.com/docker/build-push-action  ·  GHA cache: https://docs.docker.com/build/cache/backends/gha/
- CML (ML in CI): https://cml.dev/doc
- Google "MLOps: Continuous delivery and automation pipelines in ML": https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
- *Continuous Delivery*, Humble & Farley — ch. 5 (deployment pipeline)

</details>

### 🏁 You can claim it when
- [ ] A PR that lowers accuracy below threshold is automatically blocked.
- [ ] Pushing a tag produces a versioned image in a registry with zero manual steps.
- [ ] Green badge in README; you've debugged at least one flaky pipeline.

<p align="right"><a href="#-master-task-tracker-start-here">⬆ Back to tracker</a></p>

---

<a id="stage-5"></a>

## ☁️ Stage 5 — Cloud: AWS in depth, Azure by mapping

| | |
|---|---|
| 📅 **When** | Week 6–7 |
| 🎯 **Skill claimed** | AWS, Azure |
| ⏭️ **Next** | [Stage 6 — Serving & Kubernetes](#stage-6) |

### ✅ Your tasks — Stage 5

**🧰 A. Setup**
- [ ] `awscli`, `terraform`, `az` installed and authenticated.
- [ ] AWS: MFA on root, admin IAM user, US$50 budget alert. Never use root again.
- [ ] Azure free-credit account active with a budget alert.
- [ ] Read Appendix B (cost checklist) and keep it open during every cloud session.
- [ ] `docs/notes/stage-5.md` created.

**📝 B. Concepts & notes** → `docs/notes/stage-5.md`
- [ ] Shared responsibility; regions/AZs; IaaS/PaaS/SaaS placement of EC2, EKS, Azure Container Apps, SageMaker.
- [ ] IAM: principals, identity vs resource policies, roles & trust policies, `AssumeRole`, instance profiles, least privilege, OIDC federation.
- [ ] Networking: VPC, public/private subnets, route tables, IGW, NAT, SGs vs NACLs, SSM Session Manager instead of port 22.
- [ ] S3: prefixes, storage classes, lifecycle, versioning, presigned URLs, consistency, cost.
- [ ] Compute: instance families, spot vs on-demand, AMIs, user data, EBS vs instance store.
- [ ] CloudWatch, Cost Explorer, Budgets, tagging.
- [ ] Terraform: providers, resources, data sources, vars/outputs, state & remote state with locking, plan vs apply, modules, workspaces, import, drift.
- [ ] Fill in the AWS/Azure mapping table with one difference per row.

**🔨 C. Build**
- [ ] `infra/aws/`: S3 (versioning + lifecycle), ECR (scan-on-push, keep 10), OIDC provider + role scoped to `repo:alichr/orbiteye:*`, remote state bucket + DynamoDB lock, `c6i.xlarge` in private subnet via SSM with instance profile.
- [ ] `infra/azure/`: ACR, Container Apps, Azure ML command job.
- [ ] DVC remote switched to S3; `ci.yml` assumes OIDC role, pulls eval set from S3, pushes image to ECR. No stored cloud secrets.
- [ ] MLflow server on EC2 (compose: Postgres + S3 artefact store), reached via SSM port-forward.
- [ ] Screenshots of live URLs on both clouds in `docs/assets/`; then everything destroyed.

**🧪 D. Exercises**
- [ ] Ex 1: account hardening + budget (see A).
- [ ] Ex 2: Terraform (a) S3 → (b) ECR → (c) OIDC role → (d) remote state → (e) EC2, one `apply` each.
- [ ] Ex 3: `dvc push` to S3; delete local cache; `dvc pull`.
- [ ] Ex 4: `ci.yml` on OIDC (`aws-actions/configure-aws-credentials`).
- [ ] Ex 5: remote MLflow on EC2; local training logs to it.
- [ ] Ex 6: one full training on EC2 via `user_data`; run appears in MLflow; `terraform destroy` the instance.
- [ ] Ex 7: Azure ACR + Container Apps + one Azure ML job; Terraform, screenshot, destroy.
- [ ] Ex 8: check Cost Explorer daily for two weeks; write down every line item.

**💥 E. Break-it drills**
- [ ] Change a security group in the console; `terraform plan` shows drift; revert.
- [ ] Point the OIDC trust policy at the wrong repo; CI fails clearly; fix.
- [ ] Audit for orphaned NAT gateways / EBS / LBs / static IPs after every `destroy`.

**🎤 F. Interview & close-out**
- [ ] Answer the 6 interview questions aloud; record once.
- [ ] Tick all 3 "You can claim it when" boxes.
- [ ] 10-line "what I learned" in `docs/notes/stage-5.md`.
- [ ] Stage 5 ✅ in tracker + `README.md`; commit via PR. **Confirm the bill.**

> [!TIP]
> **Honest framing for the CV:** you will go deep on **one** cloud (AWS) and do a thin but real deployment on Azure. In interviews say exactly that: "primary AWS, working knowledge of Azure equivalents." That is what most candidates who list both actually mean, and you'll be able to back it up.

### 📚 Learn — AWS, deeply
- IAM (users, roles, policies, least privilege, instance profiles), VPC basics, S3 (buckets, lifecycle, presigned URLs), EC2 (AMIs, spot, security groups, key pairs), ECR, EKS (stage 6), CloudWatch, cost explorer and budgets + alerts (set a US$50 alarm on day one).
- Infrastructure as code with **Terraform**: S3 bucket for DVC + MLflow artefacts, ECR repo, an IAM role for GitHub Actions via OIDC (no long-lived keys), a small EC2 instance for training and MLflow.
- Move MLflow server to a small EC2 with RDS Postgres backend (or keep in docker compose on EC2; both fine).

### 📚 Learn — Azure, mapping

| Concept | AWS | Azure |
|---|---|---|
| Object storage | S3 | Blob Storage |
| Container registry | ECR | ACR |
| Managed K8s | EKS | AKS |
| Compute VM | EC2 | Virtual Machines |
| Managed ML | SageMaker | Azure ML |
| Identity | IAM | Entra ID / RBAC |
| Serverless containers | App Runner / ECS Fargate | Container Apps |
| IaC (native) | CloudFormation | Bicep / ARM |

On Azure, using free credits: push your serving image to ACR, deploy it to Azure Container Apps (serverless containers, cheapest way to get a real URL), and run one managed-training job (Azure ML command job) with your training container. Terraform all of it.

### 🔨 Build
- `infra/aws/*.tf`, `infra/azure/*.tf`.
- DVC remote = S3; CI uses OIDC role to pull data and push images to ECR.
- One training run on a CPU EC2 instance (e.g. `c6i.xlarge`) launched from your Terraform template, logged to the remote MLflow.

### 🧠 Core concepts you must be able to explain
- Shared-responsibility model; regions and availability zones; the three service tiers (IaaS/PaaS/SaaS) and where EC2, EKS, Azure Container Apps, SageMaker sit.
- IAM properly: principals, policies (identity vs resource), roles and trust policies, `AssumeRole`, instance profiles, least privilege, why access keys on a laptop are a smell, OIDC federation from GitHub.
- Networking basics: VPC, subnets (public/private), route tables, internet gateway, NAT, security groups vs NACLs, why your training VM should be in a private subnet with SSM Session Manager instead of an open port 22.
- S3: buckets, prefixes (not folders), storage classes, lifecycle rules, versioning, presigned URLs, consistency, cost per GB and per request.
- Compute: instance families, spot vs on-demand, AMIs, user data, EBS vs instance store.
- Observability & cost: CloudWatch metrics/logs/alarms, Cost Explorer, Budgets, tagging strategy for cost allocation.
- Terraform: providers, resources, data sources, variables/outputs, state (and remote state in S3 with locking), `plan` vs `apply`, modules, workspaces, import, drift, and what *not* to put in Terraform.
- Cloud mapping: for every AWS service you use, name the Azure equivalent and one difference.

### 🧪 Hands-on exercises
1. Create the AWS account, enable MFA on root, create an admin IAM user, set a US$50 budget with email alert. Never use root again.
2. Terraform, in this order, one `apply` at a time: (a) S3 bucket for DVC + MLflow with versioning + lifecycle rule; (b) ECR repo with scan-on-push and a lifecycle policy keeping 10 images; (c) IAM OIDC provider + role that GitHub Actions can assume, scoped to `repo:alichr/orbiteye:*`; (d) remote Terraform state bucket + DynamoDB lock; (e) a `c6i.xlarge` in a private subnet reachable via SSM, with an instance profile that can read the S3 bucket.
3. Switch DVC remote to S3; `dvc push`; delete local cache; `dvc pull`.
4. Update `ci.yml` to assume the OIDC role (`aws-actions/configure-aws-credentials`), pull the eval set from S3, push the image to ECR. No secrets stored in GitHub.
5. Run MLflow server on the EC2 instance (docker compose, Postgres + S3 artefact store); point local training at it over an SSM port-forward.
6. Run one full training on the EC2 instance from a `user_data` bootstrap script; confirm the run appears in MLflow; terminate the instance from Terraform.
7. Azure: `az` setup, ACR, deploy the serving image to Azure Container Apps, run one Azure ML command job with the training image. Terraform it, screenshot, destroy.
8. Look at Cost Explorer daily for two weeks and write down what each line item is.

### 💥 Common pitfalls & break-it drills
- Leaving an instance or a NAT gateway running: NAT gateways cost ~US$32/month even idle. Destroy when done.
- Public S3 bucket or `0.0.0.0/0` on port 22: understand why the scanners will find it in minutes.
- Terraform state committed to git or lost: use remote state from exercise 2d onward.
- Drill: manually change a security group in the console, run `terraform plan`, watch it detect drift, and revert.
- Drill: rotate the OIDC role's trust policy to the wrong repo and watch CI fail with a clear error; fix.

### 🎤 Interview questions
- How does GitHub Actions authenticate to AWS without a stored secret? Walk through the OIDC flow.
- What is the difference between an IAM role and an IAM user? When would you use each?
- Your training VM in a private subnet needs to `pip install`. What are the options and their costs?
- What is in Terraform state and why is it sensitive?
- Compare EKS and AKS in two sentences. Compare SageMaker and Azure ML.
- Your AWS bill doubled this month. How do you find out why?

<details>
<summary><b>📖 Resources</b> — click to expand</summary>

- AWS Skill Builder "Cloud Practitioner Essentials" (free): https://skillbuilder.aws/
- AWS IAM docs "How IAM works" + policy evaluation logic: https://docs.aws.amazon.com/IAM/latest/UserGuide/intro-structure.html
- GitHub OIDC with AWS: https://docs.github.com/en/actions/security-for-github-actions/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services
- Terraform "Get Started – AWS": https://developer.hashicorp.com/terraform/tutorials/aws-get-started  ·  Terraform AWS provider docs
- AWS Well-Architected Framework (skim the six pillars): https://aws.amazon.com/architecture/well-architected/
- Azure Container Apps quickstart: https://learn.microsoft.com/en-us/azure/container-apps/  ·  Azure ML command jobs: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-model
- AWS-to-Azure service comparison: https://learn.microsoft.com/en-us/azure/architecture/aws-professional/services

</details>

### 🏁 You can claim it when
- [ ] `terraform apply` / `destroy` recreates all AWS infra from scratch, nothing was clicked in the console.
- [ ] The serving container is reachable at a public URL on both clouds (screenshots in README, then torn down).
- [ ] You know your monthly bill to the dollar and have a budget alarm.

<p align="right"><a href="#-master-task-tracker-start-here">⬆ Back to tracker</a></p>

---

<a id="stage-6"></a>

## ☸️ Stage 6 — Model serving and Kubernetes

| | |
|---|---|
| 📅 **When** | Week 8–9 |
| 🎯 **Skill claimed** | Kubernetes, "deploy" half of end-to-end |
| ⏭️ **Next** | [Stage 7 — Monitoring](#stage-7) |

### ✅ Your tasks — Stage 6

**🧰 A. Setup**
- [ ] `kind`, `helm`, `k6` installed; `kind create cluster` works.
- [ ] `fastapi`, `uvicorn`, `onnxruntime`, `onnx` in the venv.
- [ ] `docs/notes/stage-6.md` created.

**📝 B. Concepts & notes** → `docs/notes/stage-6.md`
- [ ] Serving: sync/async/batch, latency vs throughput, dynamic batching, cold start, model-loading strategies, versioned endpoints, input validation, why ONNX Runtime, threads, warm-up.
- [ ] K8s architecture: control plane vs nodes; reconciliation loop; declarative state.
- [ ] Objects: Pod, Deployment → ReplicaSet → Pod, Service types & ClusterIP DNS, Ingress, ConfigMap/Secret, Namespace, ServiceAccount + RBAC, PV/PVC, Job/CronJob.
- [ ] Probes, requests/limits & QoS, rolling updates & rollbacks, HPA & metrics-server, Helm (chart, values, release, revisions).
- [ ] Failure states: `ImagePullBackOff`, `CrashLoopBackOff`, `Pending`, `OOMKilled`.

**🔨 C. Build**
- [ ] `src/orbiteye/export.py`: PyTorch → ONNX export (static shape `[1,3,64,64]`).
- [ ] `serving/app.py`: loads ONNX from S3/registry by version at startup, `/predict`, `/healthz`, `/readyz`, `/metrics`, warm-up; tests with `TestClient`.
- [ ] `serving/loadtest.js` (k6) and a load-test report with p50/p95/p99 at N replicas.
- [ ] `deploy/kind/`: cluster config + raw manifests (Deployment, Service, ConfigMap, HPA).
- [ ] `deploy/helm/orbiteye/`: chart with probes, resources, HPA, RBAC, values.
- [ ] `release.yml` runs `helm upgrade --install` on tag.
- [ ] `infra/aws/eks.tf` using `terraform-aws-modules/eks`, IRSA for S3, ALB Ingress.
- [ ] HPA scaling screenshot in `docs/assets/`; serving numbers in the README results table.

**🧪 D. Exercises**
- [ ] Ex 1: `serving/app.py` (see C).
- [ ] Ex 2: k6 at 10/50/200 VUs; record p50/p95/p99 + CPU; tune `intra_op_num_threads`; compare.
- [ ] Ex 3: 3-node kind cluster; raw manifests first, then Helm chart; keep both.
- [ ] Ex 4: memory limit too low → `OOMKilled`; CPU limit too low → latency; fix; explain.
- [ ] Ex 5: `/readyz` → 500; pod leaves Service endpoints but stays alive.
- [ ] Ex 6: roll out a bad tag; rollout stalls; `helm rollback`; describe both ReplicaSets.
- [ ] Ex 7: metrics-server + k6; `kubectl get hpa -w` scales 1 → 4 → 1.
- [ ] Ex 8: EKS via Terraform; chart with IRSA; ALB; same load test; screenshot; **destroy same day**.
- [ ] Ex 9: `kubectl` fluency drill without docs.

**💥 E. Break-it drills**
- [ ] `kubectl delete pod`; explain what recreated it. `kubectl drain` a node; watch pods move.
- [ ] Cause `ImagePullBackOff`, `CrashLoopBackOff`, `Pending`; diagnose each from `describe` + `logs` only.

**🎤 F. Interview & close-out**
- [ ] Answer the interview questions aloud; record once.
- [ ] Tick all 3 "You can claim it when" boxes.
- [ ] 10-line "what I learned" in `docs/notes/stage-6.md`.
- [ ] Stage 6 ✅ in tracker + `README.md`; commit via PR. **EKS destroyed.**

### 📚 Learn — deeply
- Serving: FastAPI app, ONNX Runtime session, input validation (Pydantic), batching, `/predict`, `/healthz`, `/readyz`, `/metrics`. Load test with `locust` or `k6`. Model loaded from S3/MLflow registry at startup by version, not baked into the image (understand the trade-off, bake it for edge).
- Kubernetes core: Pods, Deployments, ReplicaSets, Services (ClusterIP/NodePort/LoadBalancer), Ingress, ConfigMaps, Secrets, resource requests/limits, liveness/readiness probes, rolling updates and rollbacks, HPA (autoscale on CPU then on custom latency metric), namespaces, RBAC, `kubectl` fluency (`describe`, `logs`, `exec`, `port-forward`, `top`). Helm charts.
- Do it locally first with **kind** (you already have `kubectl`), then on **EKS** via Terraform. Optional but impressive: canary rollout with Argo Rollouts, or KServe as the "this is what managed model serving looks like" comparison.

### 🔨 Build
- `serving/app.py`, `deploy/helm/orbiteye/` chart, `deploy/kind/` local setup.
- CD in `release.yml` runs `helm upgrade --install` against the cluster on tag.
- Load-test report: p50/p95/p99 latency at N replicas, HPA scaling event captured in a screenshot.

### 🧠 Core concepts you must be able to explain
- Serving: sync vs async vs batch inference, latency vs throughput, dynamic batching, cold start, model loading strategies, versioned endpoints, input validation and adversarial inputs, why ONNX Runtime (graph optimisations, execution providers), thread settings, warm-up requests.
- Kubernetes architecture: control plane (API server, etcd, scheduler, controller manager) vs nodes (kubelet, kube-proxy, container runtime); the reconciliation loop; declarative desired state.
- Objects: Pod (and why you never create bare Pods), Deployment → ReplicaSet → Pod, Service types and how ClusterIP DNS works, Ingress and ingress controllers, ConfigMap/Secret (and that Secrets are only base64), Namespace, ServiceAccount + RBAC, PersistentVolume/Claim, Job/CronJob (used for the drift job in stage 7).
- Scheduling & reliability: requests vs limits (and CPU throttling vs OOMKill), QoS classes, liveness vs readiness vs startup probes, rolling update parameters (`maxSurge`, `maxUnavailable`), rollback and revision history, PodDisruptionBudget, HPA (metrics-server, target utilisation, scale-down stabilisation), node autoscaling.
- Helm: charts, values, templates, releases, `upgrade --install`, `rollback`, `--atomic`, chart dependencies.
- EKS specifics: node groups, IAM roles for service accounts (IRSA) so a pod can read S3 without keys, the AWS Load Balancer Controller, cost of a control plane (~US$73/month, so create and destroy within a few days).

### 🧪 Hands-on exercises
1. Write `serving/app.py`: load ONNX model from S3 by version at startup, `/predict` (multipart image), `/healthz`, `/readyz` (false until model loaded), `/metrics`. Add a warm-up inference. Unit-test with `TestClient`.
2. Load-test locally with `k6` (`serving/loadtest.js`) at 10/50/200 virtual users; record p50/p95/p99 and CPU; then set ONNX Runtime `intra_op_num_threads` and compare.
3. `kind create cluster` with 3 nodes; write raw manifests first (Deployment, Service, ConfigMap, HPA), apply them, then convert to a Helm chart. Keep both; the manifests teach, the chart ships.
4. Set requests/limits; deliberately set memory limit too low and observe `OOMKilled`; set CPU limit too low and observe latency; fix and explain.
5. Break readiness: make `/readyz` return 500 and watch the pod be removed from the Service endpoints while staying alive.
6. Rolling update to a bad image tag; observe the rollout stall; `helm rollback`; describe what the two ReplicaSets did.
7. Install metrics-server, drive load with k6, watch `kubectl get hpa -w` scale 1 → 4 → 1.
8. Terraform an EKS cluster (`terraform-aws-modules/eks`), install the chart with IRSA for S3 access, expose via ALB Ingress, run the same load test, screenshot, **destroy the cluster the same day**.
9. `kubectl` fluency drill: without docs, do `get/describe/logs -f --previous/exec -it/port-forward/top/rollout status/rollout undo/explain`.

### 💥 Common pitfalls & break-it drills
- No resource requests → scheduler packs pods → noisy neighbours. No limits → one pod eats the node.
- Liveness probe that depends on a downstream service → cascading restarts.
- Secrets in `values.yaml` committed to git. Use external secrets or sealed secrets; at minimum, `--set` from CI.
- Forgetting EKS costs; forgetting the ALB and EBS volumes left behind after `destroy`.
- Drill: `kubectl delete pod` on a running pod; explain what recreated it. `kubectl drain` a node; watch pods move.
- Drill: cause `ImagePullBackOff` (wrong tag), `CrashLoopBackOff` (bad env var), `Pending` (requests too large). Diagnose each from `describe` + `logs` only.

### 🎤 Interview questions
- What happens, component by component, when you run `kubectl apply -f deployment.yaml`?
- Difference between liveness and readiness probes; give a failure caused by getting each wrong.
- How does a Service route traffic to pods? What is an Endpoint?
- Requests vs limits: what does the scheduler use, what does the kubelet enforce?
- How would you deploy a new model version with zero downtime and roll back in under a minute?
- How does a pod in EKS access S3 without an access key?
- Why ONNX Runtime instead of serving the PyTorch model directly?

<details>
<summary><b>📖 Resources</b> — click to expand</summary>

- Kubernetes docs "Concepts" (Workloads, Services, Configuration, Policies) and "Tasks": https://kubernetes.io/docs/concepts/
- *Kubernetes Up & Running*, 3rd ed. (Burns, Beda, Hightower, Evenson)
- kind quick start: https://kind.sigs.k8s.io/docs/user/quick-start/  ·  Helm docs: https://helm.sh/docs/
- Kubernetes The Hard Way (read once, do if curious): https://github.com/kelseyhightower/kubernetes-the-hard-way
- EKS Terraform module: https://github.com/terraform-aws-modules/terraform-aws-eks  ·  IRSA: https://docs.aws.amazon.com/eks/latest/userguide/iam-roles-for-service-accounts.html
- FastAPI docs: https://fastapi.tiangolo.com/  ·  ONNX Runtime Python + performance tuning: https://onnxruntime.ai/docs/performance/
- k6 docs: https://grafana.com/docs/k6/latest/
- KServe (read to understand what managed serving adds): https://kserve.github.io/website/

</details>

### 🏁 You can claim it when
- [ ] A bad model version rolled out and you rolled it back with one command, and can explain what the ReplicaSets did.
- [ ] HPA scaled your pods from 1 to ≥ 4 under a load test and back down.
- [ ] You can debug a `CrashLoopBackOff` and a `Pending` pod without documentation.

<p align="right"><a href="#-master-task-tracker-start-here">⬆ Back to tracker</a></p>

---

<a id="stage-7"></a>

## 📈 Stage 7 — Model monitoring

| | |
|---|---|
| 📅 **When** | Week 10 |
| 🎯 **Skill claimed** | model monitoring |
| ⏭️ **Next** | [Stage 8 — Edge deployment](#stage-8) |

### ✅ Your tasks — Stage 7

**🧰 A. Setup**
- [ ] `prometheus-fastapi-instrumentator`, `evidently` in the venv.
- [ ] `kube-prometheus-stack` and Loki Helm repos added.
- [ ] Email or Slack webhook for Alertmanager.
- [ ] `docs/notes/stage-7.md` created.

**📝 B. Concepts & notes** → `docs/notes/stage-7.md`
- [ ] Why ML fails silently; software vs model monitoring.
- [ ] Three layers: system, service (RED/USE), model metrics.
- [ ] Prometheus model: pull, metric types, labels/cardinality, PromQL `rate()`/histograms, Pushgateway, ServiceMonitor; Alertmanager routing/silences.
- [ ] Drift: covariate vs concept vs label shift; KS/JSD/PSI; reference vs current windows; thresholds.
- [ ] Structured logging, request IDs, Loki; SLI/SLO/error budget; runbooks.
- [ ] Continuous training loop: alert → retrain → gate → human approval.

**🔨 C. Build**
- [ ] Instrumented `serving/app.py`: latency histogram, status counter, in-flight gauge, model-version label, mean-confidence gauge, per-class prediction counters.
- [ ] `deploy/monitoring/`: kube-prometheus-stack values, `ServiceMonitor`, alert rules, Grafana dashboard JSON, Loki config.
- [ ] Inference logging as JSON to stdout → Loki; shipped to S3 for the drift job.
- [ ] Drift `CronJob`: pulls 24 h of logs from S3, Evidently `DataDriftPreset` + image-statistic tests, HTML report to S3, `drift_score` to Pushgateway.
- [ ] Drift simulation script (haze / winter brightness / channel swap).
- [ ] Drift alert → `repository_dispatch` → `nightly-train.yml` → PR through quality gate.
- [ ] `docs/RUNBOOK.md`: per alert → meaning, causes, first 3 commands, escalation.
- [ ] Grafana screenshot in `README.md`.

**🧪 D. Exercises**
- [ ] Ex 1: instrument the app (see C).
- [ ] Ex 2: kube-prometheus-stack in kind; PromQL for p95 + error rate; dashboard JSON committed.
- [ ] Ex 3: alert rules (p95 > 300 ms/5 min, error rate > 1 %, class collapse, drift > threshold); route via Alertmanager; trigger each deliberately.
- [ ] Ex 4: JSON inference logs → Loki; query by request ID.
- [ ] Ex 5: drift CronJob end to end.
- [ ] Ex 6: run the 3 drift simulations (2,000 requests each); record which signal fired first in notes.
- [ ] Ex 7: drift alert triggers retraining PR.
- [ ] Ex 8: `RUNBOOK.md` written.

**💥 E. Break-it drills**
- [ ] Kill Prometheus; see what is lost; add persistence.
- [ ] Silence an alert and "forget"; discover the gap; add a silence-expiry check.
- [ ] Add a high-cardinality label (e.g. request ID) to a metric; watch memory; remove it.

**🎤 F. Interview & close-out**
- [ ] Answer the interview questions aloud; record once.
- [ ] Tick both "You can claim it when" boxes.
- [ ] 10-line "what I learned" in `docs/notes/stage-7.md`.
- [ ] Stage 7 ✅ in tracker + `README.md`; commit via PR.

### 📚 Learn
- The three layers: system metrics (CPU/mem/GPU), service metrics (RPS, latency, error rate, via `prometheus-fastapi-instrumentator`), model metrics (prediction distribution, confidence histogram, input feature statistics, data drift, concept drift when labels arrive).
- Prometheus + Grafana on the cluster (kube-prometheus-stack Helm chart), Alertmanager rule → email/Slack.
- Drift with **Evidently** (or `alibi-detect`): log inference inputs/outputs to S3, nightly job computes drift report vs training reference, pushes a drift score to Prometheus.
- Structured JSON logging, request IDs, log aggregation with Loki.
- Close the loop: drift alert → triggers `nightly-train.yml` (continuous training) → new model to registry → CD.

### 🔨 Build — this is where the satellite framing pays off
- Simulate drift realistically: serve images with synthetic haze / different season / different sensor gain and watch the drift score cross the threshold.
- Grafana dashboard JSON committed to repo; screenshot in README.
- A runbook `RUNBOOK.md`: what each alert means and what to do.

### 🧠 Core concepts you must be able to explain
- Why ML systems fail silently: no exceptions, just worse predictions. The difference between software monitoring and model monitoring.
- Metric types: counters, gauges, histograms, summaries; labels/cardinality; RED (rate, errors, duration) and USE methods; SLIs/SLOs/error budgets; percentiles and why averages lie.
- Prometheus model: pull-based scraping, `ServiceMonitor`, PromQL basics (`rate`, `histogram_quantile`, `sum by`), recording rules, alerting rules, Alertmanager routing/silencing.
- Grafana: dashboards as code (JSON), variables, panels for latency heatmaps, annotations for deployments.
- Model-level signals: prediction class distribution, confidence distribution, input statistics (per-channel mean/std, brightness, cloud fraction), embedding drift, ground-truth delay and how to evaluate when labels arrive late.
- Drift: covariate shift vs prior shift vs concept drift; statistical tests (KS, PSI, Jensen–Shannon, Wasserstein) and their sensitivity to sample size; reference window vs current window; thresholds and alert fatigue.
- Logging: structured JSON, correlation/request IDs, sampling, PII considerations, log-based metrics, Loki + LogQL.
- Closing the loop: alert → runbook → retrain trigger → evaluation gate → human approval → deploy; and why fully automatic promotion is usually wrong.

### 🧪 Hands-on exercises
1. Instrument the FastAPI app: request latency histogram, request counter by status, in-flight gauge, model version label, plus custom gauges for mean predicted confidence and per-class prediction counts.
2. Install `kube-prometheus-stack` in kind; add a `ServiceMonitor`; write PromQL for p95 latency and error rate; build a dashboard; commit its JSON.
3. Write alert rules: p95 > 300 ms for 5 min, error rate > 1 %, prediction distribution collapsed to one class, drift score > threshold. Route to email (or a Slack webhook) via Alertmanager. Trigger each one deliberately.
4. Log every inference (features summary + prediction + confidence + request ID) as JSON to stdout; ship to Loki; query by request ID.
5. Build the drift job: a Kubernetes `CronJob` that pulls the last 24 h of inference logs from S3, runs an Evidently `DataDriftPreset` + custom image-statistic tests against the training reference, writes an HTML report to S3 and pushes a `drift_score` gauge to Prometheus via Pushgateway.
6. Simulate drift: a script that sends 2,000 requests of EuroSAT images with (a) added haze, (b) reduced brightness (winter), (c) channel swap (new sensor). Watch the drift score, the confidence gauge and the class-distribution alert. Record which signal fired first for each shift and write it in `docs/notes/stage-7.md`.
7. Wire the drift alert to trigger `nightly-train.yml` via `repository_dispatch`; the retrain opens a PR that must pass the quality gate and be approved by you.
8. Write `RUNBOOK.md`: for each alert → meaning, likely causes, first three commands to run, escalation.

### 💥 Common pitfalls & break-it drills
- High-cardinality labels (request ID as a Prometheus label) → Prometheus falls over. Understand cardinality.
- Drift tests on 50 samples fire constantly; on 50,000 they fire on trivial shifts. Tune windows and thresholds.
- Monitoring accuracy when you have no labels: be explicit about proxies.
- Drill: kill Prometheus; see what you lose; add persistence. Silence an alert; forget to unsilence; discover the gap.

### 🎤 Interview questions
- What would you monitor for this model in production, in order of priority, and why?
- How do you detect that a model is degrading when you don't get labels for weeks?
- Explain data drift vs concept drift with an example from satellite imagery.
- What is `histogram_quantile` doing? Why can't you average p95s across pods?
- An alert fires at 3 a.m. saying drift > threshold. What do you do, and what should happen automatically?

<details>
<summary><b>📖 Resources</b> — click to expand</summary>

- Prometheus docs (Concepts, Querying basics, Alerting): https://prometheus.io/docs/introduction/overview/
- Grafana dashboards best practices: https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/best-practices/
- kube-prometheus-stack chart: https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack
- Evidently docs + tutorials: https://docs.evidentlyai.com/  ·  Evidently "ML monitoring" free course: https://learn.evidentlyai.com/
- Google SRE book, ch. 6 "Monitoring Distributed Systems" (free): https://sre.google/sre-book/monitoring-distributed-systems/
- *Designing ML Systems*, ch. 8 (data distribution shifts and monitoring)
- prometheus-fastapi-instrumentator: https://github.com/trallnag/prometheus-fastapi-instrumentator
- Loki docs: https://grafana.com/docs/loki/latest/

</details>

### 🏁 You can claim it when
- [ ] A drift alert fired, you diagnosed it from the dashboard, and retraining was triggered automatically.
- [ ] You can list exactly which metrics you'd monitor for a new model and why.

<p align="right"><a href="#-master-task-tracker-start-here">⬆ Back to tracker</a></p>

---

<a id="stage-8"></a>

## 🛰️ Stage 8 — Edge deployment: Raspberry Pi target, embedded and onboard-satellite constraints

| | |
|---|---|
| 📅 **When** | Week 11–12 |
| 🎯 **Skill claimed** | edge deployment (embedded ARM devices, onboard/offline inference systems) |
| ⏭️ **Next** | [Stage 9 — Capstone](#stage-9) |

### ✅ Your tasks — Stage 8

**🧰 A. Setup**
- [ ] `qemu` installed; `docker buildx` can target `linux/arm64`; `docker run --platform linux/arm64 python:3.12-slim uname -m` prints `aarch64`.
- [ ] `onnxsim`, `onnxruntime` (quantisation tools), `cryptography` in the venv.
- [ ] Raspberry Pi OS Lite 64-bit image downloaded for `qemu-system-aarch64`.
- [ ] `docs/notes/stage-8.md` created.

**📝 B. Concepts & notes** → `docs/notes/stage-8.md`
- [ ] CPU inference cost model; batch size 1; thread scaling on 4 small cores.
- [ ] Quantisation: FP32/FP16/INT8, symmetric vs asymmetric, per-tensor vs per-channel, static vs dynamic vs QAT, accuracy loss, ops that quantise badly.
- [ ] Distillation & pruning.
- [ ] ONNX opsets, graph optimisation, `onnxsim`, execution providers, session options; when TFLite instead.
- [ ] QEMU user-mode vs system emulation; why emulated timings are unreliable; multi-arch manifests; arm64 wheel availability.
- [ ] Embedded Linux: Pi OS Lite, first boot, systemd hardening (`ProtectSystem`, `MemoryMax`, `WatchdogSec` + `sd_notify`), read-only root, journald caps, SD wear, thermal, power estimation.
- [ ] OTA design: Ed25519 signatures, manifest hashes, A/B slots, self-test, rollback, resumable downloads, contact windows.
- [ ] Onboard-satellite context: radiation & watchdogs, power budgets, no interactive access, Φ-sat-1, OPS-SAT.

**🔨 C. Build** (all in `edge/`, all on the Mac)
- [ ] `edge/export_and_quantize.py` → `model_fp32.onnx`, `model_int8.onnx`, `model_distilled_int8.onnx`.
- [ ] `edge/benchmark.py`: accuracy, p50/p95 latency, RSS, size, threads sweep; native + arm64 emulated.
- [ ] `edge/Dockerfile.pi` (arm64 slim + onnxruntime), pushed as multi-arch to GHCR.
- [ ] `edge/onboard/`: offline service loop (local sensor folder, per-frame time budget, append-only telemetry, no sockets).
- [ ] `edge/onboard/orbiteye.service` (watchdog, hardening) + `edge/provision/firstboot.sh` + `edge/provision/README.md`.
- [ ] `edge/ground/`: `sign_bundle.py`, `downlink.py` (verify → inactive slot → self-test → switch/rollback), `uplink.py`.
- [ ] `edge/tests/`: signature verification, A/B switching, rollback, time budget, no-network, power-cut mid-install.
- [ ] `.github/workflows/edge.yml`: build arm64 image, run edge tests under emulation, 10-min soak, publish table.
- [ ] `edge/README.md` + edge results table in root `README.md`; enable the edge badge.

**🧪 D. Exercises**
- [ ] Ex 1: ONNX export static `[1,3,64,64]`; `onnxsim`; outputs match PyTorch within 1e-4 on 100 images.
- [ ] Ex 2: static INT8 (500 calibration images) vs dynamic; accuracy deltas written down.
- [ ] Ex 3: distil to MobileNetV3-small (T=4, α=0.7); quantise; fill the FP32/INT8/distilled table.
- [ ] Ex 4: threads sweep 1/2/4 under emulation; relative numbers only; explain.
- [ ] Ex 5: `Dockerfile.pi` via `buildx`; run with `--platform linux/arm64`; tests pass inside.
- [ ] Ex 6: boot Pi OS Lite in `qemu-system-aarch64`; SSH; `firstboot.sh`; unit installed; starts on boot; `kill -9` → watchdog restarts; journald caps checked.
- [ ] Ex 7: onboard loop with 200 ms budget; budget-miss telemetry; `strace -f -e trace=network` shows no socket.
- [ ] Ex 8: OTA cycle with all three failure tests (tampered signature, corrupt model rollback, power-cut).
- [ ] Ex 9: soak test: 10 min in CI, 24 h once locally; RSS growth < 5 %, zero budget misses.
- [ ] Ex 10: energy per inference estimate (CPU time × 5 W) with assumption in README.

**💥 E. Break-it drills**
- [ ] Inspect the quantised graph for ops that fell back to FP32.
- [ ] Calibrate on mismatched-brightness images; watch INT8 accuracy collapse; fix with representative set.
- [ ] Configure watchdog without `sd_notify` → restart loop; then fix.
- [ ] Flip one byte in the signed model; verification fails *before* load.

**🎤 F. Interview & close-out**
- [ ] Answer the interview questions aloud; record once.
- [ ] Tick all 4 "You can claim it when" boxes.
- [ ] 10-line "what I learned" in `docs/notes/stage-8.md`.
- [ ] Stage 8 ✅ in tracker + `README.md`; commit via PR.

> [!NOTE]
> **Your situation:** you have no Raspberry Pi and no Jetson. That is fine. The target device is a
> **Raspberry Pi 4/5 (ARM64, CPU only)**, and you build, test and exercise the *entire* edge lifecycle on
> your Mac using ARM64 emulation. Every artefact is real and deployable; the only thing you skip is
> plugging in the board. If you ever get a Pi (~A$120) it is a 30-minute job to run the same image on it.

> [!NOTE]
> **Why a Pi is a good stand-in for "embedded / onboard satellite":** CPU-only ARM, a few GB of RAM,
> tight power budget, no GPU, no reliable network. Many CubeSat payload computers are exactly this class
> of hardware, so the constraints you design for are the real ones.

### 📚 Learn — deeply
- Model optimisation for CPU: ONNX export + `onnxsim`, static shapes, INT8 post-training quantisation
  with a calibration set (ONNX Runtime quantisation tools), structured pruning, knowledge distillation
  ResNet-18 → MobileNetV3-small. Measure accuracy vs latency vs model size at every step (table in README).
- CPU inference runtimes and why they matter: ONNX Runtime (default), TFLite/LiteRT via onnx2tf
  (very common on Pi), OpenVINO awareness. Thread count tuning, NHWC vs NCHW, arena allocators.
- Cross-platform containers: `docker buildx --platform linux/arm64`, multi-arch manifests, QEMU
  user-mode emulation on macOS (`docker run --platform linux/arm64`), and why emulated timings are
  not representative (you report *relative* numbers and say so).
- Device provisioning without the device: Raspberry Pi OS Lite 64-bit image, `cloud-init`/first-boot
  scripts, headless SSH, `systemd` service with `Restart=on-failure` + `WatchdogSec=`, journald log limits,
  read-only root filesystem with an overlay, log rotation, `/boot/config.txt` power tweaks.
  Emulate the OS itself with QEMU (`qemu-system-aarch64` booting the official Pi image) so you can actually
  run your `systemd` unit and first-boot script end to end on your Mac.
- OTA (over-the-air) update cycle: signed model bundle (`model.onnx` + `manifest.json` + Ed25519
  signature), A/B model slots with automatic rollback if the new model fails a self-test, all triggered
  only when a "ground contact" (network) is detected.
- **Onboard-satellite constraints** to design for explicitly: no network at inference time, fixed power
  budget (Watt-per-inference estimated from CPU time), radiation → watchdog and graceful restart,
  deterministic memory (pre-allocated buffers, no growth over a 24 h soak test), tiny storage, delayed and
  partial telemetry downlink, per-frame time budget. Read ESA Φ-sat-1 (cloud detection in orbit) and
  OPS-SAT experiments as reference architectures and write a one-page comparison.

### 🔨 Build — all in `edge/`, all runnable on the Mac
- `edge/export_and_quantize.py` → `model_fp32.onnx`, `model_int8.onnx`, `model_distilled_int8.onnx`.
- `edge/benchmark.py`: accuracy, p50/p95 latency, RSS memory, model size, threads sweep; runs natively on
  the Mac and under `--platform linux/arm64` emulation, output as a Markdown table.
- `edge/Dockerfile.pi` (arm64, `python:3.12-slim-bookworm` + `onnxruntime`), pushed as a multi-arch image
  from CI with `buildx` (amd64 + arm64).
- `edge/onboard/` — the **offline inference service**: reads patches from a local "sensor" folder, enforces
  a per-frame time budget, writes compact predictions to an append-only telemetry log, exposes a local
  health file, and never opens a network socket in this mode.
- `edge/onboard/orbiteye.service` (systemd, watchdog) + `edge/provision/firstboot.sh` + `edge/provision/README.md`
  (the exact steps to flash a Pi, so anyone with a board can deploy in 30 minutes).
- `edge/ground/` — the **ground-contact cycle**: `uplink.py` (sync telemetry to S3 when network is present),
  `downlink.py` (fetch signed model bundle, verify signature, install to inactive slot, run self-test,
  switch or roll back), `sign_bundle.py` (run on the "ground" side, i.e. in CI on model promotion).
- `edge/tests/`: unit tests for signature verification, A/B slot switching, rollback, time-budget
  enforcement, and a 10-minute soak test that asserts memory does not grow.
- CI job `edge.yml`: builds the arm64 image, runs the edge tests inside it under emulation, publishes
  the benchmark table as a PR comment. This is the "cycle" you asked for: model promoted in MLflow →
  CI signs a bundle → bundle lands in S3 → emulated device fetches, verifies, self-tests, switches.

### 🧠 Core concepts you must be able to explain
- Why edge: latency, bandwidth, privacy, autonomy when disconnected. Edge vs cloud trade-offs and the "compute where the data is" argument for satellites (downlink is the bottleneck, not compute).
- CPU inference cost model: FLOPs vs memory bandwidth, NHWC vs NCHW, cache effects, why batch size 1 dominates on-device, thread scaling limits on 4 small cores.
- Quantisation: FP32 → FP16 → INT8; symmetric vs asymmetric, per-tensor vs per-channel; post-training static (calibration) vs dynamic vs quantisation-aware training; typical accuracy loss and how to measure it; operators that don't quantise well.
- Distillation and pruning: teacher/student loss, temperature, structured vs unstructured pruning and why unstructured rarely speeds up CPUs.
- ONNX: opsets, graph optimisation levels, `onnxsim`, execution providers, session options, IO binding; when to convert to TFLite/LiteRT instead.
- ARM64 and cross-building: what QEMU user-mode emulation does, why timings are unreliable under it, multi-arch manifests, `python:slim` arm64 wheels availability (check `onnxruntime` and `pillow` have arm64 wheels).
- Embedded Linux: Raspberry Pi OS Lite, first-boot provisioning, `systemd` hardening (`ProtectSystem`, `MemoryMax`, `WatchdogSec` + `sd_notify`), overlay/read-only root, journald size caps, SD-card wear, thermal throttling, power measurement (or estimation from CPU time × TDP).
- OTA update design: signed artefacts (Ed25519), manifest with hashes, A/B slots, health self-test, automatic rollback, idempotent and resumable downloads, update only during "contact windows".
- Onboard-satellite context: radiation-induced faults and watchdogs, power budgets in watts, no interactive access, delayed/limited telemetry, determinism, and the reference missions Φ-sat-1 (Intel Movidius, cloud detection) and OPS-SAT.

### 🧪 Hands-on exercises
1. Export ResNet-18 to ONNX with static shape `[1,3,64,64]`, run `onnxsim`, verify outputs match PyTorch within 1e-4 on 100 images.
2. Static INT8 quantisation with ONNX Runtime using 500 calibration images; measure accuracy delta on the full test set. Then try dynamic quantisation and compare. Write both numbers down.
3. Distil to MobileNetV3-small (temperature 4, α 0.7 as a start); quantise; fill the FP32/INT8/distilled table with accuracy, size, p50 latency (native Mac, then arm64 emulated), RSS.
4. Threads sweep (1/2/4) under arm64 emulation and record relative change; explain why absolute numbers are meaningless under QEMU.
5. Build `edge/Dockerfile.pi` for `linux/arm64` with `buildx`; run it with `--platform linux/arm64` on the Mac; run the edge tests inside.
6. Boot the official Raspberry Pi OS Lite 64-bit image in `qemu-system-aarch64`; SSH into it; run `firstboot.sh`; install the `systemd` unit; reboot; confirm the service starts on boot; `kill -9` it and watch the watchdog restart it; check `journalctl` size caps.
7. Implement the onboard service loop with a per-frame budget of e.g. 200 ms; if exceeded, log a "budget miss" telemetry event and skip. Prove with `strace -f -e trace=network` that no socket is opened in onboard mode.
8. Implement the OTA cycle: `sign_bundle.py` (ground/CI), `downlink.py` (verify → install to inactive slot → self-test on 20 golden images → switch or rollback), `uplink.py`. Tests: tampered signature rejected; corrupted model fails self-test and rolls back; power-cut mid-install (kill during copy) leaves the active slot intact.
9. 10-minute soak test in CI (24 h once locally): assert RSS growth < 5 % and zero budget misses.
10. Estimate energy per inference: CPU time × assumed 5 W for a Pi 4 under load; put the number and the assumption in the README.

### 💥 Common pitfalls & break-it drills
- Dynamic input shapes and `Resize` ops that silently fall back to FP32 after quantisation; inspect the quantised graph.
- Calibrating on training images that don't match deployment brightness → INT8 accuracy collapse; use a representative set.
- Timing under emulation and reporting it as real. Always label emulated numbers.
- Watchdog configured but the service never calls `sd_notify(WATCHDOG=1)` → restart loop. Test it.
- Drill: flip one byte in the model file after signing; ensure verification fails *before* the model is loaded.

### 🎤 Interview questions
- How would you get a 45 MB FP32 model to run at 5 fps on a 4-core ARM CPU with 2 GB RAM? Walk through the steps and expected losses.
- Explain post-training static quantisation. Why do you need calibration data? Per-channel vs per-tensor?
- How do you update a model on a device you can only reach for 8 minutes every 90 minutes, over a lossy link, without ever bricking it?
- What is a watchdog and how does `systemd` implement one?
- Why might a distilled model be a better edge choice than a pruned one?
- What breaks first on a real device that you couldn't see under emulation?

<details>
<summary><b>📖 Resources</b> — click to expand</summary>

- ONNX Runtime quantisation guide: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- ONNX Runtime on ARM / performance tuning: https://onnxruntime.ai/docs/performance/tune-performance/
- PyTorch ONNX export: https://pytorch.org/docs/stable/onnx.html  ·  onnx-simplifier: https://github.com/daquexian/onnx-simplifier
- Distillation paper (Hinton et al.): https://arxiv.org/abs/1503.02531  ·  MobileNetV3 paper: https://arxiv.org/abs/1905.02244
- Docker multi-platform builds: https://docs.docker.com/build/building/multi-platform/
- Raspberry Pi OS docs (headless setup, `raspi-config`, `config.txt`): https://www.raspberrypi.com/documentation/computers/
- Booting Raspberry Pi OS in QEMU: https://github.com/dhruvvyas90/qemu-rpi-kernel (and the `qemu-system-aarch64 -M raspi3b` docs: https://www.qemu.org/docs/master/system/arm/raspi.html)
- systemd watchdog & hardening: `man systemd.service` (WatchdogSec), `man systemd.exec`; https://0pointer.de/blog/projects/watchdog.html
- Ed25519 in Python (`cryptography`): https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ed25519/
- The Update Framework (TUF) — read the threat model: https://theupdateframework.io/
- ESA Φ-sat-1 overview: https://www.esa.int/Applications/Observing_the_Earth/Ph-sat  ·  Giuffrida et al., "The Φ-Sat-1 Mission" (IEEE TGRS 2021)
- ESA OPS-SAT: https://www.esa.int/Enabling_Support/Operations/OPS-SAT
- TinyML book (Warden & Situnayake) for the embedded mindset

</details>

### 🏁 You can claim it when
- [ ] Table: FP32 → INT8 → distilled INT8, with accuracy, latency, size, memory (native + arm64 emulated).
- [ ] The arm64 image boots in the emulated Pi OS as a `systemd` service, runs fully offline, survives
      `kill -9` via the watchdog, and passes the soak test.
- [ ] A signed model update was pulled during a simulated ground contact; a deliberately broken bundle
      was rejected and rolled back automatically.
- [ ] You can explain why INT8 ONNX Runtime on a CPU-only ARM board beats shipping PyTorch, with numbers.

> [!TIP]
> **How to word it on the CV (truthfully):** "Edge deployment: built and CI-tested an offline, watchdog-supervised
> INT8 ONNX inference service for ARM64/Raspberry Pi with signed OTA model updates, designed for onboard-satellite
> constraints; validated under ARM64 emulation." In interviews say plainly that you did not have physical hardware
> and that the image is ready to flash. That is a strong, honest answer; the Jetson mention comes off the CV line.

<p align="right"><a href="#-master-task-tracker-start-here">⬆ Back to tracker</a></p>

---

<a id="stage-9"></a>

## 🎓 Stage 9 — Capstone: make it claimable

| | |
|---|---|
| 📅 **When** | Final week |
| ⏭️ **Next** | [Appendices](#appendix-a) |

### ✅ Your tasks — Stage 9

**🧰 A. Setup**
- [ ] Screen-recording tool ready; all clouds recreatable via `terraform apply`.

**📝 B. Concepts & notes**
- [ ] Fill in Appendix C glossary with your own one-line definitions.
- [ ] Reread every `docs/notes/stage-N.md` "what I learned" and list your 5 weakest topics; revise them.

**🔨 C. Build**
- [ ] `README.md`: architecture diagram, all three results tables filled from scripts, all status rows ✅, screenshots.
- [ ] `docs/adr/`: 5–8 ADRs (ONNX Runtime vs TorchServe, INT8 on edge, registry-pull vs bake-in, OIDC, quality gate in CI, Pi as satellite stand-in…).
- [ ] `docs/DESIGN.md` and `docs/RUNBOOK.md` final pass.
- [ ] 3-minute demo video: PR → CI → release → k8s → Grafana → drift → retrain → signed bundle → emulated device. Link from README.
- [ ] One blog / LinkedIn post per major stage; link from README and CV.
- [ ] Every Appendix A evidence row has a clickable link.

**🧪 D. Exercises**
- [ ] Do a full fresh-clone run-through of the Quickstart on a clean machine or container; fix everything that breaks.
- [ ] Recreate the whole AWS stack with `terraform apply`, record the demo, `terraform destroy`.

**💥 E. Break-it drills**
- [ ] Ask a friend (or an AI) to interview you for 30 min using the interview questions from all stages, without notes.

**🎤 F. Close-out**
- [ ] Tear down all cloud resources; confirm zero spend next month.
- [ ] Write the three CV bullets below into your CV, truthfully.
- [ ] Stage 9 ✅ in tracker + `README.md`; final commit; tag `v1.0.0`.

- README with architecture diagram, the three results tables (baseline, serving load test, edge optimisation), and a 3-minute demo video.
- `docs/` with `DESIGN.md`, `RUNBOOK.md`, `ADR/` (architecture decision records, 5–8 short ones: "why ONNX Runtime not TorchServe", "why INT8 on the edge", etc.). Interviewers love ADRs.
- Write one blog post / LinkedIn article per major stage. Link them from the CV.
- Tear down all cloud resources; keep Terraform so you can recreate in a demo.

### 📝 CV bullets you'll be able to write truthfully
- Built an end-to-end MLOps pipeline for satellite land-cover classification: DVC + MLflow experiment tracking, Dockerised training/serving, GitHub Actions CI/CD with model quality gates, Helm-deployed FastAPI/ONNX inference on Kubernetes (EKS) with HPA, Prometheus/Grafana monitoring and Evidently drift-triggered retraining.
- Provisioned infrastructure with Terraform on AWS (S3, ECR, EKS, EC2, IAM OIDC), with an equivalent container deployment on Azure Container Apps and a managed Azure ML training job.
- Built and CI-tested an offline, watchdog-supervised INT8 ONNX inference service for ARM64/Raspberry Pi with signed OTA model updates, designed for onboard-satellite constraints (power, no-network, A/B rollback); validated under ARM64 emulation.

<p align="right"><a href="#-master-task-tracker-start-here">⬆ Back to tracker</a></p>

---

<a id="resources"></a>

## 📚 Primary resources (official-first, cross-stage)

- Linux/Git: *The Linux Command Line* (Shotts, free PDF), Pro Git (free), MIT "Missing Semester".
- DVC docs "Get Started", MLflow docs "Tracking" + "Model Registry".
- Docker docs "Get started" + "Best practices for Dockerfiles"; *Docker Deep Dive* (Poulton).
- GitHub Actions docs; CML (iterative.ai) docs.
- AWS Skill Builder free tier; Terraform "Get Started – AWS"; Azure free-tier docs for Container Apps and Azure ML.
- Kubernetes docs "Concepts" + "Tasks"; *Kubernetes Up & Running*; kind docs; Helm docs.
- Prometheus/Grafana docs; Evidently docs; Google SRE book chapter on monitoring.
- ONNX Runtime quantisation docs; Docker buildx multi-platform docs; Raspberry Pi OS headless setup docs; QEMU aarch64 Pi boot guides; systemd watchdog docs; ESA Φ-sat-1 and OPS-SAT papers.
- Whole-lifecycle framing: *Designing Machine Learning Systems* (Huyen), Made With ML MLOps course.

---

<a id="appendix-a"></a>

## 🗂️ Appendix A — Evidence you must have at the end

*What a hiring manager can click.*

| Claim | Evidence in repo / links |
|---|---|
| Git / Linux | Commit history with PRs, `Makefile`, provisioning scripts, `docs/notes/stage-1.md` |
| Reproducible ML | `dvc.yaml`, `params.yaml`, MLflow screenshots, `DESIGN.md`, tests |
| Docker | `docker/` with size table in `docker/README.md`, Trivy report artefact in CI |
| CI/CD | Green workflow runs, a PR blocked by the quality gate (link to it), a tagged release with image digest |
| AWS / Azure | `infra/*` Terraform, screenshots of live URLs in `docs/assets/`, cost screenshot |
| Kubernetes | Helm chart, k6 report, HPA scaling screenshot, rollback demo in the video |
| Monitoring | Grafana JSON, alert rules, drift report HTML, `RUNBOOK.md`, drift-triggered retrain PR |
| Edge | Optimisation table, arm64 image in GHCR, OTA tests green, soak-test output, provisioning README |
| End-to-end | 3-minute demo video: PR → CI → release → k8s → Grafana → drift → retrain → signed bundle → emulated device |

## 💸 Appendix B — Cost-control checklist

> [!WARNING]
> Read this before **every** cloud session. Idle NAT gateways, load balancers and clusters are where the money goes.

- [ ] Budget alarm exists on every cloud account.
- [ ] `terraform destroy` at the end of every session; check the console for orphaned load balancers, EBS volumes, NAT gateways, static IPs.
- [ ] EKS/AKS clusters live for hours or days, never weeks.
- [ ] Use spot for anything restartable; use `t3`/`c6i` CPU instances for training here.
- [ ] Tag everything `project=orbiteye` and filter Cost Explorer by tag weekly.

## 🔤 Appendix C — Glossary

*Write your own one-line definition next to each term; this is a study tool.*

A/B slots · ALB · AMI · Alertmanager · Apptainer · artefact store · autoscaling (HPA / cluster) · backend store · blue/green vs canary · buildx · calibration (quantisation) · cardinality · CD vs continuous deployment · cgroups · ClusterIP · concept drift · covariate shift · CronJob · CrashLoopBackOff · data contract · DDP (know what it is even though you skip it) · digest · distillation · DVC remote · Ed25519 · EKS · endpoint (k8s) · error budget · execution provider · GHCR · HPA · IAM role vs user · Ingress · IRSA · JSD / KS / PSI · kind · kubelet · layer cache · liveness / readiness / startup probe · Loki · macro-F1 · manifest list · metrics-server · MLflow registry stage · multi-stage build · namespace (Linux) · namespace (k8s) · NAT gateway · OIDC · ONNX opset · onboard vs ground segment · OOMKilled · overlay filesystem · p95 / p99 · Pending (pod) · PID 1 · PodDisruptionBudget · post-training quantisation · PromQL · Pushgateway · QoS class · quality gate · QEMU user-mode vs system emulation · rate() · read-only rootfs · reconciliation loop · reflog · remote state · ReplicaSet · requests vs limits · RED / USE · rolling update · runbook · sd_notify · SLI / SLO · smoke test · spot instance · ServiceMonitor · soak test · stratified split · systemd unit · Terraform drift · TUF · watchdog · workflow_dispatch

## 🆘 Appendix D — If you get stuck

- Every tool above has an official "getting started" that works; do that first, then adapt to OrbitEye. Don't start from a random blog.
- When something fails, read the error fully, then `describe`/`logs`/`journalctl`, then docs, then search. Write the fix in `docs/notes/`.
- Timebox: 45 minutes stuck → write down exactly what you tried → ask (a person, a forum, an AI) with that write-up. The write-up itself solves half of them.
