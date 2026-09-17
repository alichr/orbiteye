# End-to-End MLOps Learning Project: "OrbitEye" — Satellite Land-Cover Classifier

**Goal:** finish one small ML project that legitimately covers every skill on this CV line:

> MLOps / Infrastructure: Docker, Kubernetes, Git, CI/CD, model monitoring; AWS, GCP, Azure;
> Linux; edge deployment (embedded ARM / Raspberry Pi, onboard satellite systems)

(Distributed multi-GPU training, HPC clusters and NVIDIA Jetson are deliberately left out: they need
hardware you do not have. The edge target is a Raspberry Pi, fully exercised via ARM64 emulation.)

**Rule of the plan:** one stage = one skill, learned deeply, with a concrete artefact in the repo that
proves it. You do not move to the next stage until the "You can claim it when…" checklist is done.

---

## 0. The project (keep it deliberately simple)

| Item | Choice | Why |
|---|---|---|
| Task | Classify 64×64 satellite image patches into 10 land-cover classes | Simple, well understood, but genuinely "satellite" so the edge/onboard story is real |
| Dataset | EuroSAT RGB (27,000 Sentinel-2 patches, ~90 MB) | Small enough to train on a laptop CPU/Apple GPU in minutes |
| Model | ResNet-18 (torchvision) fine-tuned, later a tiny MobileNetV3 for edge | Trains in minutes; exports cleanly to ONNX and quantises well for CPU |
| Framework | PyTorch | Industry default; clean ONNX export |
| Serving | FastAPI + ONNX Runtime | Framework-agnostic, fast, runs identically in cloud and on a Raspberry Pi |

The ML is intentionally boring. All the learning is in the lifecycle around it.

**End-state architecture (what you will have built):**

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

**Hardware reality check (you are on a Mac):**
- Stages 1–4, 6, 7 and all of 8 run entirely on your Mac (CPU or Apple MPS). No GPU and no physical edge device is needed anywhere in this plan.
- Edge (stage 8) targets a Raspberry Pi but is built and tested under ARM64 emulation (Docker buildx + QEMU) on your Mac. If you later buy a Pi (~A$120) the same image flashes and runs unchanged.
- Total cloud budget for the whole plan: roughly US$40–70 if you shut things down.

**Timeline:** 9 stages, ~10–12 weeks at 8–10 h/week. Faster if you already know Git/Linux.

---

## Stage 1 — Linux + Git: the professional repo skeleton  (Week 1)

**Skill claimed:** Git, Linux

**Why first:** everything later (CI, Docker, K8s, cloud) is driven from a Linux shell and a Git history. A recruiter opening your repo judges you in 10 seconds by this skeleton.

**Learn (deeply):**
- Shell: `ssh`, `scp`/`rsync`, `tmux`, `htop`, `chmod`/`chown`, environment variables, `systemd` basics, `journalctl`, pipes, `grep`/`awk`/`sed`, cron. Do all of this on a real Linux box: start a free-tier `t3.micro` Ubuntu EC2 instance (this also plants the AWS seed for stage 5).
- Git: branching model (main + feature branches), rebase vs merge, `git bisect`, tags, conventional commits, pull requests with reviews (review your own PRs), `.gitignore` for ML (no data, no checkpoints), pre-commit hooks.
- Python project hygiene: `pyproject.toml`, `uv` or `pip-tools` lockfile, `ruff`, `pytest`, `Makefile`.

**Build:**
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

**You can claim it when:**
- [ ] Repo on GitHub, protected `main`, every change via PR, ≥ 20 meaningful commits with conventional messages.
- [ ] Pre-commit runs `ruff` + `pytest` locally.
- [ ] You can SSH into a Linux VM, set up Python, run a background job under `tmux`, inspect it with `htop`, and schedule a cron job, without looking anything up.

---

## Stage 2 — ML design, data versioning and experiment tracking  (Week 2–3)

**Skill claimed:** "design" half of "design and deploy end-to-end ML models"; data/model versioning

**Learn:**
- Problem framing document: business goal → ML metric (macro-F1) → offline/online evaluation plan → failure modes (cloud cover, seasonal shift → this becomes your drift story in stage 7).
- Data versioning with **DVC**: `dvc init`, `dvc add data/`, remote = local folder now, S3 in stage 5. Reproducible pipeline `dvc.yaml` with stages `prepare → train → evaluate`.
- Experiment tracking with **MLflow**: params, metrics, artefacts, model registry (Staging → Production). Run the MLflow server locally first.
- Config management with **Hydra** or plain YAML + dataclasses. Seeds and determinism.
- Test your ML code: shape tests, a 2-batch overfit test, a data-contract test (class names, image size).

**Build:**
- `train.py` that trains ResNet-18 on EuroSAT on your Mac (CPU/MPS, ~10 min for a few epochs), logs to MLflow, saves `model.pt` + `metrics.json`.
- `dvc repro` reproduces the whole pipeline from raw data.
- A baseline reaching ~95% accuracy (EuroSAT is easy; that's fine).

**You can claim it when:**
- [ ] `git checkout <commit> && dvc pull && dvc repro` rebuilds any past result.
- [ ] MLflow UI shows ≥ 10 tracked runs and a registered model with a version in "Production".
- [ ] `DESIGN.md` in the repo with framing, metrics, data card, model card.

---

## Stage 3 — Docker  (Week 4)

**Skill claimed:** Docker

**Learn (deeply):**
- Images vs containers, layers and caching, multi-stage builds, `.dockerignore`, non-root users, `ENTRYPOINT` vs `CMD`, volumes vs bind mounts, networks, `docker compose`, image tagging strategy (git SHA + semver), registry push/pull, image scanning (`trivy`), size reduction (a 6 GB "works on my machine" image vs a 900 MB one).
- GPU containers (read only, no hardware needed): know what `nvidia/cuda` base images and the NVIDIA Container Toolkit are and how `--gpus all` works, so you can answer the interview question. You will not need them in this plan.
- Reproducibility: pinned base image digest, lockfile installed inside the image.

**Build:**
- `docker/Dockerfile.train` (multi-stage, CPU PyTorch) and `docker/Dockerfile.serve` (slim CPU base, for stage 6).
- `docker-compose.yml` that starts MLflow server + Postgres + MinIO (S3-compatible) locally so your tracking stack is containerised too.
- `make train-docker` runs training inside the container with data mounted as a volume.

**You can claim it when:**
- [ ] Serving image < 1 GB, trains/serves from a clean machine with only Docker installed.
- [ ] You can explain (and have written in the README) why each layer is ordered as it is and what the cache hit rate is on a code-only change.
- [ ] `trivy` scan has no critical CVEs.

---

## Stage 4 — CI/CD  (Week 5)

**Skill claimed:** CI/CD

**Learn:**
- GitHub Actions: triggers, jobs, matrices, caching (`pip`, Docker layers via `buildx`), secrets, environments with manual approval, reusable workflows, artefacts.
- The ML-specific CI pyramid: lint → unit tests → data-contract tests → 1-epoch smoke training on a tiny subset → model quality gate (fail if macro-F1 < threshold on a fixed eval set) → build & push image.
- CD: on a git tag `v*`, build the serving image, push to GHCR (and to ECR in stage 5), and deploy to the cluster (stage 6 wires the last step).
- Branch protection with required checks.

**Build:**
- `.github/workflows/ci.yml` (on PR), `release.yml` (on tag), `nightly-train.yml` (scheduled small retrain + MLflow log; this feeds "continuous training").
- A `CML`-style PR comment that posts the confusion matrix and metrics diff versus `main`.

**You can claim it when:**
- [ ] A PR that lowers accuracy below threshold is automatically blocked.
- [ ] Pushing a tag produces a versioned image in a registry with zero manual steps.
- [ ] Green badge in README; you've debugged at least one flaky pipeline.

---

## Stage 5 — Cloud: AWS in depth, GCP and Azure by mapping  (Week 6–7)

**Skill claimed:** AWS, GCP, Azure

**Honest framing for the CV:** you will go deep on **one** cloud (AWS) and do a thin but real deployment on the other two. In interviews say exactly that: "primary AWS, working knowledge of GCP/Azure equivalents." That is what most candidates who list all three actually mean, and you'll be able to back it up.

**Learn (AWS, deeply):**
- IAM (users, roles, policies, least privilege, instance profiles), VPC basics, S3 (buckets, lifecycle, presigned URLs), EC2 (AMIs, spot, security groups, key pairs), ECR, EKS (stage 6), CloudWatch, cost explorer and budgets + alerts (set a US$50 alarm on day one).
- Infrastructure as code with **Terraform**: S3 bucket for DVC + MLflow artefacts, ECR repo, an IAM role for GitHub Actions via OIDC (no long-lived keys), a small EC2 instance for training and MLflow.
- Move MLflow server to a small EC2 with RDS Postgres backend (or keep in docker compose on EC2; both fine).

**Learn (GCP + Azure, mapping):**

| Concept | AWS | GCP | Azure |
|---|---|---|---|
| Object storage | S3 | GCS | Blob Storage |
| Container registry | ECR | Artifact Registry | ACR |
| Managed K8s | EKS | GKE | AKS |
| Compute VM | EC2 | Compute Engine | Virtual Machines |
| Managed ML | SageMaker | Vertex AI | Azure ML |
| Identity | IAM | IAM | Entra ID / RBAC |

For each of GCP and Azure, using free credits: push your serving image to their registry, deploy it to Cloud Run / Azure Container Apps (serverless containers, cheapest way to get a real URL), and run one managed-training job (Vertex AI custom job / Azure ML command job) with your training container. Terraform both.

**Build:**
- `infra/aws/*.tf`, `infra/gcp/*.tf`, `infra/azure/*.tf`.
- DVC remote = S3; CI uses OIDC role to pull data and push images to ECR.
- One training run on a CPU EC2 instance (e.g. `c6i.xlarge`) launched from your Terraform template, logged to the remote MLflow.

**You can claim it when:**
- [ ] `terraform apply` / `destroy` recreates all AWS infra from scratch, nothing was clicked in the console.
- [ ] The serving container is reachable at a public URL on all three clouds (screenshots in README, then torn down).
- [ ] You know your monthly bill to the dollar and have a budget alarm.

---

## Stage 6 — Model serving and Kubernetes  (Week 8–9)

**Skill claimed:** Kubernetes, "deploy" half of end-to-end

**Learn (deeply):**
- Serving: FastAPI app, ONNX Runtime session, input validation (Pydantic), batching, `/predict`, `/healthz`, `/readyz`, `/metrics`. Load test with `locust` or `k6`. Model loaded from S3/MLflow registry at startup by version, not baked into the image (understand the trade-off, bake it for edge).
- Kubernetes core: Pods, Deployments, ReplicaSets, Services (ClusterIP/NodePort/LoadBalancer), Ingress, ConfigMaps, Secrets, resource requests/limits, liveness/readiness probes, rolling updates and rollbacks, HPA (autoscale on CPU then on custom latency metric), namespaces, RBAC, `kubectl` fluency (`describe`, `logs`, `exec`, `port-forward`, `top`). Helm charts.
- Do it locally first with **kind** (you already have `kubectl`), then on **EKS** via Terraform. Optional but impressive: canary rollout with Argo Rollouts, or KServe as the "this is what managed model serving looks like" comparison.

**Build:**
- `serving/app.py`, `deploy/helm/orbiteye/` chart, `deploy/kind/` local setup.
- CD in `release.yml` runs `helm upgrade --install` against the cluster on tag.
- Load-test report: p50/p95/p99 latency at N replicas, HPA scaling event captured in a screenshot.

**You can claim it when:**
- [ ] A bad model version rolled out and you rolled it back with one command, and can explain what the ReplicaSets did.
- [ ] HPA scaled your pods from 1 to ≥ 4 under a load test and back down.
- [ ] You can debug a `CrashLoopBackOff` and a `Pending` pod without documentation.

---

## Stage 7 — Model monitoring  (Week 10)

**Skill claimed:** model monitoring

**Learn:**
- The three layers: system metrics (CPU/mem/GPU), service metrics (RPS, latency, error rate, via `prometheus-fastapi-instrumentator`), model metrics (prediction distribution, confidence histogram, input feature statistics, data drift, concept drift when labels arrive).
- Prometheus + Grafana on the cluster (kube-prometheus-stack Helm chart), Alertmanager rule → email/Slack.
- Drift with **Evidently** (or `alibi-detect`): log inference inputs/outputs to S3, nightly job computes drift report vs training reference, pushes a drift score to Prometheus.
- Structured JSON logging, request IDs, log aggregation with Loki.
- Close the loop: drift alert → triggers `nightly-train.yml` (continuous training) → new model to registry → CD.

**Build (this is where the satellite framing pays off):**
- Simulate drift realistically: serve images with synthetic haze / different season / different sensor gain and watch the drift score cross the threshold.
- Grafana dashboard JSON committed to repo; screenshot in README.
- A runbook `RUNBOOK.md`: what each alert means and what to do.

**You can claim it when:**
- [ ] A drift alert fired, you diagnosed it from the dashboard, and retraining was triggered automatically.
- [ ] You can list exactly which metrics you'd monitor for a new model and why.

---

## Stage 8 — Edge deployment: Raspberry Pi target, embedded and onboard-satellite constraints  (Week 11–12)

**Skill claimed:** edge deployment (embedded ARM devices, onboard/offline inference systems)

**Your situation:** you have no Raspberry Pi and no Jetson. That is fine. The target device is a
**Raspberry Pi 4/5 (ARM64, CPU only)**, and you build, test and exercise the *entire* edge lifecycle on
your Mac using ARM64 emulation. Every artefact is real and deployable; the only thing you skip is
plugging in the board. If you ever get a Pi (~A$120) it is a 30-minute job to run the same image on it.

**Why a Pi is a good stand-in for "embedded / onboard satellite":** CPU-only ARM, a few GB of RAM,
tight power budget, no GPU, no reliable network. Many CubeSat payload computers are exactly this class
of hardware, so the constraints you design for are the real ones.

**Learn (deeply):**
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

**Build (all in `edge/`, all runnable on the Mac):**
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

**You can claim it when:**
- [ ] Table: FP32 → INT8 → distilled INT8, with accuracy, latency, size, memory (native + arm64 emulated).
- [ ] The arm64 image boots in the emulated Pi OS as a `systemd` service, runs fully offline, survives
      `kill -9` via the watchdog, and passes the soak test.
- [ ] A signed model update was pulled during a simulated ground contact; a deliberately broken bundle
      was rejected and rolled back automatically.
- [ ] You can explain why INT8 ONNX Runtime on a CPU-only ARM board beats shipping PyTorch, with numbers.

**How to word it on the CV (truthfully):** "Edge deployment: built and CI-tested an offline, watchdog-supervised
INT8 ONNX inference service for ARM64/Raspberry Pi with signed OTA model updates, designed for onboard-satellite
constraints; validated under ARM64 emulation." In interviews say plainly that you did not have physical hardware
and that the image is ready to flash. That is a strong, honest answer; the Jetson mention comes off the CV line.

---

## Stage 9 — Capstone: make it claimable  (final week)

- README with architecture diagram, the three results tables (baseline, serving load test, edge optimisation), and a 3-minute demo video.
- `docs/` with `DESIGN.md`, `RUNBOOK.md`, `ADR/` (architecture decision records, 5–8 short ones: "why ONNX Runtime not TorchServe", "why INT8 on the edge", etc.). Interviewers love ADRs.
- Write one blog post / LinkedIn article per major stage. Link them from the CV.
- Tear down all cloud resources; keep Terraform so you can recreate in a demo.

**CV bullets you'll be able to write truthfully:**
- Built an end-to-end MLOps pipeline for satellite land-cover classification: DVC + MLflow experiment tracking, Dockerised training/serving, GitHub Actions CI/CD with model quality gates, Helm-deployed FastAPI/ONNX inference on Kubernetes (EKS) with HPA, Prometheus/Grafana monitoring and Evidently drift-triggered retraining.
- Provisioned infrastructure with Terraform on AWS (S3, ECR, EKS, EC2, IAM OIDC), with equivalent container deployments on GCP Cloud Run and Azure Container Apps.
- Built and CI-tested an offline, watchdog-supervised INT8 ONNX inference service for ARM64/Raspberry Pi with signed OTA model updates, designed for onboard-satellite constraints (power, no-network, A/B rollback); validated under ARM64 emulation.

---

## Weekly cadence and how to study each stage

1. **Concept day (2 h):** read the official docs for the tool (not blog posts), write a one-page summary in `docs/notes/<stage>.md` in your own words.
2. **Build days (5–6 h):** implement the stage's artefacts. Use Claude Code to unblock, but type the core pieces yourself and be able to rewrite them from scratch.
3. **Break-it day (1–2 h):** deliberately break it (kill a pod, corrupt a config, revoke a permission, delete a layer) and fix it. Interview questions are almost always "what happens when X goes wrong".
4. **Checklist:** tick every "You can claim it when…" box before starting the next stage. Commit the ticked list.

## Primary resources (official-first)

- Linux/Git: *The Linux Command Line* (Shotts, free PDF), Pro Git (free), MIT "Missing Semester".
- DVC docs "Get Started", MLflow docs "Tracking" + "Model Registry".
- Docker docs "Get started" + "Best practices for Dockerfiles"; *Docker Deep Dive* (Poulton).
- GitHub Actions docs; CML (iterative.ai) docs.
- AWS Skill Builder free tier; Terraform "Get Started – AWS"; GCP/Azure free-tier docs for Cloud Run / Container Apps.
- Kubernetes docs "Concepts" + "Tasks"; *Kubernetes Up & Running*; kind docs; Helm docs.
- Prometheus/Grafana docs; Evidently docs; Google SRE book chapter on monitoring.
- ONNX Runtime quantisation docs; Docker buildx multi-platform docs; Raspberry Pi OS headless setup docs; QEMU aarch64 Pi boot guides; systemd watchdog docs; ESA Φ-sat-1 and OPS-SAT papers.
- Whole-lifecycle framing: *Designing Machine Learning Systems* (Huyen), Made With ML MLOps course.
