# LinkRadius H200 experiment handoff

Last updated: 2026-08-19 (Asia/Singapore)

## Instruction for the next Codex session

This chat is not available on the new cloud platform. After cloning the source,
start a new Codex session there with this instruction:

> Read `handoff.md` completely, inspect the checkout and hardware, and continue
> the LinkRadius H200 run from the exact checkpoint described here. Do not reuse
> artifacts from an older output root, change experiment source after starting,
> install FLA, or skip an authenticated stage.

The next session must first report:

- the Git revision and LinkRadius `source_hash`;
- Python, PyTorch, CUDA, and Transformers versions;
- the number, model, total memory, and free memory of visible GPUs;
- whether the platform uses Slurm or provides a dedicated GPU instance.

Do not start model execution until those checks agree with this handoff.

## Non-negotiable scientific status

The preregistered `sequential_light` baseline under
`outputs/linkradius-v3` failed the mandatory Phase 2 smoke gate. After the
strict parser correction, its complete 40-row validation screen contained only
3 dual-correct rows; the required range was 10--20. That remains a failed
baseline and must not be relabelled as a pass.

The scaled experiment is an exploratory stronger-model follow-up:

```text
STYLE=sequential_scaled
METHOD=ours_recursive
DATASETS=gpqa
ROUNDS=2
SEEDS=42
BATCH_SIZE=1
LATENT_LENGTH=48
```

It uses a Gemma-3 4B planner, Llama-3.2 3B critic, Qwen-3.5 4B solver, and the
released scaled outer adapters. Results from this condition do not replace the
failed light baseline.

## Exact source state to transfer

The minimum implementation revision containing the current fixes is:

```text
implementation commit: 174d773e6bb8a50dd0f18bf8951acb4d4ae92d3a
expected LinkRadius source_hash: f313855756bbc3febedec70ec472fc618cae8addf1b0a73dc4a8b59cc2315229
strict parser: linkradius_choice_v2
```

Commit this updated `handoff.md` and push the private working branch. The
resulting Git commit will be newer than the implementation commit above, but
the expected `source_hash` must remain exactly the same because `handoff.md` is
outside the result-affecting source tree.

The relevant implementation already includes:

- three-role model placement on logical `cuda:0`, `cuda:1`, and `cuda:2`;
- source-role placement for learned outer adapters;
- authenticated `RELAY_TRANSFER_MODE=cpu_staged` transfers;
- source CUDA -> CPU float32 -> destination CUDA consumer-dtype relays without
  detaching autograd;
- edge-level requested/realized transfer provenance and forward-finiteness
  diagnostics;
- sequential one-candidate terminal margin backpropagation to bound scorer
  memory;
- strict artifact/completion/source authentication throughout the workflow.

At this source revision, the complete local LinkRadius suite ran 163 tests with
no failures (59 PyTorch-dependent skips in the local lightweight environment),
all Python files compiled, all seven launchers passed `bash -n`, the canonical
discovery grid contained 20 tasks, and `git diff --check` passed.

On the new platform, verify the checkout before doing anything else:

```bash
git status --short --untracked-files=all RecursiveMAS experiments/linkradius

python - <<'PY'
from pathlib import Path
from RecursiveMAS.inference_utils.linkradius import STRICT_CHOICE_VERSION
from experiments.linkradius.io_utils import source_hash

expected = "f313855756bbc3febedec70ec472fc618cae8addf1b0a73dc4a8b59cc2315229"
actual = source_hash(Path.cwd())
print({
    "strict_parser": STRICT_CHOICE_VERSION,
    "source_hash": actual,
    "expected": expected,
    "match": actual == expected,
})
raise SystemExit(actual != expected)
PY
```

The scoped `git status` must be empty and the source hash must match. Hidden
Jupyter checkpoint files inside `experiments/linkradius` must not exist.

## Why the old 80 GB run stopped

The old-platform root is:

```text
outputs/linkradius-scaled-mp-membounded-v4
```

It is halted and must not be resumed or copied into the H200 root. Reported
progress on that root was:

- split completed;
- all 20 discovery shards completed and the full authentication/numerical
  audit passed;
- execution freeze completed;
- clean capture completed;
- replay tasks 0--9 completed;
- probe grid and probe tasks 0--1 completed;
- gradient task 1 failed with a true-capacity OOM;
- engineering validation and its gate were never completed.

The two gradient manifests recorded:

```text
array 0: job 892082, node02, CUDA_VISIBLE_DEVICES=1,2,3, edge c2s@1
array 1: job 892099, node04, CUDA_VISIBLE_DEVICES=0,3,4, edge p2c@0
```

Array 1 is the difficult early-edge autograd replay. Immediately before it
loaded, physical GPUs 0, 3, and 4 were all at 0 MiB. Logical `cuda:2` therefore
mapped to an empty physical GPU 4. The process itself then allocated 78.57 GiB,
used 79.23 GiB in total, had only 139 MiB reserved-but-unallocated, and failed
while requesting another 20 MiB. This rules out another user's process,
allocator fragmentation, and incorrect GPU selection. Pinning or requesting an
exclusive A100 would not solve it.

No activation-checkpointing or saved-tensor CPU-offload change has been made.
The next run deliberately tests the unchanged, authenticated implementation on
larger-memory H200 devices. If early-edge gradient task 1 still OOMs on H200,
stop. The next code change should then be an authenticated autograd
checkpoint/offload mode and another fresh root.

Earlier roots (`linkradius-scaled-v1`, FLA roots, `membounded-v1`, v2, and v3)
are also historical failures or diagnostics. Never mix their artifacts.

## Environment on the new platform

Use the ordinary environment without FLA. The pinned direct dependencies are
in `RecursiveMAS/requirements.txt`; the old working environment used Python
3.10, PyTorch 2.9.0 with CUDA 12.8, Transformers 5.3.0, and Triton 3.5.0.

If the environment is not already available:

```bash
conda create -n recursivemas python=3.10 -y
conda activate recursivemas
python -m pip install --upgrade pip
python -m pip install -r RecursiveMAS/requirements.txt
python -m pip check
```

Do not install `fla-core`, `flash-linear-attention`, or `causal-conv1d` for this
root. The fallback implementation is the condition used by all authenticated
v4 stages.

Inspect the environment and visible devices inside a real GPU allocation:

```bash
python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import json
import torch

packages = {}
for name in ("transformers", "triton", "flash-linear-attention", "fla-core"):
    try:
        packages[name] = version(name)
    except PackageNotFoundError:
        packages[name] = None

gpus = []
for index in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(index)
    properties = torch.cuda.get_device_properties(index)
    gpus.append({
        "logical_index": index,
        "name": properties.name,
        "total_gib": round(total / 2**30, 2),
        "free_gib": round(free / 2**30, 2),
    })

result = {
    "python_torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "packages": packages,
    "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
    "gpus": gpus,
}
print(json.dumps(result, indent=2))

if len(gpus) < 3:
    raise SystemExit("the prescribed model-parallel topology requires 3 visible GPUs")
if any("H200" not in gpu["name"].upper() for gpu in gpus[:3]):
    raise SystemExit("the first three logical devices are not all H200 GPUs")
if packages["flash-linear-attention"] or packages["fla-core"]:
    raise SystemExit("FLA packages are installed; use a clean fallback environment")
PY
```

The prescribed run requires three simultaneously visible H200 GPUs. If the new
platform exposes only one H200, stop and ask the next Codex session to evaluate
a separate single-device condition; do not silently change all role devices to
`cuda:0` under the root defined below.

Run the real two-GPU differentiable relay regression inside the allocation:

```bash
python -m unittest \
  experiments.linkradius.tests.test_model_parallelism.SystemLoaderPlacementTests.test_cpu_staged_cross_gpu_relay_preserves_values_and_gradients
```

It must pass rather than skip.

## Fresh H200 experiment identity

Use a completely new root on the new platform:

```bash
export PYTHON_BIN="$CONDA_PREFIX/bin/python"
export OUT_ROOT="$PWD/outputs/linkradius-scaled-mp-membounded-h200-v1"
export STYLE=sequential_scaled
export METHOD=ours_recursive
export DATASETS=gpqa
export ROUNDS=2
export SEEDS=42
export BATCH_SIZE=1
export LATENT_LENGTH=48
export DEVICE=cuda:0
export PLANNER_DEVICE=cuda:0
export CRITIC_DEVICE=cuda:1
export SOLVER_DEVICE=cuda:2
export RELAY_TRANSFER_MODE=cpu_staged
export DISCOVERY_BATCHES=20
export PROBE_RADII="1e-3 3e-3"
export PROBE_SEEDS=101
export K=8
export GPU_LIST=

mkdir -p logs
```

Export exactly these values for every command, including CPU stages. CPU stages
reconstruct upstream task identities, so omitting the role map or transfer mode
makes valid GPU outputs appear missing.

The role device numbers are logical indices inside the platform-provided
`CUDA_VISIBLE_DEVICES`. Never use `GPU_LIST` with model parallelism. On Slurm,
do not manually replace `CUDA_VISIBLE_DEVICES`; request three GPUs and let the
scheduler provide the mask.

## Launching on the new platform

### Slurm platform

Set the site-specific partition and optional node. Do not edit `#SBATCH` lines
inside the tracked launcher:

```bash
export H200_PARTITION='<replace-with-H200-partition>'
export H200_NODE=''

SBATCH_H200=(-p "$H200_PARTITION" --ntasks=1 --gres=gpu:3)
if [[ -n "$H200_NODE" ]]; then
  SBATCH_H200+=(-w "$H200_NODE")
fi
```

If the site requires a typed GRES such as `--gres=gpu:h200:3`, replace only
that array element. Each experiment array element is one process using three
GPUs. With only three GPUs available, use `%1` array throttling.

### Dedicated non-Slurm instance

Only on a dedicated instance where GPUs 0, 1, and 2 are assigned to this user:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2

run_gpu_task() {
  local stage="$1"
  local index="$2"
  SLURM_ARRAY_TASK_ID="$index" OVERWRITE=1 LR_STAGE="$stage" \
    bash experiments/linkradius/run_linkradius_engineering.sh
}
```

Do not set `SLURM_JOB_ID` manually. For multiple tasks, call `run_gpu_task`
sequentially unless the instance has another independent set of three GPUs.

## Exact Phase 1 run order

The commands below use Slurm. On a dedicated instance, replace each `sbatch`
command with `run_gpu_task STAGE INDEX` for every listed index.

### 1. Split

```bash
OVERWRITE=1 LR_STAGE=split \
bash experiments/linkradius/run_linkradius_engineering.sh
```

### 2. Discovery task 0 only

```bash
OVERWRITE=1 LR_STAGE=discover \
sbatch "${SBATCH_H200[@]}" --array=0 \
  experiments/linkradius/run_linkradius_engineering.sh
```

After it finishes, require one current authenticated completion, finite relays
and scorer, and the realized CPU-staged mode:

```bash
python - <<'PY'
from pathlib import Path

from experiments.linkradius.io_utils import load_json, load_jsonl, source_hash, verify_completion

repo = Path.cwd().resolve()
root = Path("outputs/linkradius-scaled-mp-membounded-h200-v1/engineering")
current = source_hash(repo)
matches = []

for marker in root.rglob(".complete.json"):
    task_dir = marker.parent
    manifest = load_json(task_dir / "manifest.json")
    task = manifest.get("task", {})
    if task.get("stage") != "discover" or task.get("array_index") != 0:
        continue
    completion = verify_completion(task_dir)
    if completion.get("source_hash") == current:
        matches.append((task_dir, manifest))

if len(matches) != 1:
    raise SystemExit(f"expected one current discovery task 0, found {len(matches)}")

task_dir, manifest = matches[0]
rows = [
    row for row in load_jsonl(task_dir / "screening_rows.jsonl")
    if row.get("record_type") == "sample"
]
if len(rows) != 1:
    raise SystemExit(f"expected one sample row, found {len(rows)}")

diagnostics = rows[0]["forward_finiteness"]
expected_edges = {"p2c@0", "c2s@0", "s2p@0", "p2c@1", "c2s@1"}
if set(diagnostics.get("edges", {})) != expected_edges:
    raise SystemExit("unexpected relay edge set")
if not diagnostics.get("all_observed_numeric_outputs_finite"):
    raise SystemExit(f"non-finite task 0: {diagnostics.get('first_nonfinite')}")
for edge_id, edge in diagnostics["edges"].items():
    if edge.get("requested_transfer_mode") != "cpu_staged":
        raise SystemExit(f"{edge_id}: wrong requested transfer mode")
    if edge.get("realized_transfer_mode") != "cpu_float32_staged_cross_device":
        raise SystemExit(f"{edge_id}: wrong realized transfer mode")

print({
    "passed": True,
    "task_dir": str(task_dir),
    "node": manifest.get("scheduler_environment", {}).get("slurm_job_nodelist"),
    "cuda_visible_devices": manifest.get("scheduler_environment", {}).get("cuda_visible_devices"),
    "dual_correct": rows[0].get("dual_correct"),
})
PY
```

`dual_correct: false` for task 0 is allowed; discovery searches all 20 rows.

### 3. Remaining discovery tasks

```bash
OVERWRITE=1 LR_STAGE=discover \
sbatch "${SBATCH_H200[@]}" --array=1-19%1 \
  experiments/linkradius/run_linkradius_engineering.sh
```

After all tasks finish, run this compact full audit:

```bash
python - <<'PY'
from collections import Counter
from pathlib import Path

from experiments.linkradius.io_utils import load_json, load_jsonl, source_hash, verify_completion
from experiments.linkradius.select_clean_correct import classify_screening_row

repo = Path.cwd().resolve()
root = Path("outputs/linkradius-scaled-mp-membounded-h200-v1/engineering")
current = source_hash(repo)
seen = {}
reasons = Counter()
dual_correct = 0

for marker in root.rglob(".complete.json"):
    task_dir = marker.parent
    manifest_path = task_dir / "manifest.json"
    if not manifest_path.is_file():
        continue
    manifest = load_json(manifest_path)
    task = manifest.get("task", {})
    if task.get("stage") != "discover":
        continue
    completion = verify_completion(task_dir)
    if completion.get("source_hash") != current:
        continue
    index = int(task["array_index"])
    if index in seen:
        raise SystemExit(f"duplicate current discovery index {index}")
    rows = [
        row for row in load_jsonl(task_dir / "screening_rows.jsonl")
        if row.get("record_type") == "sample"
    ]
    if len(rows) != 1:
        raise SystemExit(f"task {index}: expected one sample row")
    diagnostics = rows[0]["forward_finiteness"]
    if not diagnostics.get("all_observed_numeric_outputs_finite"):
        raise SystemExit(f"task {index}: non-finite outputs")
    for edge_id, edge in diagnostics.get("edges", {}).items():
        if edge.get("realized_transfer_mode") != "cpu_float32_staged_cross_device":
            raise SystemExit(f"task {index}/{edge_id}: wrong transfer mode")
    passed, reason = classify_screening_row(rows[0])
    dual_correct += int(passed)
    if not passed:
        reasons[reason] += 1
    seen[index] = str(task_dir)

missing = sorted(set(range(20)) - set(seen))
if missing:
    raise SystemExit(f"missing discovery indices: {missing}")
if dual_correct < 1:
    raise SystemExit("no dual-correct discovery row; freeze cannot proceed")

print({
    "passed": True,
    "completed_indices": sorted(seen),
    "dual_correct_count": dual_correct,
    "exclusions": dict(sorted(reasons.items())),
})
PY
```

### 4. Freeze and clean capture

```bash
OVERWRITE=1 LR_STAGE=freeze_execution \
bash experiments/linkradius/run_linkradius_engineering.sh

OVERWRITE=1 LR_STAGE=clean \
sbatch "${SBATCH_H200[@]}" --array=0 \
  experiments/linkradius/run_linkradius_engineering.sh
```

The freeze is CPU-only. Clean is a one-element three-GPU stage.

### 5. Replay

```bash
OVERWRITE=1 LR_STAGE=replay \
sbatch "${SBATCH_H200[@]}" --array=0-9%1 \
  experiments/linkradius/run_linkradius_engineering.sh
```

Require all ten authenticated completions before continuing.

### 6. Probe

```bash
GRID_TARGET_STAGE=probe LR_STAGE=grid \
bash experiments/linkradius/run_linkradius_engineering.sh
```

The grid must report `total_tasks=2` and `max_array_index=1`. Then submit:

```bash
OVERWRITE=1 LR_STAGE=probe \
sbatch "${SBATCH_H200[@]}" --array=0-1%1 \
  experiments/linkradius/run_linkradius_engineering.sh
```

### 7. Gradient checks, one at a time

Run the terminal check first:

```bash
OVERWRITE=1 LR_STAGE=gradient \
sbatch "${SBATCH_H200[@]}" --array=0 \
  experiments/linkradius/run_linkradius_engineering.sh
```

Only after array 0 has an authenticated completion, run the decisive early-edge
check:

```bash
OVERWRITE=1 LR_STAGE=gradient \
sbatch "${SBATCH_H200[@]}" --array=1 \
  experiments/linkradius/run_linkradius_engineering.sh
```

Capture the array-1 peak-memory log. If it OOMs on an otherwise empty H200, do
not retry on another H200 and do not alter the environment within this root.
Stop and implement a source-hashed activation-checkpoint/saved-tensor-offload
mode under a new output root.

### 8. Legacy equivalence and engineering validation

After both gradient tasks pass, locate the unique clean trajectory:

```bash
python - <<'PY'
from pathlib import Path
import torch

from experiments.linkradius.io_utils import source_hash, verify_completion

repo = Path.cwd().resolve()
root = repo / "outputs/linkradius-scaled-mp-membounded-h200-v1/engineering/gpqa/R2/validation/clean"
current = source_hash(repo)
candidates = []

for path in root.rglob("clean_trajectory.pt"):
    try:
        completion = verify_completion(path.parent)
    except Exception:
        continue
    declared = {item.get("path") for item in completion.get("artifacts", [])}
    if completion.get("source_hash") == current and path.name in declared:
        candidates.append(path.resolve())

if len(candidates) != 1:
    raise SystemExit(f"expected one authenticated clean trajectory: {candidates}")

trajectory = torch.load(candidates[0], map_location="cpu", weights_only=False)
print(f"export CLEAN_TRAJECTORY={candidates[0]}")
print(f"export RAW_INDEX={int(trajectory.raw_indices[0])}")
PY
```

Copy the two printed exports. Run the release path inside a one-GPU H200
allocation in the same environment:

```bash
export LEGACY_RESULTS="$OUT_ROOT/legacy_release.jsonl"
export LEGACY_TRACE="$OUT_ROOT/legacy_release_trace.pt"

python RecursiveMAS/run.py \
  --style sequential_scaled \
  --dataset gpqa --dataset_split train \
  --method ours_recursive --num_recursive_rounds 2 \
  --num_samples 1 --sample_indices "$RAW_INDEX" \
  --batch_size 1 --latent_length 48 \
  --seed 42 --deterministic 1 --device cuda:0 \
  --result_jsonl "$LEGACY_RESULTS" \
  --lc_trace_path "$LEGACY_TRACE" \
  --lc_trace_sites p2c,c2s,s2p \
  --lc_trace_rounds 0,1 --lc_trace_dtype float32

python -m experiments.linkradius.compare_legacy_equivalence \
  --trajectory "$CLEAN_TRAJECTORY" \
  --legacy-trace "$LEGACY_TRACE" \
  --legacy-results "$LEGACY_RESULTS" \
  --output "$OUT_ROOT/legacy_equivalence.json"
```

If the comparator reports `"passed": true`, run the CPU validator with the
same exported topology and transfer mode:

```bash
export LEGACY_EQUIVALENCE="$OUT_ROOT/legacy_equivalence.json"

OVERWRITE=1 LR_STAGE=validate \
bash experiments/linkradius/run_linkradius_engineering.sh
```

Do not start Phase 2 unless this produces a passed, authenticated
`$OUT_ROOT/engineering_gate.json`.

## Authentication and safety rules

- A file existing is not proof of success. Require `.complete.json`, verify its
  hashes, and require the exact expected array indices.
- Never copy a clean trajectory, split, gate, or completion from another root.
- Never mix Python environments within one output root.
- Do not modify `RecursiveMAS` or `experiments/linkradius` after `split`.
- Scheduler partition, node, and physical GPU allocation may change safely;
  logical role devices and `RELAY_TRANSFER_MODE` may not.
- `nvidia-smi` can show every physical GPU even when a job is masked. Use the
  task manifest's `scheduler_environment.cuda_visible_devices` and PyTorch's
  logical device accounting.
- `GPU_LIST="0 1 2"` does not combine memory. It means round-robin
  one-GPU-per-array-task routing and is deliberately rejected with role maps.
- Do not weaken JSON finiteness, scorer agreement, parser, provenance, or gate
  checks to make a stage pass.
- A node change does not alter the source hash. A tracked source edit does.

Expected known-good checkpoint identities from the old resolver were:

```text
model_hash:   2c18218f3fcb202e23746abf17b438cfebdbc6ce735baf407c2f2d62a494191b
adapter_hash: 245f624ddcab7747b901a42c50d8a642bebc3cac4a5907807d16e19237e24341
```

Compare these after the new clean capture. A mismatch requires investigation;
do not continue to replay or gradient validation.

## Moving the work to the other cloud

1. Commit this handoff and push the current private branch.
2. Clone or pull that exact branch on the H200 platform.
3. Verify the expected LinkRadius source hash before installing/running.
4. Keep model caches and `outputs/` on persistent cloud storage, not in Git.
5. Do not transfer old experiment artifacts to populate the fresh H200 root.
6. Update this file with job IDs, completed/failed indices, GPU masks, and the
   exact next command before switching platforms again.
7. Use `tmux` or the platform's persistent job mechanism for long commands.

If source must change after the H200 run begins, stop all jobs, update this
handoff, choose a new output-root name, and restart from `split`.
