# Changes from upstream OpenRLHF (file-level)

This document records the **intentional** differences between:

- **Upstream base:** OpenRLHF `openrlhf/` at commit  
  `f372a2d41e26c3c47a0f6653fb94c31f5c257942` (describe: `v0.9.0-3-gf372a2d`)
- **This repo:** `openrlhf/`

Primary goals of this log:
- **auditability**: quickly answer “what upstream files changed and why?”
- **reproducibility**: document integration points that affect behavior
- **rebasing**: reduce conflict surface when updating upstream

> Scope note: upstream OpenRLHF includes many additional CLIs/trainers (SFT/RM/DPO/...) that are
> intentionally not vendored in this paper code release. This repo focuses on the subset needed
> for the C3 paper pipeline and PPO/Ray training/eval integration.

---

## A) High-level behavioral differences (what actually changes)

### 1) C3 task/config integration
Upstream is extended to accept C3 task/role configs (via `c3.task.config` and `configs/tasks/*.yaml`)
and to propagate task metadata into training/eval.

**Primary touchpoint:**  
- `openrlhf/cli/train_ppo_ray.py` (major extension)

### 2) Multi-agent rollout / experience fields
Experience making and training loop are extended to support multi-agent rollout structures used by C3
(e.g., transcript nodes, leaf markers, depth, teammate context).

**Primary touchpoints:**  
- `openrlhf/trainer/ppo_utils/experience_maker.py`  
- `openrlhf/trainer/ppo_trainer.py` (+ async variant)

### 3) Reproducibility run artifacts / metadata
A small run-metadata utility is added so runs can dump reproducibility metadata (config snapshot, command line,
environment hints).

**Primary touchpoint:**  
- `openrlhf/utils/run_metadata.py` (new)

---

## B) File-level diff summary (relative to upstream `openrlhf/`)

Counts (within `openrlhf/`):
- **Added:** 4 files
- **Modified:** 23 files
- **Removed (not vendored):** 20 files

> These counts and lists compare **non-blank, non-provenance lines**. 39 of the
> 44 vendored `.py` files start with the three-line C3 provenance comment
> (`# Derived from OpenRLHF ...`, `# Modified by the C3 authors ...`,
> `# See docs/40_upstream.md ...`), and adding it also moved the blank line next
> to it in a few files, so a raw byte comparison against upstream marks all 39 as
> modified and does not reproduce the numbers above. The regeneration script
> below skips both kinds of line before hashing; the counts here were last
> regenerated with it against upstream at the pinned commit.
>
> These lists are reproducible. See the "How to regenerate this summary" section below.

### B.1 Added files

- `openrlhf/cli/train_ppo_ray_tooling.py`  
  Split-out CLI tooling/helpers used by `train_ppo_ray.py` (argument normalization, ray init, logging helpers).

- `openrlhf/trainer/ppo_trainer_plugins.py`  
  Minimal plugin hooks used by the C3 integration to keep upstream modifications localized.

- `openrlhf/trainer/ppo_utils/dynamic_filtering.py`  
  Dynamic filtering support extracted/extended for the C3 pipeline.

- `openrlhf/utils/run_metadata.py`  
  Reproducibility helpers to initialize/write run artifacts and metadata.

### B.2 Modified files

**Core integration points (most important):**
- `openrlhf/cli/train_ppo_ray.py`  
  Major extension: C3 task loading integration + additional planning/normalization utilities.

- `openrlhf/trainer/ppo_trainer.py`  
  C3-compatible experience fields + integration hooks.

- `openrlhf/trainer/ppo_utils/experience_maker.py`  
  C3 rollout field propagation (multi-agent structures).

- `openrlhf/trainer/ray/ppo_critic.py`  
  Extended critic actor variants used by C3 pipeline.

**Packaging only (no behavioral difference when flash-attn is installed):**
- `openrlhf/models/ring_attn_utils.py`  
  Upstream imports `flash_attn.bert_padding` and `flash_attn.utils.distributed` at module level,
  which makes flash-attn a hard requirement of `import openrlhf.models` and therefore of the
  training CLI, the evaluation sweep and `scripts/30_smoke/smoke.sh --tier gpu`. flash-attn ships
  no wheels (it builds from source against a CUDA toolchain), so it cannot be pinned in the lock
  files that CPU users install. The two imports are wrapped in a `try`/`except ImportError` that
  binds the same five names (`index_first_axis`, `pad_input`, `rearrange`, `unpad_input`,
  `all_gather`) to `None` when the package is absent, and the only two functions that use them,
  `unpad_and_slice_tensor` and `gather_and_pad_tensor`, now open with a `_require_flash_attn()`
  call that raises `RuntimeError` naming ring attention and the install command. With flash-attn
  installed the imported names, the call sites and every code path are unchanged.

**Complete modified-file list (for audit):**
- `openrlhf/cli/train_ppo_ray.py`
- `openrlhf/datasets/__init__.py`
- `openrlhf/datasets/prompts_dataset.py`
- `openrlhf/models/actor.py`
- `openrlhf/models/loss.py`
- `openrlhf/models/model.py`
- `openrlhf/models/ring_attn_utils.py`
- `openrlhf/models/utils.py`
- `openrlhf/trainer/ppo_trainer.py`
- `openrlhf/trainer/ppo_trainer_async.py`
- `openrlhf/trainer/ppo_utils/experience_maker.py`
- `openrlhf/trainer/ppo_utils/replay_buffer.py`
- `openrlhf/trainer/ray/launcher.py`
- `openrlhf/trainer/ray/ppo_actor.py`
- `openrlhf/trainer/ray/ppo_critic.py`
- `openrlhf/trainer/ray/vllm_engine.py`
- `openrlhf/trainer/ray/vllm_engine_async.py`
- `openrlhf/trainer/ray/vllm_worker_wrap.py`
- `openrlhf/utils/deepspeed/deepspeed.py`
- `openrlhf/utils/distributed_sampler.py`
- `openrlhf/utils/logging_utils.py`
- `openrlhf/utils/remote_rm_utils.py`
- `openrlhf/utils/utils.py`

### B.3 Removed (not vendored) files

The following upstream files exist in OpenRLHF `openrlhf/` at the pinned commit but are intentionally
not included in this release to keep the paper codebase focused:

**Removed CLIs:**
- `openrlhf/cli/batch_inference.py`
- `openrlhf/cli/interactive_chat.py`
- `openrlhf/cli/lora_combiner.py`
- `openrlhf/cli/serve_rm.py`
- `openrlhf/cli/train_dpo.py`
- `openrlhf/cli/train_kd.py`
- `openrlhf/cli/train_kto.py`
- `openrlhf/cli/train_prm.py`
- `openrlhf/cli/train_rm.py`
- `openrlhf/cli/train_sft.py`

**Removed datasets:**
- `openrlhf/datasets/process_reward_dataset.py`
- `openrlhf/datasets/reward_dataset.py`
- `openrlhf/datasets/sft_dataset.py`
- `openrlhf/datasets/unpaired_preference_dataset.py`

**Removed trainers:**
- `openrlhf/trainer/dpo_trainer.py`
- `openrlhf/trainer/kd_trainer.py`
- `openrlhf/trainer/kto_trainer.py`
- `openrlhf/trainer/prm_trainer.py`
- `openrlhf/trainer/rm_trainer.py`
- `openrlhf/trainer/sft_trainer.py`

---

## How to regenerate this summary (Added / Modified / Removed lists)

This section describes how to reproduce Section B.

### 1) Prepare upstream at the pinned commit

```bash
git clone https://github.com/OpenRLHF/OpenRLHF.git <path-to-OpenRLHF>
cd <path-to-OpenRLHF>
git checkout f372a2d41e26c3c47a0f6653fb94c31f5c257942
```

### 2) Compute the lists (recommended: content-hash based)

From your C3 repo root:

```bash
python - <<'PY'
from pathlib import Path
import hashlib

c3 = Path("openrlhf")
up = Path("<path-to-OpenRLHF>/openrlhf")

# 39 of the 44 vendored files start with the three-line C3 provenance comment,
# and adding it also moved the blank line next to it in a few files. Hash the
# non-blank, non-provenance lines, or every one of those files reports as
# modified.
PROVENANCE = (
    b"# Derived from OpenRLHF",
    b"# Modified by the C3 authors",
    b"# See docs/40_upstream.md",
)

def files(root: Path):
    return sorted([p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()])

def sha256(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for line in f:
            if line.startswith(PROVENANCE) or not line.strip():
                continue
            h.update(line)
    return h.hexdigest()

c3_files = set(files(c3))
up_files = set(files(up))

added = sorted(c3_files - up_files)
removed = sorted(up_files - c3_files)
common = sorted(c3_files & up_files)

modified = []
for rel in common:
    if sha256(c3 / rel) != sha256(up / rel):
        modified.append(rel)

print("Added:", len(added))
print("Modified:", len(modified))
print("Removed (not vendored):", len(removed))

print("\n# Added")
[print(x) for x in added]
print("\n# Modified")
[print(x) for x in modified]
print("\n# Removed (not vendored)")
[print(x) for x in removed]
PY
```

---

## C) Notes for rebasing

When updating the upstream anchor:

1. Update `docs/40_upstream.md` with the new commit (and describe string).
2. Re-diff `openrlhf/` against upstream `openrlhf/` and refresh Section B lists:
   - Added / Modified / Removed
3. Ensure integration touchpoints remain minimal and stable:
   - `cli/train_ppo_ray.py`
   - `trainer/ppo_trainer.py`
   - `trainer/ppo_utils/experience_maker.py`
   - `trainer/ray/ppo_critic.py`
   - `utils/run_metadata.py`
4. Run the release gate: `bash scripts/90_audit/pre_release.sh`
