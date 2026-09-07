# Network Mirrors

Every download in this repository goes to the official source by default. Nothing
here points at a mirror unless you opt in, and the repository never hard-codes a
mirror host. If your network reaches PyPI, Hugging Face and GitHub directly, you
can skip this page.

The knobs below are plain environment variables. `scripts/_lib/mirrors.sh`
validates them, prints which ones are active, and is sourced by the data and
model preparation scripts, so every run states in its log where its bytes came
from.

## The knobs

| Variable | Default | What it changes |
|---|---|---|
| `HF_ENDPOINT` | `https://huggingface.co` | The Hugging Face Hub endpoint used by `datasets` and `huggingface_hub`, so it covers dataset and model downloads. |
| `GITHUB_MIRROR_PREFIX` | empty | A prefix prepended to a full `github.com` URL. Used to fetch the MBPP+ release asset that EvalPlus downloads from a GitHub release. Must end with a slash. |
| `PIP_INDEX_URL` | PyPI | The package index pip installs from. |
| `PIP_EXTRA_INDEX_URL` | empty | An additional index, for example a mirror of the PyTorch wheel index. |

## Examples

```bash
# Hugging Face through a mirror
export HF_ENDPOINT=https://hf-mirror.com

# GitHub release assets through a proxy (note the trailing slash)
export GITHUB_MIRROR_PREFIX=https://ghproxy.net/

# pip through a mirror
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

bash scripts/10_data/prepare_all.sh --out_dir data --strict 1
```

The proxy hosts above are examples, not endorsements. Pick one your network can
reach and that you trust: a mirror sees every request you send through it.

## What each knob is for

### `HF_ENDPOINT`

`scripts/10_data/prepare_math.py` and `prepare_code.py` download their sources
from the Hugging Face Hub at pinned commit revisions. The revision pin means a
mirror cannot silently give you different data: `--strict 1` recomputes the
SHA256 of every produced file and compares it against `configs/data_manifest.yaml`.

If a run fails against a mirror, `scripts/10_data/prepare_all.sh` retries once
against the official endpoint. Set `FALLBACK_TO_OFFICIAL=0` to disable that retry.

### `GITHUB_MIRROR_PREFIX`

MBPP+ is not on the Hugging Face Hub in the form this repository pins. EvalPlus
downloads it from a GitHub release and caches the gunzipped file under its own
cache directory. When `GITHUB_MIRROR_PREFIX` is set, `prepare_all.sh` fetches
that release asset through the prefix and writes it to the exact path EvalPlus
expects, so EvalPlus itself never contacts `github.com`. The result is verified
by the same SHA256 pin as every other dataset.

A prefix is prepended verbatim, so it must end with a slash:

```
https://ghproxy.net/ + https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz
```

### `PIP_INDEX_URL` and `PIP_EXTRA_INDEX_URL`

`requirements/cpu.lock.txt` needs the official PyTorch CPU wheel index
(`https://download.pytorch.org/whl/cpu`), which some networks cannot reach.
Several public mirrors republish that index and work as a drop-in replacement:

```bash
python -m pip install \
  --extra-index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/ \
  -r requirements/cpu.lock.txt
```

The lock file always ships the official URL. A mirror stays a local choice.

## Verifying that a mirror did not change anything

Data integrity does not depend on trusting the mirror:

```bash
bash scripts/10_data/prepare_all.sh --out_dir data --strict 1
```

Strict mode fails on any SHA256 mismatch against `configs/data_manifest.yaml`
and refuses unpinned upstream revisions. Run it after any mirrored preparation.
