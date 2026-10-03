#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_tag_hf.py -- one DeepSpeed checkpoint tag, written out as a Hugging Face model dir.

The evaluation of record reads a policy from `<run>/final_hf`
(scripts/70_rebuild/final_eval.py --policy), and a trainer only writes that
directory for the LAST step it reached. When the interesting model is an EARLIER step, its
weights exist only as the ZeRO shards of a checkpoint tag, so somebody has to
turn one tag into the same `final_hf` shape the evaluation already knows how to
read. That is all this script does.

    python scripts/70_rebuild/export_tag_hf.py \
        --base  <the Qwen3-4B-Instruct-2507 snapshot directory> \
        --state <run dir>/magrpo_s250/_export/fp32 \
        --out   <run dir>/magrpo_s250/final_hf

`--state` is what DeepSpeed's own `zero_to_fp32.py` produced from the tag, and
its shape differs by DeepSpeed version: older versions write ONE
`pytorch_model.bin`, newer ones write a DIRECTORY of shards (safetensors with
`--safe_serialization`, `.bin` without). Both are accepted, because the version
in the container is not this script's business.

The one judgement call is the key prefix. OpenRLHF's `Actor` holds the Hugging
Face model as `self.model`, so what DeepSpeed saved is the ACTOR's state dict
and every name in it carries an extra `model.` in front of the name the Hugging
Face model answers to (`model.model.layers.0...`, `model.lm_head.weight`). The
depth of that wrapping is a property of the training code, not of the
checkpoint, so rather than hard-code one prefix the script tries the candidates
and keeps the one that makes the two key sets overlap most (`choose_prefix`).
Then `load_state_dict` has to come back clean: nothing unexpected at all, and
nothing missing except the one name a tied embedding legitimately lacks.

It prints, for the operator: the key counts, the prefix it stripped, the missing
and unexpected lists, the parameter total, and the L2 distance between the base
snapshot and the exported model on the two largest tensors. That last line is
the one that says the export really carries TRAINED weights: a zero there means
the base snapshot was copied and the checkpoint never landed.

CPU only, no Ray, no GPU. A 4B model in fp32 is about 16 GB in and 8 GB out.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

#: The prefixes a saved actor state dict may carry in front of the names the
#: Hugging Face model answers to. `model.` is OpenRLHF's `Actor.model`,
#: `module.` is what a bare DistributedDataParallel wrapper adds, and
#: `module.model.` is both. The empty string is the already-unwrapped case, and
#: it is first so that a tag that needs no stripping is reported as such.
PREFIX_CANDIDATES = ("", "model.", "module.", "module.model.")

#: The output projection, which a tied model holds as the SAME tensor as its
#: input embedding. It shows up twice in this script, for two reasons that are
#: easy to confuse:
#:
#:  * on the way IN, it is the only name `load_state_dict` may report as
#:    missing, and only on a tied model: such a checkpoint may carry the tensor
#:    once, under either name.
#:  * on the way OUT, it is the name that has to be DROPPED before a safetensors
#:    file is written. safetensors refuses to store two names that share memory,
#:    and a tied model's two names do. Measured on the long-generation run: DeepSpeed's
#:    own `zero_to_fp32.py --safe_serialization` died on exactly this, with
#:    `RuntimeError ... [{'lm_head.weight', 'model.embed_tokens.weight'}]`
#:    (deepspeed 0.19.6, safetensors 0.8.0). Dropping the name is not losing the
#:    weight: `tie_word_embeddings` in the written config is what makes
#:    `from_pretrained` bind `lm_head` back to the embedding, which is how the
#:    base snapshot itself is stored.
TIED_LM_HEAD = "lm_head.weight"

#: The name the tied output projection is tied TO. Kept, where TIED_LM_HEAD is
#: dropped, so that whichever of the two the checkpoint carried, the tensor
#: reaches the export under the name the embedding layer answers to.
TIED_EMBEDDING = "model.embed_tokens.weight"

#: Tokenizer and generation files copied from the base snapshot into the export.
#: The evaluation loads the policy directory with `AutoTokenizer`, so a model
#: directory without these is a directory the evaluation cannot use. Whichever
#: of them the snapshot has is copied; a pattern that matches nothing is not an
#: error, because the set differs between tokenizer implementations.
TOKENIZER_PATTERNS = (
    "tokenizer*.json",
    "tokenizer*.model",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template*",
)

#: A checkpoint file may hold the state dict under one wrapper key. Unwrapping
#: is deliberately narrow: only a mapping whose ONLY key is one of these, so a
#: real state dict that happens to contain a parameter called `model.weight`
#: is never mistaken for a wrapper.
STATE_WRAPPER_KEYS = ("module", "state_dict", "model")

#: How many tensors the "these weights really moved" evidence line compares.
N_EVIDENCE_TENSORS = 2

#: Passed to `save_pretrained`, so the export is ONE `model.safetensors` and not
#: a set of shards. This is not a preference: an evaluation chain that tests the
#: policy directory by the file name `final_hf/model.safetensors` would refuse
#: an export that sharded at the default 5 GB. Large
#: enough for every model of this project: an 8B model in bf16 is about 16 GB.
MAX_SHARD_SIZE = "100GB"


class ExportError(RuntimeError):
    """The export refused: a prefix that explains nothing, or a dirty load."""


# ---------------------------------------------------------------------------
# key handling (no torch: this is the part the CPU tier can test)
# ---------------------------------------------------------------------------


def choose_prefix(state_keys: Iterable[str], model_keys: Iterable[str],
                  candidates: Sequence[str] = PREFIX_CANDIDATES) -> Tuple[str, int]:
    """The candidate prefix whose removal overlaps the model's names most.

    Returns `(prefix, overlap)`. Ties keep the earlier candidate, so the empty
    prefix wins a tie and the report says "nothing was stripped" instead of
    naming a prefix that made no difference.
    """
    model = set(model_keys)
    state = list(state_keys)
    best_prefix = ""
    best_overlap = -1
    for prefix in candidates:
        if prefix:
            stripped = {key[len(prefix):] for key in state if key.startswith(prefix)}
        else:
            stripped = set(state)
        overlap = len(stripped & model)
        if overlap > best_overlap:
            best_prefix, best_overlap = prefix, overlap
    return best_prefix, max(0, best_overlap)


def strip_prefix(state: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    """The state dict with `prefix` removed from the names that carry it."""
    if not prefix:
        return dict(state)
    out: Dict[str, Any] = {}
    for key, value in state.items():
        out[key[len(prefix):] if key.startswith(prefix) else key] = value
    return out


def classify_keys(model_keys: Iterable[str],
                  state_keys: Iterable[str]) -> Tuple[List[str], List[str]]:
    """`(missing, unexpected)`: names the model wants and the state dict has spare."""
    model = set(model_keys)
    state = set(state_keys)
    return sorted(model - state), sorted(state - model)


def check_load(missing: Sequence[str], unexpected: Sequence[str], *, tied: bool) -> None:
    """Raise ExportError unless these are the two lists a clean tag produces.

    Nothing unexpected, ever: a spare name means the prefix was wrong or the
    checkpoint belongs to a different architecture, and `strict=False` would
    have swallowed both. Nothing missing either, except `lm_head.weight` on a
    model whose config ties it to the input embedding.
    """
    problems: List[str] = []
    if unexpected:
        problems.append("the checkpoint carries %d name(s) the model does not have: %s"
                        % (len(unexpected), ", ".join(unexpected[:8])))
    allowed = {TIED_LM_HEAD} if tied else set()
    unexplained = [key for key in missing if key not in allowed]
    if unexplained:
        problems.append("the model has %d name(s) the checkpoint does not carry: %s"
                        % (len(unexplained), ", ".join(unexplained[:8])))
    if problems:
        raise ExportError("; ".join(problems))


def unwrap_state(obj: Any, *, depth: int = 4) -> Any:
    """The state dict inside a one-key wrapper, or `obj` unchanged."""
    current = obj
    for _ in range(max(0, depth)):
        if not isinstance(current, dict) or len(current) != 1:
            return current
        (only_key, only_value), = current.items()
        if only_key not in STATE_WRAPPER_KEYS or not isinstance(only_value, dict):
            return current
        current = only_value
    return current


def tokenizer_files(base_dir: str,
                    patterns: Sequence[str] = TOKENIZER_PATTERNS) -> List[str]:
    """File names in `base_dir` matching the tokenizer patterns, sorted, once each."""
    found: List[str] = []
    seen = set()
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(base_dir, pattern))):
            name = os.path.basename(path)
            if name in seen or not os.path.isfile(path):
                continue
            seen.add(name)
            found.append(name)
    return sorted(found)


def copy_tokenizer_files(base_dir: str, out_dir: str,
                         patterns: Sequence[str] = TOKENIZER_PATTERNS) -> List[str]:
    """Copy whichever tokenizer files the snapshot has into the export. Returns the names."""
    names = tokenizer_files(base_dir, patterns)
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        shutil.copy2(os.path.join(base_dir, name), os.path.join(out_dir, name))
    return names


def pick_evidence_keys(numel_by_key: Mapping[str, int],
                       count: int = N_EVIDENCE_TENSORS, *, tied: bool = False) -> List[str]:
    """The `count` largest tensors, largest first, name as the tie-break.

    On a tied model the output projection is left out: it is the SAME tensor as
    the embedding, so comparing both would report one distance twice and call it
    two pieces of evidence.
    """
    candidates = dict(numel_by_key)
    if tied:
        candidates.pop(TIED_LM_HEAD, None)
    ranked = sorted(candidates.items(), key=lambda item: (-int(item[1]), item[0]))
    return [name for name, _numel in ranked[:max(0, count)]]


def drop_tied_keys(names: Iterable[str], *, tied: bool) -> List[str]:
    """The names a safetensors file may carry, sorted: the tied head taken off.

    Only when the model really ties them, and only when the name it is tied to
    is there to carry the tensor. A tied checkpoint that somehow holds the head
    and NOT the embedding keeps the head, because dropping it would drop the
    weight rather than a duplicate of it.
    """
    kept = list(names)
    if tied and TIED_LM_HEAD in kept and TIED_EMBEDDING in kept:
        kept = [name for name in kept if name != TIED_LM_HEAD]
    return sorted(kept)


def state_shard_paths(state: str) -> List[str]:
    """Every file the `--state` argument names, in load order.

    A file is itself. A directory is the shards DeepSpeed wrote into it:
    safetensors first if any are there, otherwise the `.bin` shards, which is
    what this project's DeepSpeed actually writes (`--safe_serialization`
    cannot write a tied checkpoint at all, see `TIED_LM_HEAD` below). An index
    file is not read: the shards are loaded whole and merged, so the index would
    only repeat what the file names already say, and it is a `.json` so neither
    glob picks it up.
    """
    if os.path.isfile(state):
        return [state]
    if not os.path.isdir(state):
        raise ExportError("--state %s is neither a file nor a directory" % state)
    for pattern in ("*.safetensors", "*.bin", "*.pt"):
        shards = sorted(glob.glob(os.path.join(state, pattern)))
        if shards:
            return shards
    raise ExportError("--state %s holds no .safetensors, .bin or .pt file" % state)


# ---------------------------------------------------------------------------
# the export itself (torch and transformers, imported where they are used)
# ---------------------------------------------------------------------------


def load_state(state: str) -> Dict[str, Any]:
    """The merged fp32 state dict of one checkpoint tag."""
    import torch

    merged: Dict[str, Any] = {}
    for path in state_shard_paths(state):
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            shard: Mapping[str, Any] = load_file(path, device="cpu")
        else:
            shard = unwrap_state(torch.load(path, map_location="cpu", weights_only=True))
        if not isinstance(shard, dict):
            raise ExportError("%s does not hold a state dict" % path)
        overlap = set(shard) & set(merged)
        if overlap:
            raise ExportError("%s repeats %d name(s) an earlier shard already carried: %s"
                              % (path, len(overlap), ", ".join(sorted(overlap)[:8])))
        merged.update(shard)
    if not merged:
        raise ExportError("--state %s produced an empty state dict" % state)
    return merged


def export(base: str, state: str, out: str, *, dtype: str = "bfloat16",
           max_shard_size: str = MAX_SHARD_SIZE, stream=None) -> Dict[str, Any]:
    """Write `out` as a Hugging Face model directory holding the tag's weights.

    Returns the report the command line prints, so a test can read the same
    numbers the operator sees.
    """
    import torch
    from transformers import AutoModelForCausalLM

    log = (stream or sys.stdout)

    def say(line: str) -> None:
        print(line, file=log)

    torch_dtype = getattr(torch, dtype, None)
    if torch_dtype is None:
        raise ExportError("--dtype %s is not a torch dtype" % dtype)

    started = time.time()
    say("[export] base     %s" % base)
    say("[export] state    %s" % state)
    say("[export] out      %s" % out)

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch_dtype)
    model_keys = list(model.state_dict().keys())
    tied = bool(getattr(model.config, "tie_word_embeddings", False))

    loaded = load_state(state)
    prefix, overlap = choose_prefix(loaded.keys(), model_keys)
    stripped = strip_prefix(loaded, prefix)
    missing, unexpected = classify_keys(model_keys, stripped.keys())
    say("[export] keys     %d in the checkpoint, %d in the model, %d overlap after "
        "stripping %r" % (len(loaded), len(model_keys), overlap, prefix))
    say("[export] missing  %s" % (", ".join(missing) if missing else "none"))
    say("[export] spare    %s" % (", ".join(unexpected) if unexpected else "none"))
    say("[export] tied_word_embeddings %s" % tied)
    check_load(missing, unexpected, tied=tied)

    # The evidence line: the two largest tensors both sides have, compared
    # before and after the load. Cloned first, because load_state_dict writes
    # into the parameters in place.
    shared = {name: int(tensor.numel())
              for name, tensor in model.state_dict().items() if name in stripped}
    evidence_keys = pick_evidence_keys(shared, tied=tied)
    before = {name: model.state_dict()[name].detach().clone().float()
              for name in evidence_keys}

    result = model.load_state_dict(stripped, strict=False)
    # The same two lists again, this time as torch reports them: two
    # independent readings of one condition, on purpose.
    check_load(list(result.missing_keys), list(result.unexpected_keys), tied=tied)

    evidence: Dict[str, float] = {}
    after = model.state_dict()
    for name in evidence_keys:
        evidence[name] = float(torch.linalg.vector_norm(
            after[name].detach().float() - before[name]).item())
        say("[export] moved    %s: L2 %.6f over %d element(s)"
            % (name, evidence[name], shared[name]))
    if evidence and all(value == 0.0 for value in evidence.values()):
        raise ExportError("every compared tensor is byte-identical to the base snapshot, so "
                          "this export carries no trained weights")

    model = model.to(torch_dtype)
    n_params = int(sum(p.numel() for p in model.parameters()))
    os.makedirs(out, exist_ok=True)

    # The state dict handed to save_pretrained, with the tied output projection
    # taken off: safetensors refuses two names that share memory, and this
    # script does the dropping itself rather than trusting the library version
    # in the container to do it. `tie_word_embeddings` stays true in the written
    # config, so from_pretrained binds the head back on load.
    out_state = model.state_dict()
    tied_names = drop_tied_keys(out_state.keys(), tied=tied)
    dropped = sorted(set(out_state) - set(tied_names))
    if dropped:
        out_state = {name: out_state[name] for name in tied_names}
        say("[export] tied     dropped %s before writing; %s carries the tensor and the config "
            "ties them back on load" % (", ".join(dropped), TIED_EMBEDDING))
    try:
        model.save_pretrained(out, safe_serialization=True, max_shard_size=max_shard_size,
                              state_dict=out_state)
        wrote_with = "state_dict"
    except TypeError as exc:
        # The `state_dict` argument is the one thing here whose spelling belongs
        # to the library rather than to this script. If a version does not take
        # it, fall back to the library's own tie handling rather than die on a
        # keyword: the single-file check below is what actually has to hold, and
        # a tie the library fails to drop comes back as the loud safetensors
        # error rather than as a quiet wrong export.
        say("[export] note     save_pretrained did not take state_dict (%s); writing the "
            "model's own and letting the library drop the tie" % exc)
        model.save_pretrained(out, safe_serialization=True, max_shard_size=max_shard_size)
        wrote_with = "library"
    copied = copy_tokenizer_files(base, out)
    single = os.path.join(out, "model.safetensors")
    if not os.path.isfile(single):
        raise ExportError("%s is not there: the export sharded, and the evaluation chain tests "
                          "the policy directory by that exact name" % single)
    say("[export] params   %d" % n_params)
    say("[export] side     %s" % (", ".join(copied) if copied else "none"))
    say("[export] wrote    %s in %.1f s" % (out, time.time() - started))

    return {
        "base": base,
        "state": state,
        "out": out,
        "dtype": dtype,
        "max_shard_size": max_shard_size,
        "prefix": prefix,
        "n_state_keys": len(loaded),
        "n_model_keys": len(model_keys),
        "overlap": overlap,
        "missing": missing,
        "unexpected": unexpected,
        "tie_word_embeddings": tied,
        "tied_lm_head": bool(dropped),
        "dropped_before_write": dropped,
        "wrote_with": wrote_with,
        "n_params": n_params,
        "side_files": copied,
        "l2_vs_base": evidence,
        "seconds": round(time.time() - started, 1),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True,
                    help="Hugging Face snapshot directory of the base model")
    ap.add_argument("--state", required=True,
                    help="the fp32 state dict zero_to_fp32.py produced: a file or a directory")
    ap.add_argument("--out", required=True, help="the final_hf directory to write")
    ap.add_argument("--dtype", default="bfloat16",
                    help="dtype of the exported weights (default bfloat16)")
    ap.add_argument("--max-shard-size", default=MAX_SHARD_SIZE, dest="max_shard_size",
                    help="passed to save_pretrained; the default keeps the export one file, "
                         "which is the name the evaluation chain looks for")
    ap.add_argument("--report", default="",
                    help="also write the printed report to this JSON file")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = export(args.base, args.state, args.out, dtype=args.dtype,
                        max_shard_size=args.max_shard_size)
    except ExportError as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 1
    if args.report:
        with open(args.report, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
