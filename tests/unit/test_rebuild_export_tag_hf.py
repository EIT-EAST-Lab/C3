# tests/unit/test_rebuild_export_tag_hf.py
"""`scripts/70_rebuild/export_tag_hf.py`: one checkpoint tag written out as a model dir.

Two halves, and the split is deliberate. The key arithmetic (which prefix the
saved actor carries, what counts as a clean load, which side files travel with
the export) is plain Python and runs anywhere, so it is pinned here on hand
written key sets rather than on a model. The other half needs torch and
transformers, builds a two-layer random model, saves a `model.`-prefixed state
dict and exports it; it is skipped where those are not installed, which is the
CPU tier.

    python -m pytest tests/unit/test_rebuild_export_tag_hf.py -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "70_rebuild"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import export_tag_hf as ex  # noqa: E402

#: The names a Qwen-shaped causal model answers to, cut down to what the prefix
#: arithmetic needs: an embedding, one layer, the norm and the output head.
MODEL_KEYS = (
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.norm.weight",
    "lm_head.weight",
)


def actor_keys(prefix: str) -> list:
    """The same names as one of OpenRLHF's wrapped actors would have saved them."""
    return [prefix + key for key in MODEL_KEYS]


# -----------------------------------------------------------------------------
# which prefix the checkpoint carries
# -----------------------------------------------------------------------------


def test_the_prefix_of_an_openrlhf_actor_is_found_and_not_guessed():
    """`Actor` holds the model as `self.model`, so every name gains one `model.`."""
    prefix, overlap = ex.choose_prefix(actor_keys("model."), MODEL_KEYS)
    assert prefix == "model."
    assert overlap == len(MODEL_KEYS)


def test_a_checkpoint_that_needs_no_stripping_reports_the_empty_prefix():
    prefix, overlap = ex.choose_prefix(MODEL_KEYS, MODEL_KEYS)
    assert prefix == ""
    assert overlap == len(MODEL_KEYS)


@pytest.mark.parametrize("prefix", ["module.", "module.model."])
def test_the_other_wrappers_are_found_too(prefix):
    found, overlap = ex.choose_prefix(actor_keys(prefix), MODEL_KEYS)
    assert found == prefix
    assert overlap == len(MODEL_KEYS)


def test_a_prefix_that_explains_nothing_leaves_the_overlap_at_zero():
    """The refusal path: a checkpoint from another architecture overlaps on
    nothing whichever prefix is taken off, and `check_load` is what turns that
    into a refusal. `choose_prefix` only reports it."""
    prefix, overlap = ex.choose_prefix(["encoder.block.0.weight"], MODEL_KEYS)
    assert overlap == 0
    assert prefix == ""


def test_stripping_leaves_a_name_without_the_prefix_alone():
    state = {"model.norm.weight": 1, "lm_head.weight": 2}
    assert ex.strip_prefix(state, "model.") == {"norm.weight": 1, "lm_head.weight": 2}
    assert ex.strip_prefix(state, "") == state


# -----------------------------------------------------------------------------
# what counts as a clean load
# -----------------------------------------------------------------------------


def test_a_clean_tag_passes_and_reports_two_empty_lists():
    missing, unexpected = ex.classify_keys(MODEL_KEYS, MODEL_KEYS)
    assert missing == [] and unexpected == []
    ex.check_load(missing, unexpected, tied=False)


def test_a_tied_model_may_be_missing_the_output_head_and_nothing_else():
    """One tensor under two names: a tied checkpoint carries it once."""
    ex.check_load([ex.TIED_LM_HEAD], [], tied=True)
    with pytest.raises(ex.ExportError) as caught:
        ex.check_load([ex.TIED_LM_HEAD], [], tied=False)
    assert "does not carry" in str(caught.value)


def test_a_name_the_model_does_not_have_is_a_refusal():
    with pytest.raises(ex.ExportError) as caught:
        ex.check_load([], ["model.layers.99.mlp.up_proj.weight"], tied=True)
    assert "does not have" in str(caught.value)


def test_a_missing_layer_is_a_refusal_even_on_a_tied_model():
    with pytest.raises(ex.ExportError) as caught:
        ex.check_load([ex.TIED_LM_HEAD, "model.norm.weight"], [], tied=True)
    assert "model.norm.weight" in str(caught.value)


def test_the_two_lists_are_reported_together_when_both_are_wrong():
    with pytest.raises(ex.ExportError) as caught:
        ex.check_load(["model.norm.weight"], ["spare.weight"], tied=False)
    message = str(caught.value)
    assert "does not have" in message and "does not carry" in message


# -----------------------------------------------------------------------------
# the state dict on disk
# -----------------------------------------------------------------------------


def test_a_one_key_wrapper_is_unwrapped_and_a_real_state_dict_is_not():
    inner = {"model.norm.weight": 1}
    assert ex.unwrap_state({"module": inner}) == inner
    assert ex.unwrap_state({"state_dict": {"module": inner}}) == inner
    # A state dict of one parameter is not a wrapper, because its key is a name.
    assert ex.unwrap_state(inner) == inner
    # Two keys is never a wrapper, whatever they are called.
    both = {"module": inner, "optimizer": {}}
    assert ex.unwrap_state(both) == both


def test_a_single_file_state_is_itself(tmp_path):
    path = tmp_path / "pytorch_model.bin"
    path.write_bytes(b"not read here")
    assert ex.state_shard_paths(str(path)) == [str(path)]


def test_a_directory_of_shards_is_read_in_name_order(tmp_path):
    for name in ("model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    shards = ex.state_shard_paths(str(tmp_path))
    assert [os.path.basename(p) for p in shards] == [
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]


def test_a_directory_of_bin_shards_is_read_when_there_is_no_safetensors(tmp_path):
    """The other shape `zero_to_fp32.py` writes: a directory, `.bin` shards."""
    for name in ("pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin"):
        (tmp_path / name).write_bytes(b"")
    shards = ex.state_shard_paths(str(tmp_path))
    assert [os.path.basename(p) for p in shards] == [
        "pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin"]


def test_an_empty_directory_and_a_missing_path_are_both_refusals(tmp_path):
    with pytest.raises(ex.ExportError):
        ex.state_shard_paths(str(tmp_path))
    with pytest.raises(ex.ExportError):
        ex.state_shard_paths(str(tmp_path / "nowhere"))


# -----------------------------------------------------------------------------
# the files that travel with the weights
# -----------------------------------------------------------------------------


SNAPSHOT_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model-00001-of-00003.safetensors",
    "model.safetensors.index.json",
)


def write_snapshot(root, names=SNAPSHOT_FILES):
    os.makedirs(str(root), exist_ok=True)
    for name in names:
        with open(os.path.join(str(root), name), "w", encoding="utf-8") as fh:
            fh.write("{}\n")
    return str(root)


def test_the_tokenizer_travels_and_the_base_weights_do_not(tmp_path):
    """The export writes its own weights, so a safetensors shard of the base
    snapshot must NOT be copied over them; the tokenizer must, because the
    evaluation loads the policy directory with `AutoTokenizer`."""
    base = write_snapshot(tmp_path / "snapshot")
    out = str(tmp_path / "final_hf")
    copied = ex.copy_tokenizer_files(base, out)

    assert "tokenizer.json" in copied
    assert "tokenizer_config.json" in copied
    assert "vocab.json" in copied and "merges.txt" in copied
    assert "generation_config.json" in copied
    assert not [name for name in copied if name.endswith(".safetensors")]
    assert "model.safetensors.index.json" not in copied
    # config.json is the model's own, written by save_pretrained, not copied.
    assert "config.json" not in copied
    assert sorted(os.listdir(out)) == sorted(copied)


def test_a_snapshot_without_a_merges_file_is_not_an_error(tmp_path):
    """Which side files exist differs by tokenizer implementation, so a pattern
    that matches nothing is an absence and not a failure."""
    base = write_snapshot(tmp_path / "snapshot",
                          names=("config.json", "tokenizer.json", "tokenizer_config.json"))
    copied = ex.copy_tokenizer_files(base, str(tmp_path / "final_hf"))
    assert copied == ["tokenizer.json", "tokenizer_config.json"]


def test_a_chat_template_file_travels_when_the_snapshot_carries_one(tmp_path):
    base = write_snapshot(tmp_path / "snapshot",
                          names=("chat_template.jinja", "tokenizer_config.json"))
    copied = ex.copy_tokenizer_files(base, str(tmp_path / "final_hf"))
    assert copied == ["chat_template.jinja", "tokenizer_config.json"]


def test_each_side_file_is_reported_once(tmp_path):
    """`tokenizer*.json` and `tokenizer_config.json` both match the same file."""
    base = write_snapshot(tmp_path / "snapshot", names=("tokenizer_config.json",))
    assert ex.tokenizer_files(base) == ["tokenizer_config.json"]


# -----------------------------------------------------------------------------
# the evidence line
# -----------------------------------------------------------------------------


def test_the_evidence_compares_the_largest_tensors(tmp_path):
    numel = {"model.embed_tokens.weight": 1000, "model.norm.weight": 10,
             "model.layers.0.mlp.gate_proj.weight": 500}
    assert ex.pick_evidence_keys(numel) == [
        "model.embed_tokens.weight", "model.layers.0.mlp.gate_proj.weight"]


def test_two_tensors_of_the_same_size_are_ordered_by_name():
    numel = {"b.weight": 8, "a.weight": 8, "c.weight": 1}
    assert ex.pick_evidence_keys(numel) == ["a.weight", "b.weight"]


def test_the_evidence_of_a_tied_model_is_two_tensors_and_not_one_twice():
    """The output projection and the embedding of a tied model are the same
    tensor, and they are the two biggest, so taking the top two by size would
    report one distance twice and present it as two pieces of evidence."""
    numel = {"lm_head.weight": 1000, "model.embed_tokens.weight": 1000,
             "model.layers.0.mlp.gate_proj.weight": 500, "model.norm.weight": 10}
    assert ex.pick_evidence_keys(numel, tied=True) == [
        "model.embed_tokens.weight", "model.layers.0.mlp.gate_proj.weight"]
    # Untied, the two names really are two tensors and both may be compared.
    assert ex.pick_evidence_keys(numel, tied=False) == [
        "lm_head.weight", "model.embed_tokens.weight"]


# -----------------------------------------------------------------------------
# the tied output projection, on the way out
#
# safetensors refuses two names that share memory. Measured on the long-generation run:
# DeepSpeed's own `zero_to_fp32.py --safe_serialization` died on exactly this
# pair, so the export writes a torch pickle in that stage and does the dropping
# itself here, where the tie is known about.
# -----------------------------------------------------------------------------


def test_the_tied_head_is_dropped_and_the_embedding_keeps_the_tensor():
    names = ["lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"]
    assert ex.drop_tied_keys(names, tied=True) == [
        "model.embed_tokens.weight", "model.norm.weight"]


def test_an_untied_model_keeps_both_names():
    """Two real tensors; dropping one would lose a weight."""
    names = ["lm_head.weight", "model.embed_tokens.weight"]
    assert ex.drop_tied_keys(names, tied=False) == names


def test_a_tied_checkpoint_that_has_only_the_head_keeps_it():
    """The guard on the guard: with nothing to fall back on, dropping the head
    drops the tensor rather than a duplicate of it."""
    assert ex.drop_tied_keys(["lm_head.weight", "model.norm.weight"], tied=True) == [
        "lm_head.weight", "model.norm.weight"]


def test_dropping_is_idempotent_and_sorted():
    once = ex.drop_tied_keys(
        ["model.norm.weight", "lm_head.weight", "model.embed_tokens.weight"], tied=True)
    assert once == ex.drop_tied_keys(once, tied=True)
    assert once == sorted(once)


# -----------------------------------------------------------------------------
# the command line
# -----------------------------------------------------------------------------


def test_the_command_line_requires_the_three_paths():
    parser = ex.build_parser()
    args = parser.parse_args(["--base", "b", "--state", "s", "--out", "o"])
    assert (args.base, args.state, args.out) == ("b", "s", "o")
    assert args.dtype == "bfloat16"
    assert args.max_shard_size == ex.MAX_SHARD_SIZE
    with pytest.raises(SystemExit):
        parser.parse_args(["--base", "b", "--state", "s"])


def test_the_export_is_one_file_because_the_chain_looks_for_that_name():
    """An evaluation chain that tests a policy directory for
    `final_hf/model.safetensors` needs the export in one file. The default shard size of `save_pretrained` is
    5 GB and a 4B model in bf16 is about 8, so an export left at the default
    would be refused by the evaluation it was made for. Pinned as a number
    rather than as prose because that is what the default has to beat."""
    assert ex.MAX_SHARD_SIZE.endswith("GB")
    assert int(ex.MAX_SHARD_SIZE[:-2]) >= 50


# -----------------------------------------------------------------------------
# the whole export, on a two-layer random model
# -----------------------------------------------------------------------------


def _torch_or_skip():
    torch = pytest.importorskip("torch", reason="the export itself needs torch")
    transformers = pytest.importorskip(
        "transformers", reason="the export itself needs transformers")
    return torch, transformers


def tiny_model(torch, transformers, tmp_path, *, tie=True):
    """A two-layer Qwen2 model with random weights, saved as a base snapshot."""
    config = transformers.Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=tie)
    model = transformers.AutoModelForCausalLM.from_config(config)
    base = str(tmp_path / "snapshot")
    model.save_pretrained(base, safe_serialization=True)
    with open(os.path.join(base, "tokenizer_config.json"), "w", encoding="utf-8") as fh:
        fh.write("{}\n")
    return model, base


def test_the_export_writes_the_trained_weights_and_not_the_base_ones(tmp_path):
    torch, transformers = _torch_or_skip()
    model, base = tiny_model(torch, transformers, tmp_path)

    # The checkpoint: the same names with OpenRLHF's `model.` in front, and
    # every value shifted, so the export has something to prove it loaded.
    trained = {"model." + name: tensor.detach().clone().float() + 0.25
               for name, tensor in model.state_dict().items()}
    state = str(tmp_path / "fp32.bin")
    torch.save(trained, state)

    out = str(tmp_path / "final_hf")
    report = ex.export(base, state, out, dtype="bfloat16")

    assert report["prefix"] == "model."
    assert report["unexpected"] == []
    assert report["missing"] in ([], [ex.TIED_LM_HEAD])
    assert report["overlap"] == report["n_model_keys"] - len(report["missing"])
    assert all(value > 0.0 for value in report["l2_vs_base"].values())
    assert os.path.isfile(os.path.join(out, "model.safetensors"))
    assert os.path.isfile(os.path.join(out, "config.json"))
    assert "tokenizer_config.json" in report["side_files"]
    # One file and no shard index: the name the evaluation chain looks for.
    assert not [name for name in os.listdir(out) if name.startswith("model-0000")]
    assert not os.path.isfile(os.path.join(out, "model.safetensors.index.json"))

    # Read back: the exported weights are the checkpoint's, at bf16 precision.
    loaded = transformers.AutoModelForCausalLM.from_pretrained(out, torch_dtype=torch.bfloat16)
    for name, tensor in loaded.state_dict().items():
        expected = trained["model." + name].to(torch.bfloat16)
        assert torch.equal(tensor, expected), name


def test_a_state_dict_from_a_directory_of_shards_loads_the_same_way(tmp_path):
    """The newer `zero_to_fp32.py` writes a directory; the export reads both."""
    torch, transformers = _torch_or_skip()
    safetensors = pytest.importorskip("safetensors", reason="the shard branch needs safetensors")
    from safetensors.torch import save_file

    model, base = tiny_model(torch, transformers, tmp_path)
    names = sorted(model.state_dict())
    half = len(names) // 2
    shards = str(tmp_path / "fp32")
    os.makedirs(shards, exist_ok=True)
    for index, chunk in enumerate((names[:half], names[half:]), start=1):
        save_file({"model." + name: model.state_dict()[name].detach().clone().float() + 0.5
                   for name in chunk},
                  os.path.join(shards, "model-0000%d-of-00002.safetensors" % index))

    report = ex.export(base, shards, str(tmp_path / "final_hf"))
    assert report["prefix"] == "model."
    assert report["unexpected"] == []
    assert all(value > 0.0 for value in report["l2_vs_base"].values())
    assert safetensors is not None


def test_a_checkpoint_of_another_architecture_is_refused(tmp_path):
    torch, transformers = _torch_or_skip()
    _model, base = tiny_model(torch, transformers, tmp_path)
    state = str(tmp_path / "fp32.bin")
    torch.save({"encoder.block.0.weight": torch.zeros(4)}, state)

    with pytest.raises(ex.ExportError):
        ex.export(base, state, str(tmp_path / "final_hf"))


def test_an_export_that_would_only_copy_the_base_snapshot_is_refused(tmp_path):
    """The failure this guard exists for: the tag never landed and the export is
    the base model wearing a method name. Nothing about the key sets is wrong
    there, so only the distance from the base catches it."""
    torch, transformers = _torch_or_skip()
    model, base = tiny_model(torch, transformers, tmp_path)
    state = str(tmp_path / "fp32.bin")
    torch.save({"model." + name: tensor.detach().clone().float()
                for name, tensor in model.state_dict().items()}, state)

    with pytest.raises(ex.ExportError) as caught:
        ex.export(base, state, str(tmp_path / "final_hf"))
    assert "no trained weights" in str(caught.value)


def test_a_checkpoint_whose_two_names_share_memory_still_writes_one_file(tmp_path):
    """The failure of the first real run, reproduced from the other side.

    The checkpoint holds ONE tensor under `model.lm_head.weight` and
    `model.model.embed_tokens.weight` (the tie, plus OpenRLHF's actor prefix),
    which is what DeepSpeed's consolidated fp32 state dict carries and what
    safetensors refuses to serialize. The export has to come out as a single
    `model.safetensors` anyway, and the head has to come back on load.
    """
    torch, transformers = _torch_or_skip()
    model, base = tiny_model(torch, transformers, tmp_path, tie=True)

    shared = model.get_input_embeddings().weight.detach().clone().float() + 0.25
    trained = {}
    for name, tensor in model.state_dict().items():
        trained["model." + name] = tensor.detach().clone().float() + 0.25
    # The same object under both names, not two equal copies: this is the
    # condition safetensors tests for.
    trained["model.lm_head.weight"] = shared
    trained["model.model.embed_tokens.weight"] = shared
    assert (trained["model.lm_head.weight"].data_ptr()
            == trained["model.model.embed_tokens.weight"].data_ptr())

    state = str(tmp_path / "fp32.bin")
    torch.save(trained, state)
    out = str(tmp_path / "final_hf")
    report = ex.export(base, state, out)

    assert report["tie_word_embeddings"] is True
    assert report["tied_lm_head"] is True
    assert report["dropped_before_write"] == [ex.TIED_LM_HEAD]
    assert report["wrote_with"] in ("state_dict", "library")
    assert report["unexpected"] == []
    # One file, no shards, no index.
    assert os.path.isfile(os.path.join(out, "model.safetensors"))
    assert not [name for name in os.listdir(out) if name.startswith("model-0000")]
    assert not os.path.isfile(os.path.join(out, "model.safetensors.index.json"))
    # The evidence is two different tensors, and both moved.
    assert ex.TIED_LM_HEAD not in report["l2_vs_base"]
    assert len(report["l2_vs_base"]) == ex.N_EVIDENCE_TENSORS
    assert all(value > 0.0 for value in report["l2_vs_base"].values())

    # Read back: the head is bound to the embedding again, and both carry the
    # checkpoint's weights.
    loaded = transformers.AutoModelForCausalLM.from_pretrained(out, torch_dtype=torch.bfloat16)
    expected = shared.to(torch.bfloat16)
    assert torch.equal(loaded.get_input_embeddings().weight, expected)
    assert torch.equal(loaded.get_output_embeddings().weight, expected)


def test_a_directory_of_bin_shards_loads_and_the_index_is_ignored(tmp_path):
    """The shape this project's DeepSpeed writes without --safe_serialization:
    a directory of `pytorch_model*.bin` plus `pytorch_model.bin.index.json`."""
    torch, transformers = _torch_or_skip()
    model, base = tiny_model(torch, transformers, tmp_path, tie=True)

    names = sorted(model.state_dict())
    half = len(names) // 2
    shards = str(tmp_path / "fp32")
    os.makedirs(shards, exist_ok=True)
    chunks = (names[:half], names[half:])
    for index, chunk in enumerate(chunks, start=1):
        torch.save({"model." + name: model.state_dict()[name].detach().clone().float() + 0.5
                    for name in chunk},
                   os.path.join(shards, "pytorch_model-0000%d-of-00002.bin" % index))
    # The index names every shard; nothing reads it, and it must not be loaded
    # as if it were one.
    with open(os.path.join(shards, "pytorch_model.bin.index.json"), "w", encoding="utf-8") as fh:
        json.dump({"metadata": {"total_size": 0}, "weight_map": {}}, fh)

    assert [os.path.basename(p) for p in ex.state_shard_paths(shards)] == [
        "pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin"]

    report = ex.export(base, shards, str(tmp_path / "final_hf"))
    assert report["prefix"] == "model."
    assert report["unexpected"] == []
    assert report["n_state_keys"] == len(names)
    assert all(value > 0.0 for value in report["l2_vs_base"].values())


def test_a_shard_directory_that_repeats_a_name_is_refused(tmp_path):
    """Two shards carrying the same name means the merge would silently keep one
    of them, and which one would depend on the file order."""
    torch, transformers = _torch_or_skip()
    model, base = tiny_model(torch, transformers, tmp_path)
    shards = str(tmp_path / "fp32")
    os.makedirs(shards, exist_ok=True)
    one = {"model.model.norm.weight": model.state_dict()["model.norm.weight"].float()}
    for index in (1, 2):
        torch.save(one, os.path.join(shards, "pytorch_model-0000%d-of-00002.bin" % index))

    with pytest.raises(ex.ExportError) as caught:
        ex.load_state(shards)
    assert "repeats" in str(caught.value)


def test_the_command_line_writes_its_report(tmp_path):
    torch, transformers = _torch_or_skip()
    model, base = tiny_model(torch, transformers, tmp_path)
    state = str(tmp_path / "fp32.bin")
    torch.save({"model." + name: tensor.detach().clone().float() + 0.25
                for name, tensor in model.state_dict().items()}, state)
    report = str(tmp_path / "export.json")

    rc = ex.main(["--base", base, "--state", state,
                  "--out", str(tmp_path / "final_hf"), "--report", report])
    assert rc == 0
    with open(report, encoding="utf-8") as fh:
        assert json.load(fh)["prefix"] == "model."
