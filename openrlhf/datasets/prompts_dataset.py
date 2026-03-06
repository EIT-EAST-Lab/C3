# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/UPSTREAM.md and docs/CHANGES_FROM_OPENRLHF.md for provenance.

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from torch.utils.data import Dataset
from tqdm import tqdm


def _normalize_str_list(v: Any) -> Optional[List[str]]:
    """Normalize a CLI-provided list-like value into List[str]."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        # Try JSON first: '["a","b"]'
        if (s.startswith("[") and s.endswith("]")) or (s.startswith('"') and s.endswith('"')):
            try:
                loaded = json.loads(s)
                if isinstance(loaded, list):
                    return [str(x) for x in loaded]
                if isinstance(loaded, str):
                    return [loaded]
            except Exception:
                pass
        # Fallback: comma-separated
        return [x.strip() for x in s.split(",") if x.strip()]
    return [str(v)]


def _extract_meta_json(
    data: Any,
    *,
    input_key: str,
    label_key: Optional[str],
    meta_keys: Optional[Sequence[str]] = None,
    meta_exclude_keys: Optional[Sequence[str]] = None,
) -> str:
    """
    Build a compact JSON string carrying "extra" fields (e.g., code tests / math verify info).

    Default behavior (when meta_keys is None):
      - Start from data["meta"] if it exists and is a dict
      - Include all top-level keys except: datasource, input_key, label_key, and some obvious prompt/label aliases
      - Apply meta_exclude_keys if provided

    If meta_keys is provided:
      - Only include those keys (plus data["meta"] if present), then apply meta_exclude_keys
    """
    meta: Dict[str, Any] = {}

    # If there is a nested meta dict, merge it first.
    try:
        nested = data.get("meta", None) if hasattr(data, "get") else None
    except Exception:
        nested = None
    if isinstance(nested, dict):
        meta.update(nested)

    exclude = set(meta_exclude_keys or [])
    # Always exclude these
    exclude.update({"datasource", input_key})
    if label_key is not None:
        exclude.add(label_key)
    # Common aliases we do NOT want to duplicate into meta by default
    exclude.update({"prompt", "instruction", "messages", "input", "output", "label"})

    def _get(k: str) -> Any:
        try:
            if hasattr(data, "get"):
                return data.get(k, None)
        except Exception:
            pass
        try:
            return data[k]
        except Exception:
            return None

    def _has(k: str) -> bool:
        try:
            return k in data
        except Exception:
            try:
                _ = data[k]
                return True
            except Exception:
                return False

    if meta_keys:
        for k in meta_keys:
            if k in exclude:
                continue
            if _has(k):
                meta[k] = _get(k)
    else:
        # Auto include all top-level keys except excludes
        try:
            keys = list(data.keys()) if hasattr(data, "keys") else []
        except Exception:
            keys = []
        for k in keys:
            if k in exclude:
                continue
            # Skip nested meta key itself (we already merged it)
            if k == "meta":
                continue
            meta[k] = _get(k)

    # Apply excludes to nested meta as well
    for k in list(meta.keys()):
        if k in exclude:
            meta.pop(k, None)

    # Always return a valid JSON string for downstream (even if empty)
    try:
        return json.dumps(meta, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError:
        # If some values are not JSON serializable, coerce them to str
        safe_meta = {k: (v if isinstance(v, (str, int, float, bool, type(None), list, dict)) else str(v)) for k, v in meta.items()}
        return json.dumps(safe_meta, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def preprocess_data(
    data,
    input_template=None,
    input_key: Optional[str] = "input",
    label_key: Optional[str] = None,
    apply_chat_template=None,
) -> Tuple[str, str]:
    # Resolve defaults safely
    if input_key is None:
        input_key = "input"

    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)

    # for Reinforced Fine-tuning
    if label_key is None:
        label = ""
    else:
        # tolerate missing labels; env-reward CodeEnv doesn't need them
        label = data.get(label_key, "") if isinstance(data, dict) else ""
    return prompt, label


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Returns by default:
        (datasource, prompt, label)

    If strategy.args.return_meta_json is True, returns:
        (datasource, prompt, label, meta_json)

    meta_json is a compact JSON string carrying extra fields (e.g., code tests / math verify info).
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None) or "input"
        label_key = getattr(self.strategy.args, "label_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        # Meta controls (optional)
        self.return_meta_json = bool(getattr(self.strategy.args, "return_meta_json", False))
        meta_keys = _normalize_str_list(getattr(self.strategy.args, "meta_keys", None))
        meta_exclude_keys = _normalize_str_list(getattr(self.strategy.args, "meta_exclude_keys", None)) or []

        self.prompts: List[str] = []
        self.labels: List[str] = []
        self.datasources: List[str] = []
        self.meta_jsons: List[str] = []

        fixed_bad_datasource = 0

        for i, data in enumerate(
            tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0())
        ):
            # Defensive: dataset row itself should not be None
            if data is None:
                raise ValueError(f"PromptDataset: dataset row is None at idx={i}")

            prompt, label = preprocess_data(
                data, input_template, input_key, label_key, apply_chat_template
            )

            # ---- critical: fail-fast with exact index instead of collate(NoneType) ----
            if prompt is None:
                # show minimal but useful context
                try:
                    keys = list(data.keys())[:50] if hasattr(data, "keys") else []
                except Exception:
                    keys = []
                try:
                    ds_val = data.get("datasource", "MISSING_KEY") if hasattr(data, "get") else "NO_GET"
                except Exception:
                    ds_val = "ERR_GET"
                try:
                    raw_inp = data.get(input_key, None) if hasattr(data, "get") else None
                except Exception:
                    raw_inp = "ERR_GET"

                raise ValueError(
                    "PromptDataset: preprocess_data produced None prompt. "
                    f"idx={i}, input_key={input_key!r}, datasource={ds_val!r}, keys={keys}, "
                    f"raw_input_type={type(raw_inp).__name__}"
                )

            if label is None:
                label = ""

            # datasource: must be non-empty string for stable collation & logging
            ds_val = "default"
            try:
                ds_val = data.get("datasource", "default") if hasattr(data, "get") else "default"
            except Exception:
                ds_val = "default"

            if ds_val is None or (isinstance(ds_val, str) and not ds_val.strip()):
                fixed_bad_datasource += 1
                ds_val = "default"
            ds_val = str(ds_val)

            meta_json = _extract_meta_json(
                data,
                input_key=input_key,
                label_key=label_key,
                meta_keys=meta_keys,
                meta_exclude_keys=meta_exclude_keys,
            )
            if meta_json is None:
                meta_json = "{}"

            self.prompts.append(prompt)
            self.labels.append(label)
            self.datasources.append(ds_val)
            self.meta_jsons.append(meta_json)

        if fixed_bad_datasource > 0 and self.strategy.is_rank_0():
            print(f"[WARN] PromptDataset: fixed {fixed_bad_datasource} samples with None/empty datasource -> 'default'.")


    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        if self.return_meta_json:
            return self.datasources[idx], self.prompts[idx], self.labels[idx], self.meta_jsons[idx]
        return self.datasources[idx], self.prompts[idx], self.labels[idx]
