# Derived from OpenRLHF (Apache-2.0).
# Modified by the C3 authors for the C3 project.
# See docs/UPSTREAM.md and docs/CHANGES_FROM_OPENRLHF.md for provenance.

import time
import ray
import requests
import inspect
from typing import Any, Dict, List, Optional

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def request_api_wrapper(url, data, try_max_times=5):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            response = requests.post(url=url, json=data, headers=headers, timeout=180)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            response = response.json()
            return response
        except requests.RequestException as e:
            logger.info(f"Request error, please check: {e}")
        except Exception as e:
            logger.info(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")


@ray.remote
def remote_rm_fn_ray(api_url, queries, prompts, labels, metas=None):
    """
    Remote RM HTTP call.

    Backward compatible:
      - If `metas` is provided but the server rejects it, we retry once without metas.
    """
    base = {"query": queries, "prompts": prompts, "labels": labels}

    if metas is None:
        return request_api_wrapper(api_url, base)

    # First try sending metas (fast-fail once), then fallback to legacy payload.
    with_metas = dict(base)
    with_metas["metas"] = metas
    try:
        return request_api_wrapper(api_url, with_metas, try_max_times=1)
    except Exception as e:
        logger.info(f"Remote RM server may not accept `metas`; fallback without metas. err={type(e).__name__}: {e}")
        return request_api_wrapper(api_url, base)


@ray.remote
class RemoteRewardModel:
    def __init__(self, args, remote_rm_url):
        self.args = args
        self.remote_rm_url = [remote_rm_url] if isinstance(remote_rm_url, str) else remote_rm_url
        self.custom_reward_func = None
        self.custom_reward_func_supports_metas = False

        if self.remote_rm_url and self.remote_rm_url[0].endswith(".py"):
            print(
                f"Loading custom `reward_func(queries, prompts, labels, metas=None, **kwargs)` "
                f"(backward compatible) from {self.remote_rm_url[0]}"
            )
            import importlib.util

            spec = importlib.util.spec_from_file_location("reward_func", self.remote_rm_url[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)

            fn = getattr(reward_module, "reward_func", None)
            if fn is None:
                raise RuntimeError(f"custom reward func file has no `reward_func`: {self.remote_rm_url[0]}")

            # Detect whether it supports metas (4th arg) or accepts **kwargs/*args.
            try:
                sig = inspect.signature(fn)
                params = list(sig.parameters.values())
                has_var = any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params)
                self.custom_reward_func_supports_metas = has_var or (len(params) >= 4)
            except Exception:
                self.custom_reward_func_supports_metas = False

            self.custom_reward_func = ray.remote(fn)

    def get_rewards(self, queries_list, prompts_list, labels_list, metas_list=None):
        """
        Return is unchanged (backward compatible): a LIST of chunk results.
          - HTTP RM usually returns dict like {"rewards":[...],"scores":[...],...}
          - Custom reward_func usually returns the same dict format.
        `metas_list` is optional; if provided, it is aligned 1:1 with queries_list.
        """
        if metas_list is not None and len(metas_list) != len(queries_list):
            raise ValueError(
                f"metas_list length mismatch: len(metas_list)={len(metas_list)} vs len(queries_list)={len(queries_list)}"
            )

        if self.custom_reward_func:
            # Let Ray automatically distribute the workload across available resources
            batch_size = self.args.micro_rollout_batch_size
            num_chunks = (len(queries_list) + batch_size - 1) // batch_size
            r_refs = []
            for i in range(num_chunks):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(queries_list))

                q = queries_list[start_idx:end_idx]
                p = prompts_list[start_idx:end_idx]
                l = labels_list[start_idx:end_idx]
                m = metas_list[start_idx:end_idx] if metas_list is not None else None

                if self.custom_reward_func_supports_metas and (m is not None):
                    r = self.custom_reward_func.remote(q, p, l, metas=m)
                else:
                    # Backward compatible call
                    r = self.custom_reward_func.remote(q, p, l)
                r_refs.append(r)
        else:
            # Distribute data across different remote reward function servers
            num_servers = len(self.remote_rm_url)
            batch_size = (len(queries_list) + num_servers - 1) // num_servers
            r_refs = []
            for i in range(num_servers):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(queries_list))
                rm = self.remote_rm_url[i]

                q = queries_list[start_idx:end_idx]
                p = prompts_list[start_idx:end_idx]
                l = labels_list[start_idx:end_idx]
                m = metas_list[start_idx:end_idx] if metas_list is not None else None

                r = remote_rm_fn_ray.remote(
                    rm,
                    queries=q,
                    prompts=p,
                    labels=l,
                    metas=m,
                )
                r_refs.append(r)

        return ray.get(r_refs)
