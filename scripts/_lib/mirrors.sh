#!/usr/bin/env bash
# scripts/_lib/mirrors.sh
#
# Optional network mirror knobs for data and model preparation.
#
# Nothing in this repository defaults to a mirror: every download goes to the
# official source unless you opt in by exporting one of the variables below.
# See docs/31_network_mirrors.md for the user-facing documentation.
#
#   HF_ENDPOINT           Hugging Face Hub endpoint.
#                         Default: https://huggingface.co
#                         Example: https://hf-mirror.com
#   GITHUB_MIRROR_PREFIX  Prefix prepended to a full github.com URL.
#                         Default: empty (talk to github.com directly)
#                         Example: https://ghproxy.net/
#   PIP_INDEX_URL         Python package index used by pip.
#   PIP_EXTRA_INDEX_URL   Additional index used by pip.
#   HF_HUB_DISABLE_XET    Turn off the Xet transfer backend of huggingface_hub.
#                         Default: 1, set by this library (plain HTTP transfer)
#                         Xet-backed repositories redirect their file transfers
#                         to cas-server.xethub.hf.co, which a Hub mirror cannot
#                         authorize: a mirrored download then stops partway with
#                         a 401. With the default, the plain HTTP transfer is
#                         used and the same download completes. Set it to 0 to
#                         allow Xet transfers; that works against the official
#                         endpoint, not through a mirror.
#
# Functions:
#   c3_mirrors_validate   Fail on a malformed value. Returns non-zero on error.
#   c3_mirrors_report     Print which knobs are active.
#   c3_mirror_github_url  Echo a github.com URL, mirrored when a prefix is set.

C3_HF_OFFICIAL_ENDPOINT="https://huggingface.co"

# The one value this library sets rather than only reads. It is not a mirror: it
# selects the transfer backend huggingface_hub uses against whatever endpoint is
# configured. A caller who exports it keeps their own value.
: "${HF_HUB_DISABLE_XET:=1}"
export HF_HUB_DISABLE_XET

_c3_mirrors_is_url() {
  case "$1" in
    http://*|https://*) return 0 ;;
    *) return 1 ;;
  esac
}

c3_mirrors_validate() {
  local ok=0
  local name value
  for name in HF_ENDPOINT GITHUB_MIRROR_PREFIX PIP_INDEX_URL PIP_EXTRA_INDEX_URL; do
    value="${!name:-}"
    [[ -n "${value}" ]] || continue
    if ! _c3_mirrors_is_url "${value}"; then
      echo "[FAIL] ${name}=${value} is not an http(s) URL." >&2
      ok=1
    fi
  done

  if [[ -n "${GITHUB_MIRROR_PREFIX:-}" && "${GITHUB_MIRROR_PREFIX}" != */ ]]; then
    echo "[FAIL] GITHUB_MIRROR_PREFIX must end with a slash (got '${GITHUB_MIRROR_PREFIX}')." >&2
    ok=1
  fi

  return "${ok}"
}

c3_mirrors_report() {
  local any=0
  echo "[INFO] Network sources:"
  if [[ -n "${HF_ENDPOINT:-}" && "${HF_ENDPOINT}" != "${C3_HF_OFFICIAL_ENDPOINT}" ]]; then
    echo "[INFO]   Hugging Face: ${HF_ENDPOINT} (mirror)"
    any=1
  else
    echo "[INFO]   Hugging Face: ${HF_ENDPOINT:-${C3_HF_OFFICIAL_ENDPOINT}} (official)"
  fi

  if [[ "${HF_HUB_DISABLE_XET}" == "0" ]]; then
    echo "[INFO]   Hugging Face transfer: Xet enabled (HF_HUB_DISABLE_XET=0)"
  else
    echo "[INFO]   Hugging Face transfer: plain HTTP, Xet off (HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET})"
  fi

  if [[ -n "${GITHUB_MIRROR_PREFIX:-}" ]]; then
    echo "[INFO]   GitHub releases: ${GITHUB_MIRROR_PREFIX}<url> (mirror)"
    any=1
  else
    echo "[INFO]   GitHub releases: https://github.com (official)"
  fi

  if [[ -n "${PIP_INDEX_URL:-}" ]]; then
    echo "[INFO]   pip index: ${PIP_INDEX_URL}"
    any=1
  fi
  if [[ -n "${PIP_EXTRA_INDEX_URL:-}" ]]; then
    echo "[INFO]   pip extra index: ${PIP_EXTRA_INDEX_URL}"
    any=1
  fi

  if [[ "${any}" -eq 0 ]]; then
    echo "[INFO]   No mirrors configured; see docs/31_network_mirrors.md to opt in."
  fi
}

c3_mirror_github_url() {
  local url="$1"
  if [[ -n "${GITHUB_MIRROR_PREFIX:-}" ]]; then
    printf '%s%s\n' "${GITHUB_MIRROR_PREFIX}" "${url}"
  else
    printf '%s\n' "${url}"
  fi
}
