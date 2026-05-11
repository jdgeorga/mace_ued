#!/usr/bin/env bash
# Source this file when torch fails with an iJIT_NotifyEvent symbol error.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this file instead of executing it:" >&2
  echo "  source ${BASH_SOURCE[0]}" >&2
  exit 2
fi

_MLIP_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_MLIP_REPO_ROOT="$(cd "${_MLIP_ENV_DIR}/.." && pwd)"
_MLIP_ITTNOTIFY_STUB="${MLIP_ITTNOTIFY_STUB:-${_MLIP_REPO_ROOT}/../4-Diffuse-Intensity/ittnotify_stub/libittnotify.so}"

if [[ ! -r "${_MLIP_ITTNOTIFY_STUB}" ]]; then
  echo "ITT Notify stub not found: ${_MLIP_ITTNOTIFY_STUB}" >&2
  echo "Set MLIP_ITTNOTIFY_STUB=/path/to/libittnotify.so and source this file again." >&2
  return 1
fi

case ":${LD_PRELOAD:-}:" in
  *":${_MLIP_ITTNOTIFY_STUB}:"*) ;;
  *) export LD_PRELOAD="${_MLIP_ITTNOTIFY_STUB}${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
esac

unset _MLIP_ENV_DIR
unset _MLIP_REPO_ROOT
unset _MLIP_ITTNOTIFY_STUB
