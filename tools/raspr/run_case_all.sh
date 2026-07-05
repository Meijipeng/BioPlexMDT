#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BIOPLEX_RASPR_EXTERNAL_SCRIPT:-}" ]]; then
  echo "[BioPlexMDT][RaSPr] Set BIOPLEX_RASPR_EXTERNAL_SCRIPT to the external run_case_all.sh path." >&2
  exit 2
fi

if [[ ! -f "${BIOPLEX_RASPR_EXTERNAL_SCRIPT}" ]]; then
  echo "[BioPlexMDT][RaSPr] External script not found: ${BIOPLEX_RASPR_EXTERNAL_SCRIPT}" >&2
  exit 2
fi

bash "${BIOPLEX_RASPR_EXTERNAL_SCRIPT}" "$@"
