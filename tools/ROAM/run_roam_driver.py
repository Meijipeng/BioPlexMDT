from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    roam_root = os.environ.get("BIOPLEX_ROAM_EXTERNAL_ROOT", "").strip()
    if not roam_root:
        print(
            "[BioPlexMDT][ROAM] Set BIOPLEX_ROAM_EXTERNAL_ROOT to the external ROAM root.",
            file=sys.stderr,
        )
        return 2
    root = Path(roam_root).expanduser()
    entry = root / "run_roam_infer.py"
    if not entry.exists():
        print(f"[BioPlexMDT][ROAM] External entry not found: {entry}", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(entry), *sys.argv[1:]]
    return subprocess.run(cmd, cwd=str(root)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
