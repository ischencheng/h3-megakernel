"""Fetch the pinned CUTLASS release used for the SM120 projection probe."""
from pathlib import Path
import subprocess

REV = "da5e086dab31d63815acafdac9a9c5893b1c69e2"  # CUTLASS v4.4.2
ROOT = Path(__file__).resolve().parents[1] / "third_party" / "CUTLASS"


def main():
    ROOT.parent.mkdir(parents=True, exist_ok=True)
    if not ROOT.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none",
                        "--branch", "v4.4.2", "https://github.com/NVIDIA/cutlass.git",
                        str(ROOT)], check=True)
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"]).strip():
        raise SystemExit("CUTLASS has local changes; refusing to overwrite them")
    found = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                    text=True).strip()
    if found != REV:
        raise SystemExit(f"CUTLASS revision {found} != pinned {REV}")
    print(f"CUTLASS ready: {REV}")


if __name__ == "__main__":
    main()
