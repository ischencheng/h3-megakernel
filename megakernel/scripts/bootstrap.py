"""Fetch the exact ThunderKittens revision used by the vendored runtime."""
from pathlib import Path
import subprocess

REV = "664c108d16f12707a73d3072ab525f26fb2b4f62"
ROOT = Path(__file__).resolve().parents[1] / "third_party" / "ThunderKittens"

def main():
    ROOT.parent.mkdir(parents=True, exist_ok=True)
    if not ROOT.exists():
        subprocess.run(["git", "clone", "--filter=blob:none",
                        "https://github.com/HazyResearch/ThunderKittens.git", str(ROOT)], check=True)
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"]).strip():
        raise SystemExit("ThunderKittens has local changes; refusing to overwrite them")
    subprocess.run(["git", "-C", str(ROOT), "checkout", "--detach", REV], check=True)
    print(f"ThunderKittens ready: {REV}")

if __name__ == "__main__":
    main()
