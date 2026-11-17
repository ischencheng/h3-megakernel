"""Print compiled resource usage of the full-DiT kernel (registers, stack, shared)."""
import glob
import re
import subprocess

path = glob.glob("/root/megakernel/build/RTXPRO6000/mk_h3_dit*.so")[0]
usage = subprocess.run(["cuobjdump", "-res-usage", path], text=True, capture_output=True)
print(usage.stdout[-3000:], usage.stderr[-500:], flush=True)
sass = subprocess.run(["cuobjdump", "-sass", path], text=True, capture_output=True).stdout
lines = sass.splitlines()
print("SASS lines", len(lines))
for pattern in (r"\bHMMA|\bBMMA|MMA", r"\[R1[+\]]", r"\bLDL\b|LDL\.", r"\bSTL\b|STL\.", r"\bLDS\b|LDS\.",
                r"\bLDG\b|LDG\.", r"\bBAR\b|BAR\."):
    print(pattern, sum(1 for line in lines if re.search(pattern, line)))
mma = [i for i, line in enumerate(lines) if "MMA" in line and "16816" in line]
print("bf16/f16 16816 mma count", len(mma))
if mma:
    start = mma[len(mma) // 2]
    print("\n".join(line.strip()[:120] for line in lines[start - 30:start + 30]))
