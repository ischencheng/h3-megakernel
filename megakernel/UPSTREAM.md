# Upstream provenance

Runtime headers in `include/` originate from
https://github.com/HazyResearch/Megakernels
commit `7309cec801537b61fea3b50d7dfe454a6cde578e` (MIT; see LICENSE.hazy).
The SM120 port adds one guarded change to `include/megakernel.cuh`: it omits
`setmaxnreg` on SM120 because ptxas rejects that instruction for this target.
All other copied runtime headers remain unchanged.

Use its matching ThunderKittens revision:
`664c108d16f12707a73d3072ab525f26fb2b4f62`.
The bootstrap script fetches that revision and retains its license.

Local integration lives in src/, scripts/, tests/, Makefile, and the guarded
SM120 register-control change in include/megakernel.cuh.
The local config corrects the upstream bool-valued sleep interval through
an integer override. The single header delta is listed above.

The standalone SM120 NVFP4 projection probe in `src/cutlass_projection.cu`
adapts NVIDIA CUTLASS example 79a at CUTLASS v4.4.2 commit
`da5e086dab31d63815acafdac9a9c5893b1c69e2` (BSD-3-Clause).
`scripts/bootstrap_cutlass.py` pins the ignored checkout.
