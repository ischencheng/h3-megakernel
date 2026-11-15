# Upstream provenance

Runtime headers in `include/` are copied unchanged from
https://github.com/HazyResearch/Megakernels
commit `7309cec801537b61fea3b50d7dfe454a6cde578e` (MIT; see LICENSE.hazy).

Use its matching ThunderKittens revision:
`664c108d16f12707a73d3072ab525f26fb2b4f62`.
The bootstrap script fetches that revision and retains its license.

Local integration lives in src/, scripts/, tests/, and Makefile.
The local config corrects the upstream bool-valued sleep interval through
an integer override; upstream headers remain unchanged for comparison.
