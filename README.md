# sparse-routing-repro

Unofficial implementation of *Budgeted Sparse Routing for Mixture-of-Experts* (Okafor, Lund, 2026).

This repository reproduces the sparse expert-routing experiments from that paper. It does not implement gated sparse attention; see the paper's appendix for the difference between the two routing schemes.

Maintainers: Repro Collective

## Contents

- `src/moe_router.py` — budgeted mixture-of-experts router
- `requirements.txt`
