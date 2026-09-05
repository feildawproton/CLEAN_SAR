# CLEAN_SAR audit — Claude Code (Opus 5)

**Auditor:** Claude Opus 5 (Anthropic), running in Claude Code
**Date:** 2026-09-03 → 2026-09-05
**Subject:** `/home/feildaw/CLEAN_SAR` working tree as-is (uncommitted refactor; last commit `f6734a1`)
**Independence:** this audit was conducted without reference to any other agent's
audit of this repository. Everything here is self-contained.

Start with **[`AUDIT_REPORT.md`](AUDIT_REPORT.md)**.

---

## Layout

```
AUDIT_REPORT.md              the report — findings, evidence, recommendations
README.md                    this file

a01 … a19                    investigation scripts (.py) + captured output (.txt)
_regressions/                generated pytest plugins used by a15/a16 (disposable)
demo_out/                    output from one short demo run

quality_metrics.py           ─┐
test_backend_parity_reference.py │ drop-in deliverables, intended to be
test_psf_physics_reference.py    │ lifted into clean_sar/ or tests/
test_clean_recovery_reference.py │
cuda_check.py                ─┘
```

## Drop-in deliverables

Each is verified — the tests are demonstrated to **fail** when the corresponding
defect is present, not merely to pass today.

| File | Addresses | Lift into | Verified by |
|---|---|---|---|
| `quality_metrics.py` | F2 — `suppression_db` cannot distinguish correct physics from none | `clean_sar/quality.py` | `a14` |
| `test_backend_parity_reference.py` | F4 — parity test compares a host-side scalar, not the arrays | `tests/` | run directly |
| `test_psf_physics_reference.py` | F5 — two tests assert their own inline algebra | `tests/` | `a15` |
| `test_clean_recovery_reference.py` | F6 — nothing validates deconvolution against ground truth | `tests/` | `a16` |
| `cuda_check.py` | F7/F8 — untyped, unchecked driver calls; discarded NVRTC logs | `clean_sar/backends/` | `a19` |

## Running the scripts

All paths are absolute; run from the repository root with the project's venv:

```bash
cd /home/feildaw/CLEAN_SAR
/home/feildaw/mypyenv/bin/python audit/claude_code_opus_5/a01_psf_math.py
/home/feildaw/mypyenv/bin/python -m pytest audit/claude_code_opus_5/test_psf_physics_reference.py -v
```

Requires the SICD data under `/home/feildaw/data/` and
`/home/feildaw/diffpfa/workspace/output/`, plus a CUDA device for the
backend-related scripts.

## Scope note

**Nothing under `clean_sar/` was modified.** Where a defect had to be
reintroduced to prove a test catches it (`a15`, `a16`), it was done by
monkey-patching at runtime inside a subprocess.
