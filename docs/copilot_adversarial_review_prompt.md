# Copilot adversarial review prompt for real-TIFF fixture PR

Use this prompt with Copilot Chat (Agent mode) to perform an adversarial review of the real-TIFF fixture integration PR.

---

You are doing a **hostile/adversarial code review** of a PR in the `CellPatcher` repository.

## Objective
Find regressions, silent assumptions, brittle tests, incorrect assertions, and notebook/runtime mismatches introduced by the PR that added:
- bundled real-TIFF fixture files,
- `tests/fixtures/helpers.py`,
- `tests/test_real_tiff_fixture_modes.py`,
- notebook integration in `CellPatcher_strategy_examples.ipynb` and `CellPatcher_v3.ipynb`,
- fixture/readme docs updates.

Prioritize evidence-based findings over style feedback.

## Review scope
Audit these files first:
- `tests/fixtures/helpers.py`
- `tests/fixtures/readme.md`
- `tests/test_real_tiff_fixture_modes.py`
- `notebooks/CellPatcher/CellPatcher_strategy_examples.ipynb`
- `notebooks/CellPatcher/CellPatcher_v3.ipynb`
- `README.md`
- `notebooks/README.md`
- `src/cellpatcher/config.py`
- `src/cellpatcher/pipeline.py`
- `src/cellpatcher/cell_scan.py`
- `src/cellpatcher/filters.py`
- `src/cellpatcher/extract.py`
- `src/cellpatcher/export.py`

## Required adversarial checks

1. **Fixture integrity and determinism**
   - Verify image/label TIFF are present and local-only (no downloads).
   - Confirm shapes and axes assumptions are correct and consistent (`YX`).
   - Check if expected constants in tests are fragile to benign fixture re-encoding.

2. **Cell-mode correctness**
   - Validate exact expected patch counts for both `exclude_border=True` and `False`.
   - Confirm behavior for `center_on` variants and binary mask policies (`connected_components`, `single_object`).
   - Check `scan_cell_patch_size` expectations vs `run_patching` results for off-by-one or policy mismatch.

3. **Random/grid/whole_fov invariants**
   - Random: deterministic replay with seed, mask-constrained sampling correctness.
   - Grid: verify overlap math and duplicate-coordinate handling.
   - Whole FOV: ensure coordinates exactly cover full image extent.

4. **Filtering and extraction edge cases**
   - Ensure filter thresholds are not tautological or accidentally always-passing.
   - Verify `pad_mode`, `constant_values`, `output_size` (int + tuple), and `resize_order` assertions actually validate behavior.
   - Confirm `apply_mask` checks are meaningful (not vacuous).

5. **Export assertions**
   - Validate exported file-count invariants and metadata CSV expectations.
   - Check that TIFF/NPY format assertions are robust.

6. **Notebook integration risks**
   - Verify fallback behavior outside repo (e.g., Colab) remains functional.
   - Confirm `load_demo_fixture` button only sets state and does not bypass backend APIs.
   - Check that widget defaults and UI state resets do not create stale-state bugs.

7. **Config/docs/tests consistency**
   - Cross-check all tunable params in `config.py` against:
     - docs mention,
     - notebook exposure,
     - real-TIFF test coverage.
   - Flag any config fields missing in tests/docs/notebook UI.

## Output format (strict)
Return findings in this table:

| Severity | File | Line(s) | Issue | Why it matters | Repro / Evidence | Suggested fix |
|---|---|---|---|---|---|---|

Then include:
1. **Top 5 highest-risk issues** (if any)
2. **False confidence checks**: tests that pass but may be weak
3. **Minimal patch plan**: smallest safe edits to fix issues
4. **Confidence level** (High/Med/Low) with rationale

## Severity rubric
- **High**: incorrect behavior, silent data corruption, deterministic failure risk, or invalid test assumption.
- **Medium**: brittle tests/docs mismatch/notebook state bug likely to confuse users.
- **Low**: clarity, maintainability, minor robustness improvements.

Be skeptical. Assume hidden regressions exist until disproven.
