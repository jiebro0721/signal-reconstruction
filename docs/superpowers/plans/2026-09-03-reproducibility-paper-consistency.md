# Reproducibility and Paper Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix confirmed reproducibility and algorithm-fidelity defects, preserve or improve numerical results, and make narrowly scoped data/wording corrections in the paper and documentation.

**Architecture:** Add focused pytest regressions around the Problem 3 GPSR solver and experiment entry point, then make the smallest solver changes that satisfy the reference formulas. Compare corrected and baseline configurations on deterministic instances before accepting numerical changes. Update only stale values, stopping-rule descriptions, and claims in existing documentation and LaTeX; do not restructure or substantially delete the paper.

**Tech Stack:** Python 3, NumPy, SciPy, pytest, LaTeX, Git/GitHub.

---

### Task 1: Reproduce and fix the debias keyword failure

**Files:**
- Create: `tests/test_problem3_regressions.py`
- Modify: `exp/problem3/sanity_check.py:40-42`

- [x] **Step 1: Write the failing test**

```python
def test_problem3_sanity_uses_supported_debias_keyword():
    source = Path("exp/problem3/sanity_check.py").read_text(encoding="utf-8")
    assert "do_debias=True" in source
    assert "debias=True" not in source
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_problem3_regressions.py::test_problem3_sanity_uses_supported_debias_keyword -v`
Expected: FAIL because the script currently passes `debias=True`.

- [x] **Step 3: Implement the minimal fix**

Replace the unsupported keyword with `do_debias=True`.

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_problem3_regressions.py::test_problem3_sanity_uses_supported_debias_keyword -v`
Expected: PASS.

### Task 2: Correct and test the GPSR-BB spectral update

**Files:**
- Modify: `tests/test_problem3_regressions.py`
- Modify: `src/cg_solvers.py:203-218`

- [x] **Step 1: Write a failing one-step spectral-update test**

Test a deterministic positive semidefinite BCQP step with a line-search fraction below one and assert that the next spectral step is `s.T@s / s.T@B@s`, independent of the line-search fraction.

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_problem3_regressions.py::test_gpsr_bb_uses_projected_direction_curvature -v`
Expected: FAIL because the current denominator contains the accepted line-search fraction.

- [x] **Step 3: Implement the reference update**

Use the projected direction `s = w - z` and compute:

```python
sBs = q.sBs(s)
ss = float(np.dot(s, s))
a = alpha_max if sBs <= 1e-300 else np.clip(ss / sBs, alpha_min, alpha_max)
```

- [x] **Step 4: Run the focused tests**

Run: `python -m pytest tests/test_problem3_regressions.py -v`
Expected: PASS.

### Task 3: Make PRP/PRP+ diagnostics truthful without changing the main method

**Files:**
- Modify: `tests/test_problem3_regressions.py`
- Modify: `src/cg_solvers.py`
- Modify: `exp/problem3/run_problem3.py`

- [x] **Step 1: Write failing tests for line-search status and counters**

Assert that strong-Wolfe success, Armijo fallback, and complete failure are distinguishable, and that nonlinear CG reports negative-beta truncations, descent restarts, and line-search fallbacks.

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_problem3_regressions.py -v`
Expected: FAIL because current output exposes only a Boolean failure flag and no counters.

- [x] **Step 3: Add minimal status/counter reporting**

Preserve the existing practical PRP+ fallback behavior. Ensure the raw PRP comparison differs only in beta truncation and reports when a Wolfe condition was not obtained.

- [x] **Step 4: Run tests and deterministic comparison**

Run: `python -m pytest tests/test_problem3_regressions.py -v`
Expected: PASS.

### Task 4: Compare corrected algorithms against the stored baseline

**Files:**
- Create or update: `exp/problem3/results/tables/problem3_validation.csv`
- Modify only if accepted: `exp/problem3/results/tables/problem3_full.csv`
- Modify only if accepted: `exp/problem3/results/tables/problem3_summary.csv`

- [x] **Step 1: Run deterministic seeds with the corrected GPSR update**

Run Problem 3 on seeds 0 through 9 and record objective, relative error, support statistics, iterations, and elapsed time.

- [x] **Step 2: Apply the acceptance rule**

Accept the corrected reference update when it preserves the final objective within numerical tolerance and does not materially worsen mean relative error/support recovery. If a proposed optional engineering change worsens recovery, retain the repository's previous practical method and document the reason.

- [x] **Step 3: Regenerate dependent tables and figures only when numerical values changed**

Run: `python exp/problem3/run_problem3.py` and `python exp/problem3/make_figures3.py`.

### Task 5: Synchronize README and model documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/problem2_model.md`
- Modify: `docs/problem3_model.md`
- Modify: `src/amf.py`

- [x] **Step 1: Correct the Lena improvement value**

Use the stored three-seed means: `38.353782 - 33.879610 = 4.474172 dB` versus AMF and `10.100208 dB` versus the 7x7 median.

- [x] **Step 2: Synchronize current experiment settings**

Document `x0=A.T@b`, the continuation sequence through `1e-6`, the relative-objective stopping rule, and the actual per-stage iteration cap.

- [x] **Step 3: Correct narrow technical wording**

Describe the power potential issue as singular second derivative/poor gradient Lipschitz behavior near zero; describe its smoothing as inspired by the cited smoothing family. Clarify that endpoint pixels excluded from the AMF candidate set remain fixed in phase two.

### Task 6: Apply minimal paper corrections

**Files:**
- Modify: `paper/main.tex`
- Regenerate: `paper/main.pdf`

- [x] **Step 1: Change only affected values and claims**

Correct the AMF gain, Problem 3 stopping rule/iteration cap, power-potential explanation, smoothing attribution, `0.08%` versus `0.01%` inconsistency, and PRP/PRP+ interpretation. Update numerical table cells only when the accepted rerun changes them.

- [x] **Step 2: Compile and inspect the PDF**

Run the repository LaTeX build command twice, verify references, render all pages, and inspect for overflow or missing figures.

### Task 7: Full regression, review, commit, and PR

**Files:**
- Modify: `docs/CHANGES_after_review.md`

- [x] **Step 1: Run all focused tests and sanity checks**

Run pytest, Python compilation, all three sanity checks, and table-consistency checks.

- [x] **Step 2: Review the complete diff**

Confirm that no source images were changed, no unrelated files were deleted, and paper edits are narrowly scoped.

- [ ] **Step 3: Commit and push the fork branch**

Create focused commits and push `fix/reproducibility-and-paper-consistency`.

- [ ] **Step 4: Create the pull request**

The PR body must contain a dedicated `Paper changes` section listing the exact sections/values/claims changed, plus test commands and before/after numerical comparison.
