from pathlib import Path
import re
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cg_solvers


def test_problem3_sanity_uses_supported_debias_keyword():
    source = (REPO_ROOT / "exp/problem3/sanity_check.py").read_text(encoding="utf-8")
    assert "do_debias=True" in source
    assert re.search(r"(?<!do_)debias=True", source) is None


def test_gpsr_bb_uses_projected_direction_curvature():
    assert hasattr(cg_solvers, "projected_bb_stepsize")
    direction = np.array([2.0, -1.0])
    b_direction = np.array([8.0, -1.0])
    curvature = float(direction @ b_direction)
    expected = float(direction @ direction / curvature)

    actual = cg_solvers.projected_bb_stepsize(
        direction, curvature, alpha_min=1e-10, alpha_max=1e6
    )

    assert actual == expected


def test_gpsr_bb_solver_runs_with_reference_spectral_update():
    matrix = np.array([[1.0, 0.5]])
    observation = np.array([1.0])

    result = cg_solvers.gpsr_bb_solve(
        matrix, observation, tau=0.1, maxit=20, tolP=1e-6,
        spectral_update="projected",
    )

    assert np.all(np.isfinite(result["x"]))
    assert np.all(np.isfinite(result["hist_f"]))


def test_gpsr_bb_keeps_repository_secant_update_as_default():
    source = (REPO_ROOT / "src/cg_solvers.py").read_text(encoding="utf-8")
    assert 'spectral_update="secant"' in source


class ShiftedQuadratic:
    def value(self, x):
        residual = x - 1.0
        return 0.5 * float(residual @ residual)

    def gradient(self, x):
        return x - 1.0

    def set_mu(self, _mu):
        pass


def test_nonlinear_cg_reports_safeguard_diagnostics():
    result = cg_solvers.nonlinear_cg(
        ShiftedQuadratic(), np.array([3.0, -2.0]), beta="prp+", maxit=10
    )

    for key in (
        "negative_beta_count",
        "beta_truncation_count",
        "descent_restart_count",
        "line_search_fallback_count",
    ):
        assert key in result
        assert result[key] >= 0


def test_raw_prp_comparison_changes_only_beta_rule():
    source = (REPO_ROOT / "exp/problem3/run_problem3.py").read_text(encoding="utf-8")
    assert '("prp", True, True, "CG-PRP无截断")' in source


def test_continuation_aggregates_safeguard_diagnostics():
    result = cg_solvers.cg_with_mu_continuation(
        ShiftedQuadratic(), np.array([3.0, -2.0]), mu_seq=(1e-1, 1e-2), maxit=10
    )

    assert len(result["stage_diagnostics"]) == 2
    assert result["beta_truncation_count"] == sum(
        stage["beta_truncation_count"] for stage in result["stage_diagnostics"]
    )


def test_problem3_results_export_safeguard_diagnostics():
    source = (REPO_ROOT / "exp/problem3/run_problem3.py").read_text(encoding="utf-8")
    for field in (
        "negative_beta_count",
        "beta_truncation_count",
        "descent_restart_count",
        "line_search_fallback_count",
    ):
        assert f'"{field}"' in source
