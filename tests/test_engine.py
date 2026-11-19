import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metodos import step_sizes
from modelo import (G0, LH2, PE_PC_SEA, PE_PC_VAC, RP1, Vehicle, build_segments, exact_segment,
                    tsiolkovsky_dv)
from simulador import (convergence_study, exact_escape_time, first_step, scipy_crosscheck,
                       simulate)


@pytest.fixture(scope="module")
def runs():
    return {p.key: simulate(p, dt=0.01) for p in (RP1, LH2)}


def test_exhaust_velocities_match_locked_values():
    assert RP1.exhaust_velocity(PE_PC_SEA) == pytest.approx(2854.59, abs=0.01)
    assert RP1.isp(PE_PC_VAC) == pytest.approx(348.00, abs=0.01)
    assert LH2.exhaust_velocity(PE_PC_SEA) == pytest.approx(4148.56, abs=0.01)
    assert LH2.exhaust_velocity(PE_PC_VAC) == pytest.approx(4877.92, abs=0.02)


def test_segment_table():
    v = Vehicle()
    assert v.liftoff_mass == pytest.approx(1_417_550.0)
    s = build_segments(RP1, v)
    assert [round(x.mdot, 2) for x in s] == [6926.73, 2664.13, 273.59]
    assert [round(x.duration, 2) for x in s] == [154.27, 61.71, 392.93]
    assert [x.drop for x in s] == [44_400.0, 25_600.0, 0.0]


def test_same_mass_flow_for_both_propellants():
    a, b = build_segments(RP1), build_segments(LH2)
    assert [x.mdot for x in a] == [x.mdot for x in b]


def test_masses_at_segment_ends(runs):
    r = runs["rp1"]
    seps = r.separations
    assert seps[0]["m_before"] == pytest.approx(348_950.0, abs=1.0)
    assert seps[0]["m_after"] == pytest.approx(304_550.0, abs=1.0)
    assert seps[1]["m_before"] == pytest.approx(140_150.0, abs=1.0)
    assert seps[1]["m_after"] == pytest.approx(114_550.0, abs=1.0)
    assert r.burnout["m"] == pytest.approx(7_050.0, abs=1.0)


@pytest.mark.parametrize("key, t_esc", [("rp1", 607.6), ("lh2", 488.9)])
def test_escape_instants(runs, key, t_esc):
    assert runs[key].escape["t"] == pytest.approx(t_esc, abs=0.5)


def test_reference_velocities(runs):
    assert runs["rp1"].burnout["v"] == pytest.approx(9761.4, abs=1.0)
    assert runs["lh2"].burnout["v"] == pytest.approx(16661.3, abs=1.0)
    assert runs["rp1"].burnout["x"] / 1e3 == pytest.approx(2265.7, abs=0.5)


def test_numeric_escape_matches_exact(runs):
    for p in (RP1, LH2):
        assert runs[p.key].escape["t"] == pytest.approx(exact_escape_time(p), abs=1e-3)


def test_payload_sensitivity():
    assert exact_escape_time(RP1, Vehicle(payload=3000.0)) is None
    assert exact_escape_time(LH2, Vehicle(payload=5000.0)) == pytest.approx(503.6, abs=0.5)


def test_last_step_lands_on_burnout():
    steps = step_sizes(154.2689, 0.1)
    assert sum(steps) == pytest.approx(154.2689, abs=1e-12)
    assert steps[-1] <= 0.1


@pytest.mark.parametrize("method, order, tol", [("euler", 1.0, 0.1), ("rk4", 4.0, 0.15)])
def test_observed_order(method, order, tol):
    dts = (4.0, 2.0, 1.0, 0.5)
    rows = convergence_study(RP1, dts=dts, methods=(method,))
    err = [r["err_abs_v"] for r in rows]
    slope = np.polyfit(np.log(dts), np.log(err), 1)[0]
    assert slope == pytest.approx(order, abs=tol)


def test_energy_balance_rk4():
    r = simulate(RP1, method="rk4", dt=0.1)
    assert r.energy_residual.max() / r.w[-1] < 1e-8
    coast = r.energy[r.segment_index == len(r.segments)]
    assert (coast.max() - coast.min()) / coast.mean() < 1e-10


def test_energy_balance_euler_is_worse():
    re = simulate(RP1, method="euler", dt=0.1)
    rr = simulate(RP1, method="rk4", dt=0.1)
    assert re.energy_residual.max() > 1e6 * rr.energy_residual.max()


def test_no_gravity_equals_tsiolkovsky():
    r = simulate(RP1, dt=0.1, gravity=False, coast_time=0.0)
    assert r.burnout["v"] == pytest.approx(tsiolkovsky_dv(r.segments), rel=1e-10)


def test_exact_segment_without_gravity_is_tsiolkovsky():
    x, v, m = exact_segment(10.0, 0.0, 0.0, 1000.0, 50.0, 3000.0, 0.0)
    assert v == pytest.approx(3000.0 * math.log(1000.0 / 500.0))
    assert m == pytest.approx(500.0)


def test_first_step_table():
    fs = first_step(RP1)
    assert fs["exact"][0] == pytest.approx(4.1729, abs=1e-4)
    assert fs["euler"][1] == 0.0
    assert fs["rk4"][0] == pytest.approx(fs["exact"][0], abs=1e-8)


def test_mass_flow_multiplier_changes_burn_time():
    a = build_segments(RP1, mdot_factor=1.0)
    b = build_segments(RP1, mdot_factor=0.8)
    assert sum(s.duration for s in b) == pytest.approx(sum(s.duration for s in a) / 0.8)


def test_scipy_crosscheck(runs):
    ref = scipy_crosscheck(RP1)
    assert ref["v"] == pytest.approx(runs["rp1"].burnout["v"], abs=1e-4)


def test_initial_conditions_are_used():
    r = simulate(RP1, dt=0.5, x0=1000.0, v0=50.0, coast_time=0.0)
    assert r.x[0] == 1000.0 and r.v[0] == 50.0
    assert G0 > 0
