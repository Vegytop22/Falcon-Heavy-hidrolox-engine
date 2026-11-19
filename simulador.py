"""Segment-by-segment simulator, exact reference solution and numerical verification."""
import math
from dataclasses import dataclass, field

import numpy as np

from modelo import (G0, RP1, Vehicle, build_segments, escape_velocity, exact_state,
                    orbital_energy, rhs, specific_energy, tsiolkovsky_dv)
from metodos import METHODS, bisect_event, integrate_segment, step_sizes


@dataclass
class Result:
    label: str
    propellant: object
    vehicle: Vehicle
    segments: list
    method: str
    dt: float
    g: float
    t: np.ndarray
    x: np.ndarray
    v: np.ndarray
    m: np.ndarray
    w: np.ndarray
    segment_index: np.ndarray
    separations: list
    burnout: dict
    escape: dict = None
    x0: float = 0.0
    v0: float = 0.0
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def escaped(self):
        return self.escape is not None

    @property
    def kinetic(self):
        return 0.5 * self.v ** 2

    @property
    def potential(self):
        return self.g * self.x

    @property
    def energy(self):
        return self.kinetic + self.potential

    @property
    def energy_residual(self):
        """|E(t) - E(0) - W(t)|: zero in the continuous model, so it is pure numerical error."""
        return np.abs(self.energy - self.energy[0] - self.w)

    @property
    def v_escape(self):
        if "v_esc" not in self._cache:
            self._cache["v_esc"] = np.sqrt(2.0 * G0 * 6.371e6 ** 2 / (6.371e6 + self.x))
        return self._cache["v_esc"]

    @property
    def thrust(self):
        mdot = np.array([s.mdot for s in self.segments] + [0.0])[self.segment_index]
        ve = np.array([s.ve for s in self.segments] + [0.0])[self.segment_index]
        return mdot * ve

    @property
    def max_altitude(self):
        return float(self.x.max())

    def exact(self):
        """Exact solution sampled on the same time grid."""
        if "exact" not in self._cache:
            arr = np.array([exact_state(ti, self.segments, self.g, self.x0, self.v0,
                                        vehicle=self.vehicle) for ti in self.t])
            self._cache["exact"] = arr
        return self._cache["exact"]


def simulate(propellant, vehicle=Vehicle(), method="rk4", dt=0.1, gravity=True,
             mdot_factor=1.0, x0=0.0, v0=0.0, coast_time=300.0, label=None):
    segments = build_segments(propellant, vehicle, mdot_factor)
    g = G0 if gravity else 0.0
    f = rhs
    y = (x0, v0, vehicle.liftoff_mass, 0.0)
    t = 0.0
    ts, ys, idx = [t], [y], [0]
    separations = []
    escape = {}

    def event(state):
        return orbital_energy(state[0], state[1])

    def make_watch(args):
        def watch(t_prev, y_prev, t_new, y_new, h):
            if not escape and event(y_prev) < 0.0 <= event(y_new):
                te, ye = bisect_event(f, t_prev, y_prev, h, method, args, event)
                escape.update(t=te, x=ye[0], v=ye[1], v_esc=escape_velocity(ye[0]))
            return False
        return watch

    for k, seg in enumerate(segments):
        args = (seg.mdot, seg.ve, g)
        seg_t, seg_y = integrate_segment(f, t, y, seg.duration, dt, method, args,
                                         make_watch(args))
        ts += seg_t[1:]
        ys += seg_y[1:]
        idx += [k] * (len(seg_t) - 1)
        t, y = seg_t[-1], seg_y[-1]
        if seg.drop > 0.0:
            separations.append(dict(t=t, x=y[0], v=y[1], m_before=y[2],
                                    m_after=y[2] - seg.drop, name=seg.name))
            y = (y[0], y[1], y[2] - seg.drop, y[3])
            ts.append(t)
            ys.append(y)
            idx.append(k + 1)

    burnout = dict(t=t, x=y[0], v=y[1], m=y[2], v_esc=escape_velocity(y[0]))

    coast_args = (0.0, 0.0, g)
    seg_t, seg_y = integrate_segment(f, t, y, coast_time, dt, method, coast_args,
                                     make_watch(coast_args))
    ts += seg_t[1:]
    ys += seg_y[1:]
    idx += [len(segments)] * (len(seg_t) - 1)

    arr = np.array(ys)
    return Result(label=label or propellant.name, propellant=propellant, vehicle=vehicle,
                  segments=segments, method=method, dt=dt, g=g, t=np.array(ts),
                  x=arr[:, 0], v=arr[:, 1], m=arr[:, 2], w=arr[:, 3],
                  segment_index=np.array(idx), separations=separations, burnout=burnout,
                  escape=escape or None, x0=x0, v0=v0)


def exact_escape_time(propellant, vehicle=Vehicle(), gravity=True, mdot_factor=1.0,
                      x0=0.0, v0=0.0, coast_time=300.0):
    """Escape instant from the analytic solution (dense scan plus bisection)."""
    segments = build_segments(propellant, vehicle, mdot_factor)
    g = G0 if gravity else 0.0
    t_end = sum(s.duration for s in segments) + coast_time

    def eps(ti):
        x, v, _ = exact_state(ti, segments, g, x0, v0, vehicle=vehicle)
        return orbital_energy(x, v)

    grid = np.arange(0.0, t_end + 1.0, 1.0)
    prev = eps(grid[0])
    for a, b in zip(grid[:-1], grid[1:]):
        cur = eps(b)
        if prev < 0.0 <= cur:
            for _ in range(80):
                mid = 0.5 * (a + b)
                if eps(mid) >= 0.0:
                    b = mid
                else:
                    a = mid
            return b
        prev = cur
    return None


def convergence_study(propellant, vehicle=Vehicle(), gravity=True, dts=(1.0, 0.1, 0.01),
                      methods=("euler", "rk4")):
    """Errors at final burnout against the exact solution for several step sizes."""
    segments = build_segments(propellant, vehicle)
    g = G0 if gravity else 0.0
    t_bo = sum(s.duration for s in segments)
    x_ex, v_ex, _ = exact_state(t_bo, segments, g, vehicle=vehicle)
    rows = []
    for method in methods:
        for dt in dts:
            r = simulate(propellant, vehicle, method, dt, gravity, coast_time=0.0)
            ev = abs(r.burnout["v"] - v_ex)
            ex = abs(r.burnout["x"] - x_ex)
            rows.append(dict(method=method, dt=dt, v=r.burnout["v"], v_exact=v_ex,
                             err_abs_v=ev, err_rel_v=ev / abs(v_ex),
                             x=r.burnout["x"], x_exact=x_ex,
                             err_abs_x=ex, err_rel_x=ex / abs(x_ex),
                             energy_residual=float(r.energy_residual.max())))
    return rows


def observed_order(rows, method, key="err_abs_v", dt_a=1.0, dt_b=0.1):
    e = {r["dt"]: r[key] for r in rows if r["method"] == method}
    if e.get(dt_a, 0) <= 0 or e.get(dt_b, 0) <= 0:
        return float("nan")
    return math.log(e[dt_a] / e[dt_b]) / math.log(dt_a / dt_b)


def first_step(propellant, vehicle=Vehicle(), h=1.0, gravity=True):
    """One step of each method from rest, against the exact solution (article Cuadro 2)."""
    seg = build_segments(propellant, vehicle)[0]
    g = G0 if gravity else 0.0
    args = (seg.mdot, seg.ve, g)
    y0 = (0.0, 0.0, vehicle.liftoff_mass, 0.0)
    xe, ve, _ = exact_state(h, [seg], g, vehicle=vehicle)
    out = {"exact": (ve, xe)}
    for name, step in METHODS.items():
        y = step(rhs, 0.0, y0, h, *args)
        out[name] = (y[1], y[0])
    return out


def scipy_crosscheck(propellant, vehicle=Vehicle(), gravity=True):
    """Independent check with SciPy's DOP853 (never used as the integrator)."""
    from scipy.integrate import solve_ivp

    segments = build_segments(propellant, vehicle)
    g = G0 if gravity else 0.0
    y = [0.0, 0.0, vehicle.liftoff_mass, 0.0]
    t = 0.0
    for seg in segments:
        sol = solve_ivp(lambda tt, yy: rhs(tt, yy, seg.mdot, seg.ve, g),
                        (t, t + seg.duration), y, method="DOP853", rtol=1e-12, atol=1e-9)
        y = list(sol.y[:, -1])
        y[2] -= seg.drop
        t += seg.duration
    return dict(t=t, x=y[0], v=y[1])


def payload_sensitivity(propellant, payloads, vehicle=Vehicle(), gravity=True):
    return [(p, exact_escape_time(propellant, vehicle.with_(payload=p), gravity))
            for p in payloads]


def no_gravity_check(propellant, vehicle=Vehicle()):
    segments = build_segments(propellant, vehicle)
    return tsiolkovsky_dv(segments, vehicle)


def step_count(result):
    return sum(len(step_sizes(s.duration, result.dt)) for s in result.segments)
