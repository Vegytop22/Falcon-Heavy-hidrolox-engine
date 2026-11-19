"""Hand-written numerical methods: explicit Euler and classical fourth-order Runge-Kutta.

Both advance the whole state vector y = (x, v, M, W) with the same signature,
step(f, t, y, h, *args) -> y_next. States are tuples of floats for speed.
"""
import math


def euler_step(f, t, y, h, *args):
    k1 = f(t, y, *args)
    return tuple(yi + h * ki for yi, ki in zip(y, k1))


def rk4_step(f, t, y, h, *args):
    k1 = f(t, y, *args)
    y2 = tuple(yi + 0.5 * h * ki for yi, ki in zip(y, k1))
    k2 = f(t + 0.5 * h, y2, *args)
    y3 = tuple(yi + 0.5 * h * ki for yi, ki in zip(y, k2))
    k3 = f(t + 0.5 * h, y3, *args)
    y4 = tuple(yi + h * ki for yi, ki in zip(y, k3))
    k4 = f(t + h, y4, *args)
    return tuple(yi + h / 6.0 * (a + 2.0 * b + 2.0 * c + d)
                 for yi, a, b, c, d in zip(y, k1, k2, k3, k4))


METHODS = {"euler": euler_step, "rk4": rk4_step}
METHOD_NAMES = {"euler": "Euler", "rk4": "Runge-Kutta 4"}
ORDER = {"euler": 1, "rk4": 4}


def step_sizes(duration, dt):
    """Steps of size dt covering `duration`; the last one is shortened to land exactly on it."""
    if duration <= 0.0:
        return []
    n = max(1, math.ceil(duration / dt - 1e-9))
    return [dt] * (n - 1) + [duration - (n - 1) * dt]


def integrate_segment(f, t0, y0, duration, dt, method, args, on_step=None):
    """Integrate one segment with a fixed step, ending exactly at t0 + duration.

    `on_step(t_prev, y_prev, t, y, h)` is called after every step; returning True stops
    the integration early (used to cut the coast). Returns the lists of times and states.
    """
    step = METHODS[method]
    ts, ys = [t0], [y0]
    t, y = t0, y0
    for h in step_sizes(duration, dt):
        y_new = step(f, t, y, h, *args)
        t_new = t + h
        ts.append(t_new)
        ys.append(y_new)
        stop = on_step(t, y, t_new, y_new, h) if on_step else False
        t, y = t_new, y_new
        if stop:
            break
    return ts, ys


def bisect_event(f, t, y, h, method, args, event, tol=1e-9, max_iter=80):
    """Locate the time inside [t, t + h] where event(y) changes sign from negative.

    The step is repeated from (t, y) with the same method and a shorter size, so the
    event error has the same order as the integrator.
    """
    step = METHODS[method]
    lo, hi = 0.0, h
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if event(step(f, t, y, mid, *args)) >= 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    y_hit = step(f, t, y, hi, *args)
    return t + hi, y_hit
