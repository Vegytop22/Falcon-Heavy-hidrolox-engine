"""Streamlit app: Falcon Heavy reaction-propulsion simulator, LOX/RP-1 vs LOX/LH2.

Run with:  streamlit run app.py
"""
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from metodos import METHOD_NAMES
from modelo import (G0, LH2, PE_PC_SEA, PE_PC_VAC, RP1, V_ESC_SURFACE, Propellant, Vehicle,
                    build_segments, tsiolkovsky_dv)
from simulador import (convergence_study, exact_escape_time, first_step, observed_order,
                       scipy_crosscheck, simulate)

ORANGE, BLUE, AQUA, GRAY = "#eb6834", "#2a78d6", "#1baf7a", "#7d7c77"
GRID, MUTED = "#e9e8e4", "#6b6a66"
COLOR = {"A": ORANGE, "B": BLUE}
METHOD_COLOR = {"euler": GRAY, "rk4": AQUA}

st.set_page_config(page_title="Simulador de propulsión · Falcon Heavy", layout="wide")


def fmt(value, decimals=0):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    s = f"{value:,.{decimals}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def sci(value, digits=1):
    if value == 0 or value is None:
        return "0"
    mant, exp = f"{value:.{digits}e}".split("e")
    sup = str(int(exp)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mant.replace('.', ',')}·10{sup}"


def pct(rel):
    return f"{fmt(rel * 100, 4)} %" if rel * 100 >= 1e-3 else f"{sci(rel * 100)} %"


def style(fig, ytitle, xtitle="Tiempo (s)", height=340, log_y=False, log_x=False):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", separators=",.",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(title_text=xtitle, showgrid=False, type="log" if log_x else None)
    fig.update_yaxes(title_text=ytitle, gridcolor=GRID, type="log" if log_y else None,
                     exponentformat="power" if log_y else None)
    return fig


def thin(r, n=1500):
    idx = set(np.linspace(0, len(r.t) - 1, min(n, len(r.t))).astype(int).tolist())
    jumps = np.nonzero(np.diff(r.t) == 0.0)[0]
    idx.update(jumps.tolist())
    idx.update((jumps + 1).tolist())
    return np.array(sorted(idx))


STAGE_NAMES = ["Tramo 1: laterales + núcleo", "Tramo 2: núcleo central",
               "Tramo 3: segunda etapa", "Vuelo inercial"]
BODY_FILL, BODY_LINE = "#f4f3ef", "#4a4945"
FLAME_OUT, FLAME_IN = "#f5a623", "#ffe066"
GHOST_SECONDS = 60.0

CORE = [(-0.16, 0.00), (0.16, 0.00), (0.16, 0.60), (-0.16, 0.60)]
UPPER = [(-0.16, 0.60), (0.16, 0.60), (0.16, 0.84), (-0.16, 0.84)]
NOSE = [(-0.16, 0.84), (0.16, 0.84), (0.12, 0.93), (0.0, 1.0), (-0.12, 0.93)]
BOOSTER_R = [(0.20, 0.00), (0.50, 0.00), (0.50, 0.60), (0.35, 0.70), (0.20, 0.60)]
BOOSTER_L = [(-u, s) for u, s in BOOSTER_R]
NOZZLES = {0: [(-0.50, -0.20, 0.0), (-0.16, 0.16, 0.0), (0.20, 0.50, 0.0)],
           1: [(-0.16, 0.16, 0.0)],
           2: [(-0.12, 0.12, 0.60)],
           3: []}


def _polys(polys, cx, base, w, h, du=0.0):
    """Join polygons in rocket units (u across, s up) into one None-separated trace."""
    xs, ys = [], []
    for poly in polys:
        pts = poly + [poly[0]]
        sign = [1.0 if u >= 0 else -1.0 for u, _ in pts]
        xs += [cx + (u + du * sg) * w for (u, _), sg in zip(pts, sign)] + [None]
        ys += [base + s * h for _, s in pts] + [None]
    return xs or [None], ys or [None]


def _flames(stage, cx, base, w, h, scale):
    outer, inner = [], []
    for a, b, s0 in NOZZLES[stage]:
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        outer.append([(a, s0), (b, s0), (mid, s0 - 0.26 * scale)])
        inner.append([(mid - 0.5 * half, s0), (mid + 0.5 * half, s0), (mid, s0 - 0.15 * scale)])
    return _polys(outer, cx, base, w, h), _polys(inner, cx, base, w, h)


def rocket_animation(ra, rb, n_frames=90):
    """Two schematic rockets climbing side by side, with their stage separations.

    View only: it samples the simulated altitude and velocity of each Result. The
    pieces that separate are drawn where they came off, for GHOST_SECONDS; the model
    does not compute their trajectories.
    """
    runs = [(ra, 0.28, ORANGE, "middle left"), (rb, 0.72, BLUE, "middle right")]
    ends = np.cumsum([s.duration for s in ra.segments])
    t_max = float(min(ra.t[-1], rb.t[-1]))
    events = [e for t in ends for e in (t - 0.5, t + 0.5)]
    events += [r.escape["t"] for r, *_ in runs if r.escaped]
    times = np.unique(np.concatenate([np.linspace(0.0, t_max, n_frames),
                                      [e for e in events if 0.0 < e < t_max]]))

    sep_alt = {id(r): [np.interp(t, r.t, r.x) / 1e3 for t in ends[:2]] for r, *_ in runs}
    w = 0.075

    def stage_at(tt):
        return int(np.searchsorted(ends, tt, side="right"))

    def frame(tt, view):
        h = 0.22 * view
        stage = stage_at(tt)
        traces = []
        for r, cx, color, side in runs:
            alt = np.interp(tt, r.t, r.x) / 1e3
            vel = np.interp(tt, r.t, r.v)
            (fx, fy), (ix, iy) = _flames(stage, cx, alt, w, h, 1.0 if stage < 2 else 0.8)
            parts = {
                "boosters": [BOOSTER_L, BOOSTER_R] if stage == 0 else [],
                "core": [CORE] if stage <= 1 else [],
                "upper": [UPPER],
                "nose": [NOSE],
            }
            traces.append(go.Scatter(x=fx, y=fy, fill="toself", mode="lines",
                                     line=dict(width=0), fillcolor=FLAME_OUT))
            traces.append(go.Scatter(x=ix, y=iy, fill="toself", mode="lines",
                                     line=dict(width=0), fillcolor=FLAME_IN))
            for key in ("boosters", "core", "upper"):
                x, y = _polys(parts[key], cx, alt, w, h)
                traces.append(go.Scatter(x=x, y=y, fill="toself", mode="lines",
                                         line=dict(color=BODY_LINE, width=1),
                                         fillcolor=BODY_FILL))
            x, y = _polys(parts["nose"], cx, alt, w, h)
            traces.append(go.Scatter(x=x, y=y, fill="toself", mode="lines",
                                     line=dict(color=BODY_LINE, width=1), fillcolor=color))

            a1, a2 = sep_alt[id(r)]
            show_b = ends[0] <= tt < ends[0] + GHOST_SECONDS
            show_c = ends[1] <= tt < ends[1] + GHOST_SECONDS
            gb = _polys([BOOSTER_L, BOOSTER_R], cx, a1, w, h, du=0.18) if show_b \
                else ([None], [None])
            gc = _polys([CORE], cx, a2 - 0.02 * view, w, h) if show_c else ([None], [None])
            for gx, gy in (gb, gc):
                traces.append(go.Scatter(x=gx, y=gy, fill="toself", mode="lines",
                                         line=dict(color=BODY_LINE, width=1, dash="dot"),
                                         fillcolor=BODY_FILL, opacity=0.45))
            gl_x, gl_y, gl_t = [], [], []
            off = -0.06 if side == "middle left" else 0.06
            if show_b:
                gl_x.append(cx + off)
                gl_y.append(a1 + 0.3 * h)
                gl_t.append(f"laterales<br>t = {fmt(ends[0], 1)} s")
            if show_c:
                gl_x.append(cx + off)
                gl_y.append(a2 + 0.3 * h)
                gl_t.append(f"núcleo<br>t = {fmt(ends[1], 1)} s")
            traces.append(go.Scatter(x=gl_x or [None], y=gl_y or [None], text=gl_t or [""],
                                     mode="text", textposition=side,
                                     textfont=dict(size=10, color=MUTED)))

            status = STAGE_NAMES[stage]
            if r.escaped and tt >= r.escape["t"]:
                status += f"<br><b>v ≥ v_esc desde t = {fmt(r.escape['t'], 1)} s</b>"
            label = (f"<b>{r.label}</b><br>{fmt(vel)} m/s · {fmt(alt)} km<br>{status}")
            traces.append(go.Scatter(x=[cx], y=[alt + 1.04 * h], text=[label], mode="text",
                                     textposition="top center",
                                     textfont=dict(size=11, color=color)))
        return traces

    views, view = [], 30.0
    for tt in times:
        top = max(np.interp(tt, r.t, r.x) / 1e3 for r, *_ in runs)
        view = max(view, top / 0.55)
        views.append(view)

    ground = go.Scatter(x=[0, 1, 1, 0], y=[0, 0, -1e7, -1e7], fill="toself", mode="lines",
                        line=dict(color="#b9b7ae", width=1), fillcolor="#e9e8e4",
                        hoverinfo="skip")
    first = frame(times[0], views[0])
    n_dyn = len(first)

    def y_range(view):
        return [-0.05 * view, view]

    frames = [go.Frame(name=str(i), data=frame(tt, vw), traces=list(range(1, n_dyn + 1)),
                       layout=go.Layout(yaxis=dict(range=y_range(vw))))
              for i, (tt, vw) in enumerate(zip(times, views))]
    fig = go.Figure(data=[ground] + first, frames=frames)
    fig.update_traces(hoverinfo="skip", showlegend=False)
    fig.update_layout(
        height=480, margin=dict(l=10, r=10, t=40, b=10), separators=",.",
        plot_bgcolor="white", showlegend=False,
        xaxis=dict(range=[0, 1], visible=False, fixedrange=True),
        yaxis=dict(range=y_range(views[0]), title="Altura (km)", gridcolor=GRID,
                   zeroline=False, fixedrange=True),
        updatemenus=[dict(type="buttons", x=0, y=1.12, showactive=False, buttons=[
            dict(label="▶ Reproducir", method="animate",
                 args=[None, dict(frame=dict(duration=90, redraw=True), fromcurrent=True,
                                  transition=dict(duration=0))]),
            dict(label="⏸ Pausa", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate",
                                    transition=dict(duration=0))])])],
        sliders=[dict(currentvalue=dict(prefix="t = ", suffix=" s"), pad=dict(t=10),
                      steps=[dict(method="animate", label=fmt(tt, 0),
                                  args=[[str(i)], dict(mode="immediate",
                                                       frame=dict(duration=0, redraw=True),
                                                       transition=dict(duration=0))])
                             for i, tt in enumerate(times)])])
    return fig


@st.cache_data(show_spinner=False, max_entries=96)
def run(prop, vehicle, method, dt, gravity, mdot_factor, x0, v0, coast):
    return simulate(prop, vehicle, method, dt, gravity, mdot_factor, x0, v0, coast)


@st.cache_data(show_spinner=False, max_entries=16)
def run_convergence(prop, vehicle, gravity, dts):
    return convergence_study(prop, vehicle, gravity, dts)


@st.cache_data(show_spinner=False, max_entries=32)
def run_exact_escape(prop, vehicle, gravity, mdot_factor):
    return exact_escape_time(prop, vehicle, gravity, mdot_factor)


# ---------------------------------------------------------------- sidebar: inputs
def propellant_inputs(label, base, key):
    st.sidebar.markdown(f"**{label}**")
    c1, c2, c3 = st.sidebar.columns(3)
    tc = c1.number_input("T_c (K)", 1000.0, 5000.0, base.tc, 10.0, key=f"tc{key}")
    mmol = c2.number_input("M_mol (kg/mol)", 0.002, 0.040, base.mmol, 0.0005, format="%.4f",
                           key=f"mm{key}")
    gamma = c3.number_input("γ", 1.05, 1.60, base.gamma, 0.01, format="%.2f", key=f"g{key}")
    return Propellant(base.key, base.name, tc, mmol, gamma)


st.sidebar.title("Parámetros")
st.sidebar.caption("Valores por defecto: sección 5 de la guía del motor.")
prop_a = propellant_inputs("A · LOX/RP-1", RP1, "a")
prop_b = propellant_inputs("B · LOX/LH2", LH2, "b")

with st.sidebar.expander("Toberas (ec. 3)", expanded=False):
    pe_sea = st.number_input("pₑ/p_c tramos 1 y 2 (nivel del mar)", 1e-5, 0.5, PE_PC_SEA,
                             0.0005, format="%.4f")
    pe_vac = st.number_input("pₑ/p_c tramo 3 (vacío)", 1e-6, 0.05, PE_PC_VAC, 1e-5,
                             format="%.5e")

for lab, p in (("A", prop_a), ("B", prop_b)):
    st.sidebar.caption(
        f"{lab}: vₑ = {fmt(p.exhaust_velocity(pe_sea))} m/s (Isp {fmt(p.isp(pe_sea))} s) · "
        f"vacío {fmt(p.exhaust_velocity(pe_vac))} m/s (Isp {fmt(p.isp(pe_vac))} s)")

with st.sidebar.expander("Vehículo (Falcon Heavy)", expanded=True):
    d = Vehicle()
    payload = st.number_input("Carga útil (kg)", 0.0, 30_000.0, d.payload, 100.0)
    core_prop = st.number_input("Propelente por núcleo (kg)", 100_000.0, 600_000.0, d.core_prop,
                                1000.0)
    side_dry = st.number_input("Masa seca de cada lateral (kg)", 5_000.0, 60_000.0, d.side_dry,
                               100.0)
    core_dry = st.number_input("Masa seca del núcleo central (kg)", 5_000.0, 60_000.0,
                               d.core_dry, 100.0)
    upper_prop = st.number_input("Propelente 2.ª etapa (kg)", 10_000.0, 300_000.0, d.upper_prop,
                                 500.0)
    upper_dry = st.number_input("Masa seca 2.ª etapa (kg)", 500.0, 20_000.0, d.upper_dry, 100.0)
    fairing = st.number_input("Cofia (kg)", 0.0, 5_000.0, d.fairing, 50.0)
    merlin = st.number_input("Empuje Merlin 1D, nivel del mar (N)", 400_000.0, 1_200_000.0,
                             d.merlin_thrust, 5_000.0)
    mvac = st.number_input("Empuje Merlin Vacuum (N)", 400_000.0, 1_500_000.0, d.mvac_thrust,
                           5_000.0)
    throttle = st.slider("Régimen del núcleo central en el tramo 1", 0.40, 1.00, d.throttle, 0.05)
    mdot_factor = st.select_slider("Multiplicador del gasto másico ṁ", [0.6, 0.8, 1.0, 1.2, 1.4],
                                   1.0)

vehicle = Vehicle(core_prop, side_dry, core_dry, upper_prop, upper_dry, fairing, payload, 9,
                  merlin, mvac, throttle, pe_sea, pe_vac)

st.sidebar.markdown("**Condiciones iniciales**")
c1, c2 = st.sidebar.columns(2)
x0 = c1.number_input("x₀ (m)", 0.0, 100_000.0, 0.0, 100.0)
v0 = c2.number_input("v₀ (m/s)", 0.0, 3_000.0, 0.0, 10.0)
coast = st.sidebar.number_input("Vuelo inercial tras el apagado (s)", 0.0, 3_000.0, 300.0, 50.0)

st.sidebar.markdown("**Simulación**")
gravity = st.sidebar.radio("Gravedad", ["Con gravedad (g₀ = 9,81 m/s²)", "Sin gravedad"],
                           key="gravity").startswith("Con")
method = st.sidebar.radio("Método numérico", ["rk4", "euler"], format_func=METHOD_NAMES.get,
                          horizontal=True, key="method")
dt = st.sidebar.select_slider("Paso Δt (s)", [1.0, 0.1, 0.01], 0.1)

ra = run(prop_a, vehicle, method, dt, gravity, mdot_factor, x0, v0, coast)
rb = run(prop_b, vehicle, method, dt, gravity, mdot_factor, x0, v0, coast)
RES = {"A": ra, "B": rb}

# ---------------------------------------------------------------- header
st.title("Simulador de propulsión por reacción")
st.caption(f"Falcon Heavy · {prop_a.name} frente a {prop_b.name} · ascenso vertical, "
           f"{'gravedad constante' if gravity else 'sin gravedad'}, sin arrastre · "
           f"{METHOD_NAMES[method]} · Δt = {fmt(dt, 2)} s")

tw = build_segments(prop_a, vehicle, mdot_factor)[0].thrust / (vehicle.liftoff_mass * G0)
if gravity and tw < 1.0:
    st.error(f"Con estos parámetros el empuje inicial de A es menor que el peso (T/W = "
             f"{fmt(tw, 2)}): el cohete no despega y el modelo deja de ser válido.")

tabs = st.tabs(["Trayectoria", "Energía y conservación", "Métodos numéricos",
                "Comparación de escenarios"])


def escape_line(r):
    if r.escaped:
        return f"escapa en t = {fmt(r.escape['t'], 1)} s"
    return "no alcanza la velocidad de escape"


# ---------------------------------------------------------------- tab 1: trajectory
with tabs[0]:
    msg = (f"**{prop_a.name}** {escape_line(ra)} · **{prop_b.name}** {escape_line(rb)} "
           f"(criterio ε = v²/2 − g₀R²/(R + x) ≥ 0).")
    (st.success if ra.escaped and rb.escaped else st.warning)(msg)

    for k, r in RES.items():
        b = r.burnout
        cols = st.columns([1.3, 1, 1, 1, 1, 1])
        cols[0].markdown(f"<div style='margin-top:1.6rem;font-weight:600;color:{COLOR[k]}'>"
                         f"{k} · {r.label}</div>", unsafe_allow_html=True)
        cols[1].metric("Escape (s)", fmt(r.escape["t"], 1) if r.escaped else "no escapa")
        cols[2].metric("v al apagado (m/s)", fmt(b["v"]))
        cols[3].metric("v_esc a esa altura (m/s)", fmt(b["v_esc"]))
        cols[4].metric("Altura al apagado (km)", fmt(b["x"] / 1e3))
        cols[5].metric("Altura máxima (km)", fmt(r.max_altitude / 1e3))

    fig = go.Figure()
    for k, r in RES.items():
        i = thin(r)
        fig.add_scatter(x=r.t[i], y=r.v[i], name=f"v · {r.label}", line=dict(color=COLOR[k]))
        fig.add_scatter(x=r.t[i], y=r.v_escape[i], name=f"v_esc(x) · {r.label}",
                        line=dict(color=COLOR[k], dash="dash", width=1.3))
        if r.escaped:
            fig.add_scatter(x=[r.escape["t"]], y=[r.escape["v"]], mode="markers",
                            marker=dict(color=COLOR[k], size=11, line=dict(color="white",
                                                                           width=2)),
                            name=f"escape {r.label}", showlegend=False)
    fig.add_hline(y=V_ESC_SURFACE, line=dict(color=MUTED, dash="dot", width=1),
                  annotation_text="11.186 m/s (superficie)", annotation_position="top left")
    for s in ra.separations:
        fig.add_vline(x=s["t"], line=dict(color=GRID, width=1))
    st.subheader("Velocidad frente a la velocidad de escape local")
    st.plotly_chart(style(fig, "Velocidad (m/s)", height=380), width="stretch")

    c1, c2 = st.columns(2)
    fig = go.Figure()
    for k, r in RES.items():
        i = thin(r)
        fig.add_scatter(x=r.t[i], y=r.x[i] / 1e3, name=r.label, line=dict(color=COLOR[k]))
    c1.markdown("**Altura**")
    c1.plotly_chart(style(fig, "Altura (km)"), width="stretch")
    fig = go.Figure()
    i = thin(ra)
    fig.add_scatter(x=ra.t[i], y=ra.m[i] / 1e3, name="Masa", line=dict(color=GRAY),
                    showlegend=False)
    for s in ra.separations:
        fig.add_annotation(x=s["t"], y=np.log10(s["m_after"] / 1e3), text=f"−{fmt((s['m_before'] - s['m_after']) / 1e3, 1)} t",
                           showarrow=True, arrowhead=0, ax=55, ay=-18)
    c2.markdown("**Masa del vehículo** (igual para A y B: mismo ṁ y mismas masas)")
    c2.plotly_chart(style(fig, "Masa (t)", log_y=True), width="stretch")

    st.markdown("**Tramos de vuelo**")
    m = vehicle.liftoff_mass
    rows, t_acc = [], 0.0
    for sa, sb in zip(ra.segments, rb.segments):
        mf = m - sa.mdot * sa.duration
        t_acc += sa.duration
        rows.append({"Tramo": sa.name, "Duración (s)": fmt(sa.duration, 2),
                     "Termina en t (s)": fmt(t_acc, 2), "ṁ (kg/s)": fmt(sa.mdot, 2),
                     "Empuje A (MN)": fmt(sa.thrust / 1e6, 2),
                     "Empuje B (MN)": fmt(sb.thrust / 1e6, 2),
                     "Masa inicio → fin (kg)": f"{fmt(m)} → {fmt(mf)}",
                     "Se desprende (kg)": fmt(sa.drop)})
        m = mf - sa.drop
    rows.append({"Tramo": "Vuelo inercial", "Duración (s)": fmt(coast, 0), "ṁ (kg/s)": "0",
                 "Empuje A (MN)": "0", "Empuje B (MN)": "0",
                 "Masa inicio → fin (kg)": fmt(m), "Se desprende (kg)": "—",
                 "Termina en t (s)": fmt(t_acc + coast, 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("F = vₑ·ṁ (ec. 7). En cada separación se descuenta la masa seca sin cambiar "
               "x ni v. El último paso de cada tramo se acorta para terminar justo en el apagado.")

    st.markdown("**Interpretación de resultados**")
    gain = (rb.burnout["v"] / ra.burnout["v"] - 1.0) * 100.0
    loss = G0 * sum(s.duration for s in ra.segments) if gravity else 0.0
    lines = [
        f"Con la misma masa de propelente y el mismo gasto másico, {prop_b.name} termina la "
        f"propulsión con {fmt(gain, 0)} % más velocidad que {prop_a.name}: su vₑ es mayor porque los "
        f"gases de escape tienen menor masa molar (ec. 3).",
    ]
    if gravity:
        lines.append(f"La gravedad resta g₀·t_quema = {fmt(loss)} m/s a ambos por igual. Sin "
                     f"gravedad la velocidad final sería la suma de los tres términos de "
                     f"Tsiolkovsky: {fmt(tsiolkovsky_dv(ra.segments, vehicle))} m/s (A) y "
                     f"{fmt(tsiolkovsky_dv(rb.segments, vehicle))} m/s (B).")
    for r in (ra, rb):
        if r.escaped:
            margin = r.burnout["v"] - r.burnout["v_esc"]
            lines.append(f"{r.label} alcanza la velocidad de escape en t = "
                         f"{fmt(r.escape['t'], 1)} s, a {fmt(r.escape['x'] / 1e3)} km de altura; "
                         f"al apagar le sobran {fmt(margin)} m/s respecto a la v_esc local.")
        else:
            lines.append(f"{r.label} no escapa: al apagar le faltan "
                         f"{fmt(r.burnout['v_esc'] - r.burnout['v'])} m/s respecto a la v_esc "
                         f"local.")
    if ra.escaped and rb.escaped:
        lines.append(f"{prop_b.name} escapa {fmt(ra.escape['t'] - rb.escape['t'], 1)} s antes.")
    lines.append("Cada separación hace caer la masa y aumenta la relación de masas del tramo "
                 "siguiente; por eso la velocidad sube más rápido tras cada desprendimiento.")
    lines.append("Por qué la v_esc es local y no 11.186 m/s: la velocidad de escape disminuye con "
                 "la altura. Como el modelo usa g₀ constante (sobrestima la gravedad en altura), la "
                 "velocidad simulada es una cota inferior de la real; si cumple la condición exacta "
                 "de escape, el cohete real también escapa.")
    if method == "euler" and not ra.escaped and run(prop_a, vehicle, "rk4", dt, gravity,
                                                    mdot_factor, x0, v0, coast).escaped:
        lines.append(f"Atención: con Euler y Δt = {fmt(dt, 2)} s {prop_a.name} no escapa, pero "
                     f"con RK4 sí. El error del método cambia la conclusión física.")
    lines.append("Limitación: ascenso vertical con gravedad constante y sin arrastre; un "
                 "lanzamiento real se inclina hacia la horizontal y pierde menos velocidad por "
                 "gravedad. El motor de LOX/LH2 es hipotético y su ventaja es una cota superior.")
    st.markdown("\n".join(f"- {s}" for s in lines))

    buf = io.StringIO()
    pd.DataFrame({"t_s": ra.t, "x_A_m": ra.x, "v_A_ms": ra.v, "M_kg": ra.m,
                  "E_A_Jkg": ra.energy}).to_csv(buf, index=False)
    st.download_button("Exportar CSV (A)", buf.getvalue(), "trayectoria_A.csv", "text/csv")

# ---------------------------------------------------------------- tab 2: energy
with tabs[1]:
    other = "euler" if method == "rk4" else "rk4"
    alt = {k: run(r.propellant, vehicle, other, dt, gravity, mdot_factor, x0, v0, coast)
           for k, r in RES.items()}
    by_method = {method: RES, other: alt}

    def rel_res(r):
        return r.energy_residual.max() / max(abs(r.w[-1]), 1.0)

    def coast_var(r):
        e = r.energy[r.segment_index == len(r.segments)]
        return (e.max() - e.min()) / abs(e.mean()) if len(e) > 1 else 0.0

    c = st.columns(3)
    c[0].metric(f"Residuo relativo máx. · RK4 (Δt = {fmt(dt, 2)} s)",
                sci(rel_res(by_method["rk4"]["A"])), "balance de energía, todo el vuelo",
                delta_color="off", delta_arrow="off")
    c[1].metric(f"Residuo relativo máx. · Euler (Δt = {fmt(dt, 2)} s)",
                sci(rel_res(by_method["euler"]["A"])),
                f"{fmt(np.log10(rel_res(by_method['euler']['A']) / max(rel_res(by_method['rk4']['A']), 1e-300)))} órdenes peor",
                delta_color="off", delta_arrow="off")
    c[2].metric("E en vuelo inercial · RK4", sci(coast_var(by_method["rk4"]["A"])),
                "variación relativa (debe ser ≈ 0)", delta_color="off", delta_arrow="off")

    fig = go.Figure()
    for k, r in RES.items():
        i = thin(r)
        fig.add_scatter(x=r.t[i], y=r.energy[i] / 1e6, name=f"E · {r.label}",
                        line=dict(color=COLOR[k]))
        fig.add_scatter(x=r.t[i], y=r.kinetic[i] / 1e6, name=f"K · {r.label}",
                        line=dict(color=COLOR[k], dash="dash", width=1.2))
        fig.add_scatter(x=r.t[i], y=r.potential[i] / 1e6, name=f"U · {r.label}",
                        line=dict(color=COLOR[k], dash="dot", width=1.2))
    st.subheader("Energía mecánica específica E = K + U")
    st.caption("Crece mientras hay empuje y queda constante en el vuelo inercial.")
    st.plotly_chart(style(fig, "Energía (MJ/kg)", height=360), width="stretch")

    sel = st.radio("Propelente para la verificación", ["A", "B"], horizontal=True,
                   format_func=lambda k: RES[k].label)
    fig = go.Figure()
    for mth in ("euler", "rk4"):
        r = by_method[mth][sel]
        i = thin(r)
        fig.add_scatter(x=r.t[i], y=np.maximum(r.energy_residual[i], 1e-16),
                        name=METHOD_NAMES[mth], line=dict(color=METHOD_COLOR[mth]))
    st.subheader("Verificación de conservación")
    st.caption("Escala logarítmica. Euler acumula error; RK4 queda en el ruido de redondeo.")
    st.plotly_chart(style(fig, "|E(t) − E(0) − W(t)|  (J/kg)", log_y=True),
                    width="stretch")

    c1, c2 = st.columns([1, 1])
    c1.markdown(
        "**Qué se verifica.** Durante la quema E no se conserva: el empuje realiza trabajo. "
        "La ec. (24) dice exactamente cuánto cambia, así que lo que no cierre es error numérico.\n\n"
        "E = ½v² + g₀x  dE/dt = v·F/M (ec. 24)\n\n"
        "W(t) = ∫₀ᵗ v·F/M dt se integra junto con (x, v, M) con el mismo método. "
        "Residuo = |E(t) − E(0) − W(t)|.\n\n"
        "En la separación E no salta: es energía por unidad de masa y x, v no cambian. "
        "En el vuelo inercial (F = 0) E debe permanecer constante.")
    rows = []
    for k, r in RES.items():
        b = r.burnout
        rows.append({"Instante": f"Apagado · {r.label}", "K (MJ/kg)": fmt(b["v"] ** 2 / 2e6, 1),
                     "U (MJ/kg)": fmt(r.g * b["x"] / 1e6, 1),
                     "E (MJ/kg)": fmt((b["v"] ** 2 / 2 + r.g * b["x"]) / 1e6, 1),
                     "W (MJ/kg)": fmt(r.w[np.searchsorted(r.t, b["t"])] / 1e6, 1)})
    rows.append({"Instante": "Escape desde la superficie (ref.)",
                 "K (MJ/kg)": fmt(V_ESC_SURFACE ** 2 / 2e6, 1), "U (MJ/kg)": "0,0",
                 "E (MJ/kg)": fmt(V_ESC_SURFACE ** 2 / 2e6, 1), "W (MJ/kg)": "—"})
    c2.markdown("**Energía en el apagado**")
    c2.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ---------------------------------------------------------------- tab 3: numerical methods
with tabs[2]:
    sel3 = st.radio("Propelente", ["A", "B"], horizontal=True, key="sel3",
                    format_func=lambda k: RES[k].label)
    prop = RES[sel3].propellant
    dts_full = (8.0, 4.0, 2.0, 1.0, 0.5, 0.1, 0.01)
    rows = run_convergence(prop, vehicle, gravity, dts_full)
    fit = [r for r in rows if r["dt"] >= 0.5]
    order = {mth: np.polyfit(np.log([r["dt"] for r in fit if r["method"] == mth]),
                             np.log([r["err_abs_v"] for r in fit if r["method"] == mth]), 1)[0]
             for mth in ("euler", "rk4")}
    c = st.columns(3)
    c[0].metric("Orden observado · Euler", fmt(order["euler"], 2),
                "error ÷10 cuando Δt ÷10", delta_color="off", delta_arrow="off")
    c[1].metric("Orden observado · RK4", fmt(order["rk4"], 2),
                "ajuste con Δt de 8 a 0,5 s", delta_color="off", delta_arrow="off")
    c[2].metric("Paso que cae en cada apagado", "sí",
                f"último paso acortado en t = {fmt(sum(s.duration for s in RES[sel3].segments), 2)} s",
                delta_color="off", delta_arrow="off")

    fig = go.Figure()
    for mth in ("euler", "rk4"):
        d = [r for r in rows if r["method"] == mth]
        fig.add_scatter(x=[r["dt"] for r in d], y=[max(r["err_abs_v"], 1e-16) for r in d],
                        mode="lines+markers", name=METHOD_NAMES[mth],
                        line=dict(color=METHOD_COLOR[mth]))
    st.subheader("Convergencia de Euler y RK4")
    st.caption(f"Escala log-log · {prop.name} · error en la velocidad al apagado final frente a "
               f"la solución exacta (ecs. 10–18 aplicadas por tramos).")
    st.plotly_chart(style(fig, "Error absoluto (m/s)", "Δt (s)", log_y=True, log_x=True),
                    width="stretch")

    st.markdown(f"**Errores frente a la solución exacta** · v exacta al apagado = "
                f"{fmt(rows[0]['v_exact'], 3)} m/s")
    tab = []
    for dt_i in (1.0, 0.1, 0.01):
        e = {r["method"]: r for r in rows if r["dt"] == dt_i}
        tab.append({"Δt (s)": fmt(dt_i, 2),
                    "Euler E_abs (m/s)": fmt(e["euler"]["err_abs_v"], 3),
                    "Euler E_rel": sci(e["euler"]["err_rel_v"]),
                    "RK4 E_abs (m/s)": sci(e["rk4"]["err_abs_v"]),
                    "RK4 E_rel": sci(e["rk4"]["err_rel_v"]),
                    "Euler E_abs x (m)": fmt(e["euler"]["err_abs_x"], 1),
                    "RK4 E_abs x (m)": sci(e["rk4"]["err_abs_x"])})
    st.dataframe(pd.DataFrame(tab), hide_index=True, width="stretch")
    st.caption("Por debajo de Δt ≈ 0,1 s el error de RK4 (~10⁻⁸ m/s) ya es ruido de redondeo "
               "de la computadora acumulado en decenas de miles de pasos; por eso deja de bajar y "
               "el orden se ajusta con pasos mayores.")

    c1, c2 = st.columns(2)
    fs_rows = []
    for k, r in RES.items():
        fs = first_step(r.propellant, vehicle, 1.0, gravity)
        vx, xx = fs["exact"]
        for name, key in (("Solución exacta", "exact"), ("Euler", "euler"),
                          ("Runge-Kutta 4", "rk4")):
            v1, x1 = fs[key]
            fs_rows.append({"Propelente": r.label, "Método": name, "v(1) (m/s)": fmt(v1, 5),
                            "E_rel (v)": "—" if key == "exact" else pct(abs(v1 - vx) / vx),
                            "x(1) (m)": fmt(x1, 5),
                            "E_rel (x)": "—" if key == "exact" else pct(abs(x1 - xx) / xx)})
    c1.markdown("**Primera iteración (Δt = 1 s)** · Cuadro 2 del artículo")
    c1.dataframe(pd.DataFrame(fs_rows), hide_index=True, width="stretch")

    esc_rows = []
    for dt_i in (2.0, 1.0, 0.1, 0.01):
        row = {"Δt (s)": fmt(dt_i, 2)}
        for mth in ("euler", "rk4"):
            r = run(prop, vehicle, mth, dt_i, gravity, mdot_factor, x0, v0, coast)
            row[f"Escape {METHOD_NAMES[mth]} (s)"] = fmt(r.escape["t"], 2) if r.escaped \
                else "no escapa"
        esc_rows.append(row)
    ex = run_exact_escape(prop, vehicle, gravity, mdot_factor)
    c2.markdown(f"**El método puede cambiar la conclusión** · exacto: "
                f"{fmt(ex, 2) if ex else 'no escapa'} s")
    c2.dataframe(pd.DataFrame(esc_rows), hide_index=True, width="stretch")
    ref = scipy_crosscheck(prop, vehicle, gravity)
    c2.caption(f"Verificación independiente con SciPy (DOP853, solo como contraste): v al apagado "
               f"= {fmt(ref['v'], 4)} m/s; diferencia con RK4 (Δt = 0,01 s) = "
               f"{sci(abs(ref['v'] - rows[-1]['v']))} m/s.")

# ---------------------------------------------------------------- tab 4: scenarios
with tabs[3]:
    def summary(r, name):
        b = r.burnout
        return {"Escenario": name, "Escape (s)": fmt(r.escape["t"], 1) if r.escaped else "no escapa",
                "t apagado (s)": fmt(b["t"], 1), "v apagado (m/s)": fmt(b["v"]),
                "v_esc local (m/s)": fmt(b["v_esc"]), "Margen (m/s)": fmt(b["v"] - b["v_esc"]),
                "Altura apagado (km)": fmt(b["x"] / 1e3), "_t": r.escape["t"] if r.escaped else None,
                "_frac": b["v"] / b["v_esc"]}

    scen = []
    for k, r in RES.items():
        p = r.propellant
        scen.append(summary(r, f"{p.name}"))
        scen.append(summary(run(p, vehicle, method, dt, not gravity, mdot_factor, x0, v0, coast),
                            f"{p.name} · {'sin' if gravity else 'con'} gravedad"))
        for fct in (0.8, 1.2):
            scen.append(summary(run(p, vehicle, method, dt, gravity, mdot_factor * fct, x0, v0,
                                    coast), f"{p.name} · ṁ × {fmt(mdot_factor * fct, 2)}"))
    df = pd.DataFrame(scen)

    fig = go.Figure()
    fig.add_bar(x=df["Escenario"], y=df["_frac"] * 100,
                marker_color=[ORANGE if "RP-1" in s else BLUE for s in df["Escenario"]],
                marker_opacity=[0.45 if "gravedad" in s else 1.0 for s in df["Escenario"]],
                text=[f"{fmt(v * 100)} %" for v in df["_frac"]], textposition="outside")
    fig.add_hline(y=100, line=dict(color=MUTED, dash="dash"))
    st.subheader("Velocidad al apagado respecto a la v_esc local")
    st.caption("Por encima de 100 % el vehículo ya escapó al apagar · barras claras: el caso de "
               "gravedad contrario al elegido en la barra lateral.")
    st.plotly_chart(style(fig, "v / v_esc (%)", "", height=360).update_layout(hovermode=False),
                    width="stretch")
    st.dataframe(df.drop(columns=["_t", "_frac"]), hide_index=True, width="stretch")
    st.caption("ṁ × k cambia el gasto másico en los tres tramos (Proyecto 11: distintas tasas de "
               "consumo); los tiempos de quema cambian como 1/k.")

    c1, c2 = st.columns(2)
    payloads = np.arange(0, 7001, 500)
    fig = go.Figure()
    for k, r in RES.items():
        tt = [run_exact_escape(r.propellant, vehicle.with_(payload=float(p)), gravity,
                               mdot_factor) for p in payloads]
        fig.add_scatter(x=payloads / 1e3, y=[np.nan if t is None else t for t in tt],
                        mode="lines+markers", name=r.label, line=dict(color=COLOR[k]))
    c1.markdown("**Sensibilidad a la carga útil** (solución exacta)")
    c1.caption("Los huecos indican que con esa carga el propelente ya no escapa.")
    c1.plotly_chart(style(fig, "Instante de escape (s)", "Carga útil (t)"),
                    width="stretch")

    c2.markdown("**Evolución temporal**")
    c2.caption("Ambos cohetes suben a la vez; cada uno pierde los laterales y luego el núcleo "
               "central al final de su tramo. Las piezas separadas se dibujan durante un minuto "
               "donde se desprendieron: el modelo no calcula su trayectoria. La escala vertical "
               "se ajusta a medida que suben.")
    c2.plotly_chart(rocket_animation(ra, rb), width="stretch")

    st.markdown("**Escenarios guardados**")
    if "saved" not in st.session_state:
        st.session_state.saved = []
    if st.button("Guardar escenario actual"):
        tag = (f"{METHOD_NAMES[method]}, Δt {fmt(dt, 2)} s, carga {fmt(payload)} kg, "
               f"ṁ × {fmt(mdot_factor, 1)}, {'con' if gravity else 'sin'} g")
        for r in (ra, rb):
            st.session_state.saved.append({k: v for k, v in summary(r, f"{r.label} · {tag}").items()
                                           if not k.startswith("_")})
    if st.session_state.saved:
        st.dataframe(pd.DataFrame(st.session_state.saved), hide_index=True,
                     width="stretch")
        if st.button("Borrar escenarios guardados"):
            st.session_state.saved = []
            st.rerun()
    else:
        st.caption("Cambia parámetros en la barra lateral y guarda escenarios para compararlos aquí.")

st.caption("Modelo: ascenso vertical 1D, gravedad constante, sin arrastre, tres tramos propulsados "
           "y vuelo inercial. Euler y RK4 programados a mano; SciPy solo como contraste.")
