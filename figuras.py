"""Article figures (PDF + PNG) and CSV tables.

Run with:  python figuras.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modelo import LH2, RP1, V_ESC_SURFACE, Vehicle
from simulador import convergence_study, exact_escape_time, simulate

OUT = Path(__file__).resolve().parent / "figuras"
COLOR = {"rp1": "#eb6834", "lh2": "#2a78d6"}
METHOD_COLOR = {"euler": "#7d7c77", "rk4": "#1baf7a"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e9e8e4", "figure.dpi": 150,
                     "axes.formatter.use_locale": False})


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    runs = {p.key: simulate(p, dt=0.01) for p in (RP1, LH2)}

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for k, r in runs.items():
        ax.plot(r.t, r.v, color=COLOR[k], label=f"v · {r.label}")
        ax.plot(r.t, r.v_escape, color=COLOR[k], ls="--", lw=1, label=f"$v_{{esc}}(x)$ · {r.label}")
        if r.escaped:
            ax.plot(r.escape["t"], r.escape["v"], "o", color=COLOR[k], mec="white")
            ax.annotate(f"escape: {r.escape['t']:.1f} s".replace(".", ","),
                        (r.escape["t"], r.escape["v"]),
                        xytext=(-110, 45) if k == "lh2" else (30, -40),
                        textcoords="offset points", color=COLOR[k], fontsize=9,
                        arrowprops=dict(arrowstyle="-", color=COLOR[k], lw=0.8))
    ax.axhline(V_ESC_SURFACE, color="#6b6a66", ls=":", lw=1, label="11 186 m/s (superficie)")
    ax.set(xlabel="Tiempo (s)", ylabel="Velocidad (m/s)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    save(fig, "fig1_velocidad_escape")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.4))
    for k, r in runs.items():
        a1.plot(r.t, r.x / 1e3, color=COLOR[k], label=r.label)
    a1.set(xlabel="Tiempo (s)", ylabel="Altura (km)")
    a1.legend(frameon=False)
    r = runs["rp1"]
    a2.semilogy(r.t, r.m / 1e3, color="#7d7c77")
    for s in r.separations:
        a2.axvline(s["t"], color="#cccccc", lw=0.8)
    a2.set(xlabel="Tiempo (s)", ylabel="Masa (t)")
    save(fig, "fig2_altura_masa")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for k, r in runs.items():
        ax.plot(r.t, r.energy / 1e6, color=COLOR[k], label=f"E · {r.label}")
        ax.plot(r.t, r.kinetic / 1e6, color=COLOR[k], ls="--", lw=1, label=f"K · {r.label}")
        ax.plot(r.t, r.potential / 1e6, color=COLOR[k], ls=":", lw=1, label=f"U · {r.label}")
    ax.set(xlabel="Tiempo (s)", ylabel="Energía (MJ/kg)")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    save(fig, "fig3_energia")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for m in ("euler", "rk4"):
        r = simulate(RP1, method=m, dt=0.1)
        ax.semilogy(r.t, np.maximum(r.energy_residual, 1e-16), color=METHOD_COLOR[m],
                    label="Euler" if m == "euler" else "RK4")
    ax.set(xlabel="Tiempo (s)", ylabel=r"$|E(t)-E(0)-W(t)|$ (J/kg)",
           title="LOX/RP-1, Δt = 0,1 s")
    ax.legend(frameon=False)
    save(fig, "fig4_residuo_energia")

    dts = (8.0, 4.0, 2.0, 1.0, 0.5, 0.1, 0.01)
    rows = convergence_study(RP1, dts=dts)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for m in ("euler", "rk4"):
        d = [r for r in rows if r["method"] == m]
        ax.loglog([r["dt"] for r in d], [max(r["err_abs_v"], 1e-16) for r in d], "o-",
                  color=METHOD_COLOR[m], label="Euler" if m == "euler" else "RK4")
    ax.set(xlabel="Δt (s)", ylabel="Error absoluto en v al apagado (m/s)")
    ax.legend(frameon=False)
    save(fig, "fig5_convergencia")
    pd.DataFrame(rows).to_csv(OUT / "tabla_convergencia.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for k, p in (("rp1", RP1), ("lh2", LH2)):
        for grav, ls in ((True, "-"), (False, "--")):
            r = simulate(p, dt=0.1, gravity=grav, coast_time=0.0)
            ax.plot(r.t, r.v, color=COLOR[k], ls=ls,
                    label=f"{p.name} · {'con' if grav else 'sin'} gravedad")
    ax.set(xlabel="Tiempo (s)", ylabel="Velocidad (m/s)")
    ax.legend(fontsize=8, frameon=False)
    save(fig, "fig6_efecto_gravedad")

    seg_rows, m = [], Vehicle().liftoff_mass
    for s in runs["rp1"].segments:
        mf = m - s.mdot * s.duration
        seg_rows.append(dict(tramo=s.name, duracion_s=s.duration, mdot_kg_s=s.mdot,
                             empuje_rp1_N=s.thrust,
                             empuje_lh2_N=s.mdot * runs["lh2"].segments[len(seg_rows)].ve,
                             masa_inicio_kg=m, masa_fin_kg=mf, masa_desprendida_kg=s.drop))
        m = mf - s.drop
    pd.DataFrame(seg_rows).to_csv(OUT / "tabla_tramos.csv", index=False)

    res = []
    for k, r in runs.items():
        b = r.burnout
        res.append(dict(propelente=r.label, t_escape_s=r.escape["t"] if r.escaped else None,
                        t_apagado_s=b["t"], v_apagado_m_s=b["v"], x_apagado_km=b["x"] / 1e3,
                        v_esc_local_m_s=b["v_esc"], margen_m_s=b["v"] - b["v_esc"]))
    pd.DataFrame(res).to_csv(OUT / "tabla_resultados.csv", index=False)

    payloads = [0, 1300, 3000, 5000]
    pd.DataFrame([dict(carga_util_kg=p,
                       t_escape_rp1_s=exact_escape_time(RP1, Vehicle(payload=p)),
                       t_escape_lh2_s=exact_escape_time(LH2, Vehicle(payload=p)))
                  for p in payloads]).to_csv(OUT / "tabla_carga_util.csv", index=False)
    print(f"Figuras y tablas en {OUT}")


if __name__ == "__main__":
    main()
