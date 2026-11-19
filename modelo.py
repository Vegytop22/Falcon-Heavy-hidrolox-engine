"""Physical model: Falcon Heavy vertical ascent, three powered segments and a coast.

Equations of motion for segment k (constant mass flow m_k and exhaust velocity ve_k):

    dx/dt = v
    dv/dt = m_k * ve_k / M - g
    dM/dt = -m_k
    dW/dt = v * m_k * ve_k / M        (work of thrust per unit mass, eq. 24)

The state vector is y = (x, v, M, W). Gravity g is constant (g0 or 0).
"""
import math
from dataclasses import dataclass, replace

G0 = 9.81
R_EARTH = 6.371e6
R_GAS = 8.314
V_ESC_SURFACE = math.sqrt(2.0 * G0 * R_EARTH)

PE_PC_SEA = 0.0104
PE_PC_VAC = 1.2428e-4


@dataclass(frozen=True)
class Propellant:
    key: str
    name: str
    tc: float
    mmol: float
    gamma: float

    def exhaust_velocity(self, pe_pc):
        """Nozzle equation (3): ve = sqrt(2 g R Tc / ((g - 1) Mmol) * (1 - (pe/pc)^((g-1)/g)))."""
        g = self.gamma
        expo = (g - 1.0) / g
        return math.sqrt(2.0 * g * R_GAS * self.tc / ((g - 1.0) * self.mmol)
                         * (1.0 - pe_pc ** expo))

    def isp(self, pe_pc):
        return self.exhaust_velocity(pe_pc) / G0


RP1 = Propellant("rp1", "LOX/RP-1", 3670.0, 0.0233, 1.22)
LH2 = Propellant("lh2", "LOX/LH2", 3500.0, 0.0100, 1.26)


@dataclass(frozen=True)
class Vehicle:
    core_prop: float = 411_000.0
    side_dry: float = 22_200.0
    core_dry: float = 25_600.0
    upper_prop: float = 107_500.0
    upper_dry: float = 4_000.0
    fairing: float = 1_750.0
    payload: float = 1_300.0
    engines_per_core: int = 9
    merlin_thrust: float = 845_000.0
    mvac_thrust: float = 934_000.0
    throttle: float = 0.60
    pe_pc_sea: float = PE_PC_SEA
    pe_pc_vac: float = PE_PC_VAC

    @property
    def liftoff_mass(self):
        return (3.0 * self.core_prop + 2.0 * self.side_dry + self.core_dry
                + self.upper_prop + self.upper_dry + self.fairing + self.payload)

    @property
    def final_mass(self):
        return self.upper_dry + self.fairing + self.payload

    def with_(self, **kw):
        return replace(self, **kw)


@dataclass(frozen=True)
class Segment:
    name: str
    duration: float
    mdot: float
    ve: float
    drop: float

    @property
    def thrust(self):
        return self.mdot * self.ve


def build_segments(propellant, vehicle=Vehicle(), mdot_factor=1.0, flow_reference=RP1):
    """Three powered segments followed by the coast.

    Mass flows are fixed by the RP-1 engines (hypothesis D7: same mass flow for both
    propellants), so only the exhaust velocity changes with the propellant.
    """
    ve_sea_ref = flow_reference.exhaust_velocity(vehicle.pe_pc_sea)
    ve_vac_ref = flow_reference.exhaust_velocity(vehicle.pe_pc_vac)
    m_merlin = mdot_factor * vehicle.merlin_thrust / ve_sea_ref
    m_core = vehicle.engines_per_core * m_merlin
    m_upper = mdot_factor * vehicle.mvac_thrust / ve_vac_ref

    ve_sea = propellant.exhaust_velocity(vehicle.pe_pc_sea)
    ve_vac = propellant.exhaust_velocity(vehicle.pe_pc_vac)

    t1 = vehicle.core_prop / m_core
    center_left = vehicle.core_prop - vehicle.throttle * m_core * t1
    t2 = center_left / m_core
    t3 = vehicle.upper_prop / m_upper
    return [
        Segment("Laterales + núcleo central", t1, (2.0 + vehicle.throttle) * m_core, ve_sea,
                2.0 * vehicle.side_dry),
        Segment("Núcleo central", t2, m_core, ve_sea, vehicle.core_dry),
        Segment("Segunda etapa (MVac)", t3, m_upper, ve_vac, 0.0),
    ]


def rhs(t, y, mdot, ve, g):
    x, v, m, _ = y
    a_thrust = mdot * ve / m
    return (v, a_thrust - g, -mdot, v * a_thrust)


def specific_energy(x, v, g):
    """Mechanical energy per unit mass used by the model: E = v^2/2 + g x."""
    return 0.5 * v * v + g * x


def orbital_energy(x, v):
    """Specific orbital energy with the exact field: eps = v^2/2 - g0 R^2 / (R + x)."""
    return 0.5 * v * v - G0 * R_EARTH ** 2 / (R_EARTH + x)


def escape_velocity(x):
    return math.sqrt(2.0 * G0 * R_EARTH ** 2 / (R_EARTH + x))


def exact_segment(tau, x0, v0, m0, mdot, ve, g):
    """Analytic solution of one segment, article eqs. (10)-(18). Returns (x, v, M)."""
    if tau <= 0.0:
        return x0, v0, m0
    if mdot == 0.0:
        return x0 + v0 * tau - 0.5 * g * tau * tau, v0 - g * tau, m0
    m = m0 - mdot * tau
    ln = math.log(m0 / m)
    v = v0 + ve * ln - g * tau
    x = x0 + v0 * tau - 0.5 * g * tau * tau + ve * tau * (1.0 - m / (mdot * tau) * ln)
    return x, v, m


def exact_state(t, segments, g, x0=0.0, v0=0.0, m0=None, vehicle=Vehicle()):
    """Exact (x, v, M) at time t, chaining segments and separations."""
    m = vehicle.liftoff_mass if m0 is None else m0
    x, v, start = x0, v0, 0.0
    for seg in segments:
        end = start + seg.duration
        if t <= end:
            return exact_segment(t - start, x, v, m, seg.mdot, seg.ve, g)
        x, v, m = exact_segment(seg.duration, x, v, m, seg.mdot, seg.ve, g)
        m -= seg.drop
        start = end
    return exact_segment(t - start, x, v, m, 0.0, 0.0, g)


def tsiolkovsky_dv(segments, vehicle=Vehicle()):
    """Ideal velocity gain without gravity: sum of ve_k ln(M0k / Mfk)."""
    m = vehicle.liftoff_mass
    dv = 0.0
    for seg in segments:
        mf = m - seg.mdot * seg.duration
        dv += seg.ve * math.log(m / mf)
        m = mf - seg.drop
    return dv
