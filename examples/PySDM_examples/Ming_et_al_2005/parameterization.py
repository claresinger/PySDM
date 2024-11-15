import numpy as np
from scipy.optimize import bisect

# (thermodynamic) constants
p_tr = 611.65  # Pa
Mv = 18.015e-3  # kg/mol
Md = 28.965e-3  # kg/mol
L0 = 2.5008e6  # J/kg
T_tr = 273.16  # K
R = 8.314  # J/mol/K
c_pd = 1004.5  # J/kg/K
g = 9.8  # m/s2
rho_w = 1000  # kg/m3
rho_d = 1.225  # kg/m3
sigma_w = 0.076  # N/m
Del_v = 1.096e-7  # m
alpha_c = 1  # -
Del_T = 2.16e-7  # m
alpha_T = 0.97  # -
D0 = 34.95e-4  # m2/s
gam_D = 1.94  # -
k0 = 4.39e-3  # J/m/s/K
beta_k = 7.1e-5  # J/m/s/K2

# model parameters
k = 2.4

# inputs
rho_aerosol = 1770  # kg/m3, ammonium sulfate
M_aerosol = 132.14e-3  # kg/mol, ammonium sulfate
ions_aerosol = 3  # ammonium sulfate (NH4)2SO4


# coefficient on updraft velocity, 1/m
def alpha(T):
    return (g * Mv * L0) / (c_pd * R * T**2) - (g * Md) / (R * T)


# coefficient on condensation rate, m3/kg
def gamma(T, p):
    return (R * T) / (Mv * pv_star(T)) + (Mv * L0**2) / (c_pd * Md * p * T)


# saturation vapor pressure, Clausius-Clapyeron, Pa
def pv_star(T):
    return p_tr * np.exp(Mv * L0 * (1 / T_tr - 1 / T) / R)


# growth coefficient, m2/s
def G(a, T, p):
    num = Mv * Dv_star(a, T, p) * pv_star(T) * ka_star(a, T) * R * T**2
    denom = (
        rho_w * R**2 * T**3 * ka_star(a, T)
        + rho_w * L0**2 * Mv**2 * Dv_star(a, T, p) * pv_star(T)
        - rho_w * L0 * R * T * Mv * Dv_star(a, T, p) * pv_star(T)
    )
    return num / denom


# water vapor diffusivity with non-continuum modifications, m2/s
def Dv_star(a, T, p):
    return Dv(T, p) / (
        a / (a + Del_v)
        + (Dv(T, p) / (a * alpha_c)) * ((2 * np.pi * Mv) / (R * T)) ** (1 / 2)
    )


# water vapor diffusivity, m2/s
def Dv(T, p):
    return D0 * (p_tr / p) * (T / T_tr) ** gam_D


# thermal conductivity with non-continuum modifications, J/m/s/K
def ka_star(a, T):
    return ka(T) / (
        a / (a + Del_T)
        + (ka(T) / (a * rho_d * c_pd * alpha_T))
        * ((2 * np.pi * Md) / (R * T)) ** (1 / 2)
    )


# thermal conductivity, J/m/s/K
def ka(T):
    return k0 + beta_k * T


# radius at maximum supersaturation, m
def amax(smax, w, T, p, ns):
    return (
        acrit(T, ns) ** 2
        + G(acrit(T, ns), T, p)
        * (smax**k - seq(acrit(T, ns), T, ns) ** k)
        / (alpha(T) * w)
    ) ** (1 / 2)


# critical radius of activation, m
def acrit(T, ns):
    return ((9 * R * T * ns) / (8 * np.pi * sigma_w)) ** (1 / 2)


# equilibrium supersaturation, Köhler curve, s = S-1
def seq(a, T, ns):
    return (
        np.exp(
            (2 * Mv * sigma_w) / (R * T * rho_w * a)
            - (ns * Mv) / (4 * np.pi * rho_w * a**3 / 3)
        )
        - 1
    )


# condensation rate, 1/s
def LHS(w, T, p):
    # print("LHS = ", alpha(T) * w / (4 * np.pi * rho_w * gamma(T, p)))
    return alpha(T) * w / (4 * np.pi * rho_w * gamma(T, p))


# condensation rate, 1/s
def RHS(smax, w, T, p, adry, Ndry):
    sum = 0
    for j, Nj in enumerate(Ndry):
        ns = moles_solute(adry[j])
        amaxj = amax(smax, w, T, p, ns)
        if amaxj >= acrit(T, ns):
            sum += G(amaxj, T, p) * (smax - seq(amaxj, T, ns)) * amaxj * Nj
    # print("RHS = ", sum)
    return sum


# moles of solute
def moles_solute(a_dry):
    m_dry = 4 / 3 * np.pi * a_dry**3 * rho_aerosol
    return m_dry / M_aerosol * ions_aerosol


def solve_smax(w, T, p, adry, Ndry):
    f = lambda smax: RHS(smax, w, T, p, adry, Ndry) - LHS(w, T, p)
    smax_l = 0.0
    smax_u = 0.1  # 10% supersaturation = 110% relative humidity
    # print(f(smax_l), f(smax_u))
    smax = bisect(f, smax_l, smax_u, rtol=1e-3)
    return smax


def droplet_number_conc(adry, Ndry, smax, T):
    CDNC = 0
    for j, Nj in enumerate(Ndry):
        ns = moles_solute(adry[j])
        if seq(acrit(T, ns), T, ns) <= smax:
            CDNC += Nj
    return CDNC


# # input conditions
# w = 1.0 # m/s
# T = 283.0 # K
# p = 800e2 # Pa
# bins = 20
# adry_per_bin = np.linspace(10,200,bins)*1e-9 # m
# Ndry_per_bin = np.ones(bins) * 10e6 # 1/m3


# smax = solve_smax(w, T, p, adry_per_bin, Ndry_per_bin)
# CDNC = droplet_number_conc(adry_per_bin, Ndry_per_bin, smax)
# AF = activated_fraction(adry_per_bin, Ndry_per_bin, smax)
# print(smax)
# print(CDNC)
# print(np.sum(Ndry_per_bin))
# print(AF)
