import numba
import numpy as np

from PySDM import Formulae
from PySDM.backends.impl_numba.conf import JIT_FLAGS as jit_flags
from PySDM.backends.impl_numba.toms748 import toms748_solve
from PySDM.physics import si

from PySDM_examples.Singer_Ward.constants_def import SINGER_CONSTS
from PySDM_examples.Singer_Ward.kappa_mcmc import param_transform


@numba.njit(**{**jit_flags, "parallel": False})
def minfun(rcrit, T, r_dry, kappa, f_org, fun_volume, fun_sigma, fun_r_cr):
    v_dry = fun_volume(r_dry)
    vcrit = fun_volume(rcrit)
    sigma = fun_sigma(T, vcrit, v_dry, f_org)
    rc = fun_r_cr(kappa, r_dry**3, T, sigma)
    return rcrit - rc


@numba.njit(**{**jit_flags, "parallel": True})
def parallel_block(
    T,
    r_dry,
    N_meas,
    kappas,
    f_orgs,
    rtol,
    max_iters,
    fun_volume,
    fun_sigma,
    fun_r_cr,
    fun_within_tolerance,
):
    rcrit = np.zeros(N_meas)
    for i in numba.prange(len(r_dry)):
        rd = r_dry[i]
        bracket = (rd / 2, 10e-6)
        rc_args = (T, rd, kappas[i], f_orgs[i], fun_volume, fun_sigma, fun_r_cr)
        rcrit_i, iters = toms748_solve(
            minfun,
            rc_args,
            *bracket,
            minfun(bracket[0], *rc_args),
            minfun(bracket[1], *rc_args),
            rtol,
            max_iters,
            fun_within_tolerance
        )
        assert iters != max_iters
        rcrit[i] = rcrit_i
    return rcrit


# evaluate the y-values of the model, given the current guess of parameter values
def get_model_jit(params, args):
    T, r_dry, _, aerosol_list, model = args

    if model == "CompressedFilmOvadnevaite":
        formulae = Formulae(
            surface_tension=model,
            constants={
                "sgm_org": param_transform(params, model)[0] * si.mN / si.m,
                "delta_min": param_transform(params, model)[1] * si.nm,
                **SINGER_CONSTS,
            },
        )
    elif model == "SzyszkowskiLangmuir":
        formulae = Formulae(
            surface_tension=model,
            constants={
                "RUEHL_nu_org": aerosol_list[0].modes[0]["nu_org"],
                "RUEHL_A0": param_transform(params, model)[0] * si.m**2,
                "RUEHL_C0": param_transform(params, model)[1],
                "RUEHL_sgm_min": param_transform(params, model)[2] * si.mN / si.m,
                **SINGER_CONSTS,
            },
        )
    elif model == "CompressedFilmRuehl":
        formulae = Formulae(
            surface_tension=model,
            constants={
                "RUEHL_nu_org": aerosol_list[0].modes[0]["nu_org"],
                "RUEHL_A0": param_transform(params, model)[0] * si.m**2,
                "RUEHL_C0": param_transform(params, model)[1],
                "RUEHL_sgm_min": param_transform(params, model)[2] * si.mN / si.m,
                "RUEHL_m_sigma": param_transform(params, model)[3] * si.J / si.m**2,
                **SINGER_CONSTS,
            },
        )
    else:
        raise AssertionError()

    fun_within_tolerance = formulae.trivia.within_tolerance
    fun_volume = formulae.trivia.volume
    fun_sigma = formulae.surface_tension.sigma
    fun_r_cr = formulae.hygroscopicity.r_cr

    N_meas = len(r_dry)
    max_iters = 1e2
    rtol = 1e-2

    kappas = np.asarray(
        [aerosol_list[i].modes[0]["kappa"][model] for i in range(len(r_dry))]
    )
    f_orgs = np.asarray([aerosol_list[i].modes[0]["f_org"] for i in range(len(r_dry))])

    rcrit = parallel_block(
        T,
        r_dry,
        N_meas,
        kappas,
        f_orgs,
        rtol,
        max_iters,
        fun_volume,
        fun_sigma,
        fun_r_cr,
        fun_within_tolerance,
    )

    kap_eff = (
        (2 * rcrit**2)
        / (3 * r_dry**3 * formulae.constants.Rv * T * formulae.constants.rho_w)
        * formulae.constants.sgm_w
    )

    return kap_eff
