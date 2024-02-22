import time
import numpy as np
import numba

from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar

from PySDM import Formulae
from PySDM.physics import si
from PySDM.backends.impl_numba.conf import JIT_FLAGS as jit_flags
from PySDM.backends.impl_numba.toms748 import toms748_solve

from PySDM_examples.Singer_Ward.constants_def import SINGER_CONSTS


"""
    parameter transformation so the MCMC parameters range from [-inf, inf]
    but the compressed film parameters are bounded appropriately

    for Ovadnevaite:
        sgm_org = [0,72.8] and delta_min = [0,inf]
    for Ruehl:
        A0 = [0,inf], C0 = [0,inf], sgm_min = [0,inf], and m_sigma = [-inf,inf]
    for SzyszkowskiLangmuir
        A0 = [0,inf], C0 = [0,inf], and sgm_min = [0,inf]
"""


def param_transform(mcmc_params, model):
    film_params = np.copy(mcmc_params)

    if model == "CompressedFilmOvadnevaite":
        film_params[0] = (
            # TODO change parameter transormation to be order of magnitude.
            Formulae().constants.sgm_w
            / (1 + np.exp(-1 * mcmc_params[0]))
            / (si.mN / si.m)
        )
        film_params[1] = np.exp(mcmc_params[1])
    elif model == "CompressedFilmRuehl":
        film_params[0] = mcmc_params[0] * 1e-20
        film_params[1] = np.exp(mcmc_params[1])
        film_params[2] = np.exp(mcmc_params[2])
        film_params[3] = mcmc_params[3] * 1e17
    elif model == "SzyszkowskiLangmuir":
        film_params[0] = mcmc_params[0] * 1e-20
        film_params[1] = np.exp(mcmc_params[1])
        film_params[2] = np.exp(mcmc_params[2])
    else:
        raise AssertionError()

    return film_params


def negSS(r_wet, SS_args):
    formulae, T, r_dry, kappa, f_org = SS_args
    v_dry = formulae.trivia.volume(r_dry)
    v_wet = formulae.trivia.volume(r_wet)
    sigma = formulae.surface_tension.sigma(T, v_wet, v_dry, f_org)
    RH_eq = formulae.hygroscopicity.RH_eq(r_wet, T, kappa, r_dry**3, sigma)
    SS = (RH_eq - 1) * 100
    return -1 * SS


# def get_rcrit(SS_args, bracket):
#     return minimize_scalar(negSS, args=SS_args, bracket=bracket).x


"""
    evaluate the y-values of the model, given the current guess of parameter values
"""


def get_model(params, args):
    T, r_dry, _, aerosol_list, model = args
    kappa = [ai.modes[0]["kappa"][model] for ai in aerosol_list]
    forg = [ai.modes[0]["f_org"] for ai in aerosol_list]
    nu_org = aerosol_list[0].modes[0]["nu_org"]

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
                "RUEHL_nu_org": nu_org,
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
                "RUEHL_nu_org": nu_org,
                "RUEHL_A0": param_transform(params, model)[0] * si.m**2,
                "RUEHL_C0": param_transform(params, model)[1],
                "RUEHL_sgm_min": param_transform(params, model)[2] * si.mN / si.m,
                "RUEHL_m_sigma": param_transform(params, model)[3] * si.J / si.m**2,
                **SINGER_CONSTS,
            },
        )
    else:
        raise AssertionError()

    rcrit = np.zeros(len(r_dry))
    for i, rd in enumerate(r_dry):
        SS_args = [formulae, T, rd, kappa[i], forg[i]]
        res = minimize_scalar(negSS, args=SS_args, bracket=[rd / 2, 100e-6])
        rcrit[i] = res.x

    # rcrit = np.array(Parallel(verbose=0, n_jobs=-1, backend="threading")(
    #     delayed(get_rcrit)(
    #         [formulae, T, rd, kappa[i], forg[i]],
    #         [rd / 2, 100e-6]
    #     )
    #     for i,rd in enumerate(r_dry)
    # ))

    kap_eff = (
        (2 * rcrit**2)
        / (3 * r_dry**3 * formulae.constants.Rv * T * formulae.constants.rho_w)
        * formulae.constants.sgm_w
    )

    return kap_eff


"""
    helper function for rc minimization
"""


@numba.njit(**{**jit_flags, "parallel": False})
def minfun(rcrit, T, r_dry, kappa, f_org, fun_volume, fun_sigma, fun_r_cr):
    v_dry = fun_volume(r_dry)
    vcrit = fun_volume(rcrit)
    sigma = fun_sigma(T, vcrit, v_dry, f_org)
    rc = fun_r_cr(kappa, r_dry**3, T, sigma)
    return rcrit - rc


"""
    helper function for parallel code
"""


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


"""
    get model using numba jitted code
"""


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


"""
    obtain the chi2 value of the model y-values given current parameters
    vs. the measured y-values
    calculate chi2 not log likelihood
"""


def get_chi2(params, args, y, error):
    # TODO figure out parallel code and numba stuff with get_model() or get_model_jit()
    model = get_model(params, args)
    chi2 = np.sum(((y - model) / error) ** 2)
    return chi2


"""
    1. propose a new parameter set
    2. take a step in one paramter of random length in random direction
            with stepsize chosen from a normal distribution with width sigma
"""


def propose_param(current_param, stepsize):
    picker = int(np.floor(np.random.random(1) * len(current_param)))
    sigma = stepsize[picker]
    perturb_value = np.random.normal(0.0, sigma)

    try_param = np.zeros(len(current_param))
    try_param[~picker] = current_param[~picker]
    try_param[picker] = current_param[picker] + perturb_value

    try_param = np.copy(current_param)
    try_param[picker] = current_param[picker] + perturb_value

    return try_param, picker


"""
    evaluate whether to step to the new trial value
"""


def step_eval(params, stepsize, args, y, error):
    chi2_old = get_chi2(params, args, y, error)
    try_param, picker = propose_param(params, stepsize)
    chi2_try = get_chi2(try_param, args, y, error)

    # determine whether a step should be taken
    if chi2_try <= chi2_old:
        new_param = try_param
        accept_value = 1
    else:
        alpha = np.exp(chi2_old - chi2_try)
        r = np.random.random(1)
        if r < alpha:
            new_param = try_param
            accept_value = 1
        else:
            new_param = params
            accept_value = 0

    chi2_value = get_chi2(new_param, args, y, error)
    return new_param, picker, accept_value, chi2_value


"""
    run the whole MCMC routine, calling the subroutines written above
"""


def MCMC(params, stepsize, args, y, error, n_steps):
    param_chain = np.zeros((len(params), n_steps))
    accept_chain = np.empty((len(params), n_steps))
    accept_chain[:] = np.nan
    chi2_chain = np.zeros(n_steps)

    for i in np.arange(n_steps):
        t = time.time()
        param_chain[:, i], ind, accept_value, chi2_chain[i] = step_eval(
            params, stepsize, args, y, error
        )
        accept_chain[ind, i] = accept_value
        params = param_chain[:, i]
        # print("step time: ", time.time() - t)

    return param_chain, accept_chain, chi2_chain


"""
    plotting options
"""


def model_options(model):
    if model == "CompressedFilmOvadnevaite":
        labels = ["$\sigma_{org}$ [mN / m]", "$\delta_{min}$ [nm]"]
        scaling = [1, 1]
        plot_order = [0, 1]
    elif model == "SzyszkowskiLangmuir":
        labels = [
            "$A_0$ [m$^2$] $\\times 10^{20}$",
            "$C_0 \\times 10^{6}$",
            "$\sigma_{org}$ [mN / m]",
        ]
        scaling = [1e20, 1e6, 1]
        plot_order = [1, 2, 0]
    elif model == "CompressedFilmRuehl":
        labels = [
            "$A_0$ [m$^2$] $\\times 10^{20}$",
            "$C_0 \\times 10^{6}$",
            "$\sigma_{org}$ [mN / m]",
            "$m_{\sigma}$ [J / m$^2$] $\\times 10^{-16}$",
        ]
        scaling = [1e20, 1e6, 1, 1e-16]
        plot_order = [1, 2, 0, 3]
    else:
        raise AssertionError()
    return labels, scaling, plot_order
