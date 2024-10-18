import numpy as np
from PySDM.physics import si
from PySDM_examples.Singer_organics.kappa_mcmc import param_transform
from PySDM_examples.Singer_organics.constants_def import plot_names


def get_median_mcmc_values(model, aerosol, Ntotal=1000, Ncut=500):
    outputfile = (
        f"mcmc_output/{aerosol.shortname}_{plot_names[model]}_n{Ntotal}_chain.csv"
    )
    param_chain = np.loadtxt(outputfile, delimiter=",")
    data = param_chain[Ncut:, :]
    medians = np.median(data, axis=0)
    return medians


def get_converged_parameter_values(model, aerosol, Ntotal=1000, Ncut=500):
    outputfile = (
        f"mcmc_output/{aerosol.shortname}_{plot_names[model]}_n{Ntotal}_chain.csv"
    )
    param_chain = np.loadtxt(outputfile, delimiter=",").T
    data = param_transform(param_chain, model).T[Ncut:, :]
    return data


def get_median_parameter_values(model, aerosol, Ntotal=1000, Ncut=500):
    data = get_converged_parameter_values(model, aerosol, Ntotal, Ncut)
    medians = np.median(data, axis=0)
    return medians


def get_dict_median_parameters(model, aerosol, Ntotal=1000, Ncut=500):
    if model == "CompressedFilmOvadnevaite":
        medians = get_median_parameter_values(model, aerosol, Ntotal, Ncut)
        stc = {
            "sgm_org": medians[0] * si.J / si.m**2,
            "delta_min": medians[1] * si.m,
        }
    elif model == "SzyszkowskiLangmuir":
        medians = get_median_parameter_values(model, aerosol, Ntotal, Ncut)
        stc = {
            "RUEHL_nu_org": aerosol.modes[0]["nu_org"],
            "RUEHL_A0": medians[0] * si.m**2,
            "RUEHL_C0": medians[1],
            "RUEHL_sgm_min": medians[2] * si.J / si.m**2,
        }
    elif model == "CompressedFilmRuehl":
        medians = get_median_parameter_values(model, aerosol, Ntotal, Ncut)
        stc = {
            "RUEHL_nu_org": aerosol.modes[0]["nu_org"],
            "RUEHL_A0": medians[0] * si.m**2,
            "RUEHL_C0": medians[1],
            "RUEHL_sgm_min": medians[2] * si.mN / si.m,
            "RUEHL_m_sigma": medians[3] * si.J / si.m**4,
        }
    else:
        stc = {}
    return stc


def get_dict_randsample_parameters(model, aerosol, Ntotal=1000, Ncut=500):
    if model == "CompressedFilmOvadnevaite":
        data = get_converged_parameter_values(model, aerosol, Ntotal, Ncut)
        randn = np.random.randint(0, len(data))
        stc = {
            "sgm_org": data[randn, 0] * si.J / si.m**2,
            "delta_min": data[randn, 1] * si.m,
        }
    elif model == "SzyszkowskiLangmuir":
        data = get_converged_parameter_values(model, aerosol, Ntotal, Ncut)
        randn = np.random.randint(0, len(data))
        stc = {
            "RUEHL_nu_org": aerosol.modes[0]["nu_org"],
            "RUEHL_A0": data[randn, 0] * si.m**2,
            "RUEHL_C0": data[randn, 1],
            "RUEHL_sgm_min": data[randn, 2] * si.J / si.m**2,
        }
    elif model == "CompressedFilmRuehl":
        data = get_converged_parameter_values(model, aerosol, Ntotal, Ncut)
        randn = np.random.randint(0, len(data))
        stc = {
            "RUEHL_nu_org": aerosol.modes[0]["nu_org"],
            "RUEHL_A0": data[randn, 0] * si.m**2,
            "RUEHL_C0": data[randn, 1],
            "RUEHL_sgm_min": data[randn, 2] * si.mN / si.m,
            "RUEHL_m_sigma": data[randn, 3] * si.J / si.m**4,
        }
    else:
        stc = {}
    return stc
