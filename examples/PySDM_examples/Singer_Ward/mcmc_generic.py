import os

# os.environ["NUMBA_DISABLE_JIT"] = "1"
import warnings

import numpy as np
from numba.core.errors import NumbaExperimentalFeatureWarning

from matplotlib import pyplot
from corner import corner
from open_atmos_jupyter_utils import show_plot
from PySDM.physics import si

from PySDM_examples.Singer_Ward.aerosol import (
    AerosolAlphaPineneDark,
    AerosolAlphaPineneLight,
    AerosolBetaCaryophylleneDark,
    AerosolBetaCaryophylleneLight,
)
from PySDM_examples.Singer_Ward.kappa_mcmc import MCMC, param_transform, model_options


def mcmc_generic(
    filename="bcary_dark.csv", model="CompressedFilmOvadnevaite", n_steps=200, plot=True
):
    ######
    # open data file
    ######
    ds = np.loadtxt("data/" + filename, skiprows=1, delimiter=",")
    if filename == "bcary_dark.csv":
        ds = np.delete(ds, [26, 65], axis=0)  # remove outliers
    r_dry = ds[:, 0] / 2 * 1e-9
    ovf = np.minimum(ds[:, 1], 0.99)
    # d_ovf = ds[:, 2]
    kappa_eff = ds[:, 3]
    d_kappa_eff = ds[:, 4]
    T = 300 * si.K

    datay = kappa_eff
    errory = d_kappa_eff

    ######
    # set up MCMC
    ######
    if model == "CompressedFilmOvadnevaite":
        params = [-1.0, -0.5]  # [0.5, 0.2]
        stepsize = [0.1, 0.05]
    elif model == "SzyszkowskiLangmuir":
        params = [75, -14, 3.2]  # [25, -11.4, 3.7]
        stepsize = [0.5, 0.1, 0.05]
    elif model == "CompressedFilmRuehl":
        params = [75, -14, 3.2, 1.0]  # [15, -11.5, 3.5, 1.0]
        stepsize = [0.5, 0.1, 0.05, 0.05]
    else:
        print("error model name not recognized")

    if filename == "bcary_dark.csv":
        aerosol_list = [AerosolBetaCaryophylleneDark(ovfi) for ovfi in ovf]
    elif filename == "bcary_light.csv":
        aerosol_list = [AerosolBetaCaryophylleneLight(ovfi) for ovfi in ovf]
    elif filename == "apinene_dark.csv":
        aerosol_list = [AerosolAlphaPineneDark(ovfi) for ovfi in ovf]
    elif filename == "apinene_light.csv":
        aerosol_list = [AerosolAlphaPineneLight(ovfi) for ovfi in ovf]
    else:
        print("error aerosol type doesn't exist")
    args = [T, r_dry, ovf, aerosol_list, model]

    print(param_transform(params, model))
    print(params)

    ######
    # run MCMC
    ######
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NumbaExperimentalFeatureWarning)
        param_chain, accept_chain, chi2_chain = MCMC(
            params, stepsize, args, datay, errory, n_steps
        )
    p = param_transform(param_chain, model)

    print(p[:, -1])
    print(param_chain[:, -1])

    ######
    # plot and save results
    ######
    if not os.path.isdir("mcmc_output/"):
        os.mkdir("mcmc_output/")

    modelname = model.split("CompressedFilm")[-1]
    aerosolname = aerosol_list[0].__class__.__name__.split("Aerosol")[-1]

    # save parameter chain to text file
    filename = (
        "mcmc_output/"
        + aerosolname
        + "_"
        + modelname
        + "_chain"
        + str(np.max(np.shape(param_chain)))
        + ".csv"
    )
    np.savetxt(filename, param_chain.T, fmt="%.6e", delimiter=",")

    # plot parameter chain
    if model == "CompressedFilmOvadnevaite":
        _, axes = pyplot.subplots(2, 1, figsize=(6, 8))
    elif model == "CompressedFilmRuehl":
        _, axes = pyplot.subplots(2, 2, figsize=(12, 8))
    elif model == "SzyszkowskiLangmuir":
        _, axes = pyplot.subplots(3, 1, figsize=(6, 12))
    else:
        raise AssertionError()

    labels, scaling, _ = model_options(model)

    for i, ax in enumerate(axes.flatten()):
        p[i, 0:100] = np.nan
        ax.plot(p[i, :] * scaling[i])
        ax.set_ylabel(labels[i])
        ax.grid()
    show_plot("mcmc_output/" + aerosolname + "_" + modelname + "_chain.png", dpi=200)

    # plot corner plot of parameter distributions
    data = p.T[100:, :]
    labels, scaling = model_options(model)

    pyplot.rcParams.update({"font.size": 12})
    _ = corner(
        data * scaling,
        labels=labels,
        label_kwargs={"fontsize": 15},
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".1f",
        title_kwargs={"fontsize": 12},
    )
    show_plot("mcmc_output/" + aerosolname + "_" + modelname + "_corner.png", dpi=200)
