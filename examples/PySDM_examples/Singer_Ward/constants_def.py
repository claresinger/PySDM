from PySDM.physics import constants_defaults, si

SINGER_CONSTS = {
    "MAC": 1,
    "HAC": 1,
    "c_pd": constants_defaults.c_pd * si.joule / si.kilogram / si.kelvin,
    "g_std": constants_defaults.g_std * si.metre / si.second**2,
    "Md": constants_defaults.Md * si.joule / si.kelvin / si.kg,
    "Mv": constants_defaults.Mv * si.joule / si.kelvin / si.kg,
    "rho_w": constants_defaults.rho_w * si.kg / si.metre**3,
    "sgm_org": 20 * si.mN / si.m,
    "delta_min": 0.4 * si.nm,
    "RUEHL_nu_org": -1,
    "RUEHL_A0": 7.5e-19 * si.m**2,
    "RUEHL_C0": 5e-7,
    "RUEHL_sgm_min": 20 * si.mN / si.m,
    "RUEHL_m_sigma": 1.5e17 * si.J / si.m**2,
}

plot_lines = {
    "Constant": "-",
    "CompressedFilmOvadnevaite": "--",
    "SzyszkowskiLangmuir": "-.",
    "CompressedFilmRuehl": ":",
}
plot_colors = {
    "Constant": "k",
    "CompressedFilmOvadnevaite": "C0",
    "SzyszkowskiLangmuir": "C2",
    "CompressedFilmRuehl": "C1",
}
plot_names = {
    "Constant": "CONST",
    "CompressedFilmOvadnevaite": "OV",
    "SzyszkowskiLangmuir": "SL",
    "CompressedFilmRuehl": "RU",
}
