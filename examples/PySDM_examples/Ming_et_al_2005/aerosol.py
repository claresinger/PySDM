from chempy import Substance
from pystrict import strict

from PySDM.initialisation import spectra
from PySDM.initialisation.aerosol_composition import DryAerosolMixture
from PySDM.physics import si


CONSTANTS_MING = {
    "Mv": 18.015 * si.g / si.mol,  # TODO
    "Md": 28.97 * si.g / si.mol,  # TODO
    "MAC": 1.0,
    "HAC": 0.97,
}


@strict
class AerosolMingSM(DryAerosolMixture):
    def __init__(
        self,
        water_molar_volume: float,
        N: float = 200 / si.cm**3,
        D: float = 0.02 * si.um,
        sig: float = 2.5,
    ):
        mode1 = {"(NH4)2SO4": 1.0}

        super().__init__(
            compounds=("(NH4)2SO4",),
            molar_masses={
                "(NH4)2SO4": Substance.from_formula("(NH4)2SO4").mass * si.g / si.mole
            },
            densities={
                "(NH4)2SO4": 1.77 * si.g / si.cm**3,
            },
            is_soluble={
                "(NH4)2SO4": True,
            },
            ionic_dissociation_phi={
                "(NH4)2SO4": 3,
            },
        )
        self.modes = (
            {
                "kappa": self.kappa(
                    mass_fractions=mode1,
                    water_molar_volume=water_molar_volume,
                ),
                "spectrum": spectra.Lognormal(
                    norm_factor=N, m_mode=D / 2.0, s_geom=sig
                ),
            },
        )


@strict
class AerosolMingTM(DryAerosolMixture):
    def __init__(
        self,
        water_molar_volume: float,
        N1: float = 340 / si.cm**3,
        D1: float = 0.01 * si.um,
        sig1: float = 1.6,
        N2: float = 60 / si.cm**3,
        D2: float = 0.07 * si.um,
        sig2: float = 2.0,
        N3: float = 3.1 / si.cm**3,
        D3: float = 0.62 * si.um,
        sig3: float = 2.7,
    ):
        mode1 = {"(NH4)2SO4": 1.0}
        mode2 = {"(NH4)2SO4": 1.0}
        mode3 = {"(NH4)2SO4": 1.0}

        super().__init__(
            compounds=("(NH4)2SO4",),
            molar_masses={
                "(NH4)2SO4": Substance.from_formula("(NH4)2SO4").mass * si.g / si.mole
            },
            densities={"(NH4)2SO4": 1.77 * si.g / si.cm**3},
            is_soluble={"(NH4)2SO4": True},
            ionic_dissociation_phi={"(NH4)2SO4": 3},
        )
        self.modes = (
            {
                "kappa": self.kappa(
                    mass_fractions=mode1, water_molar_volume=water_molar_volume
                ),
                "spectrum": spectra.Lognormal(
                    norm_factor=N1, m_mode=D1 / 2.0, s_geom=sig1
                ),
            },
            {
                "kappa": self.kappa(
                    mass_fractions=mode2, water_molar_volume=water_molar_volume
                ),
                "spectrum": spectra.Lognormal(
                    norm_factor=N2, m_mode=D2 / 2.0, s_geom=sig2
                ),
            },
            {
                "kappa": self.kappa(
                    mass_fractions=mode3, water_molar_volume=water_molar_volume
                ),
                "spectrum": spectra.Lognormal(
                    norm_factor=N3, m_mode=D3 / 2.0, s_geom=sig3
                ),
            },
        )


@strict
class AerosolMingTM_organic1C(DryAerosolMixture):
    def __init__(
        self,
        water_molar_volume: float,
        N1: float = 1000 / si.cm**3,
        D1: float = 0.016 * si.um,
        sig1: float = 1.6,
        N2: float = 800 / si.cm**3,
        D2: float = 0.068 * si.um,
        sig2: float = 2.1,
        N3: float = 0.72 / si.cm**3,
        D3: float = 0.92 * si.um,
        sig3: float = 2.2,
    ):
        mode1 = {"(NH4)2SO4": 0.8, "soluble_organic": 0.2}
        mode2 = {"(NH4)2SO4": 0.8, "soluble_organic": 0.2}
        mode3 = {"(NH4)2SO4": 0.8, "soluble_organic": 0.2}

        super().__init__(
            compounds=("(NH4)2SO4", "soluble_organic"),
            molar_masses={
                "(NH4)2SO4": Substance.from_formula("(NH4)2SO4").mass * si.g / si.mole,
                "soluble_organic": 50 * si.g / si.mole,  # TODO
            },
            densities={
                "(NH4)2SO4": 1.77 * si.g / si.cm**3,
                "soluble_organic": 1.77 * si.g / si.cm**3,  # TODO
            },
            is_soluble={"(NH4)2SO4": True, "soluble_organic": True},
            ionic_dissociation_phi={"(NH4)2SO4": 3, "soluble_organic": 0},
        )
        self.modes = (
            {
                "kappa": self.kappa(
                    mass_fractions=mode1, water_molar_volume=water_molar_volume
                ),
                "spectrum": spectra.Lognormal(
                    norm_factor=N1, m_mode=D1 / 2.0, s_geom=sig1
                ),
            },
            {
                "kappa": self.kappa(
                    mass_fractions=mode2, water_molar_volume=water_molar_volume
                ),
                "spectrum": spectra.Lognormal(
                    norm_factor=N2, m_mode=D2 / 2.0, s_geom=sig2
                ),
            },
            {
                "kappa": self.kappa(
                    mass_fractions=mode3, water_molar_volume=water_molar_volume
                ),
                "spectrum": spectra.Lognormal(
                    norm_factor=N3, m_mode=D3 / 2.0, s_geom=sig3
                ),
            },
        )
