from PySDM.simulation.physics import constants
from PySDM.simulation.physics.dimensional_analysis import DimensionalAnalysis
from PySDM.simulation.physics import formulae


class TestFormulae:
    def test_pvs(self):
        with DimensionalAnalysis():
            # Arrange
            si = constants.si
            sut = formulae.pvs
            T = 300 * si.kelvins

            # Act
            pvs = sut(T)

            # Assert
            assert pvs.units == si.hectopascals

    def test_r_cr(self):
        with DimensionalAnalysis():
            # Arrange
            si = constants.si
            sut = formulae.r_cr

            kp = .5
            rd = .1 * si.micrometre
            T = 300 * si.kelvins

            # Act
            r_cr = sut(kp, rd, T)

            # Assert
            assert r_cr.to_base_units().units == si.metres
