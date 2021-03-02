from PySDM_examples.Kreidenweis_et_al_2003 import Settings, Simulation
from PySDM.dynamics.aqueous_chemistry.aqueous_chemistry import GASEOUS_COMPOUNDS
from PySDM.physics import si
from matplotlib import pyplot
import numpy as np
import pytest


@pytest.fixture(scope='session')
def example_output():
    settings = Settings(n_sd=2, dt=1*si.s)
    simulation = Simulation(settings)
    output = simulation.run()
    return output

Z_CB = 196 * si.m


class TestFig1:
    @staticmethod
    def test_a(example_output, plot=True):
        # Plot
        if plot:
            name = 'ql'
            #prod = simulation.core.products['ql']
            pyplot.plot(example_output[name], np.asarray(example_output['t']) - Z_CB * si.s)
            #pyplot.xlabel(f"{prod.name} [{prod.unit}]")  # TODO #157
            pyplot.ylabel(f"time above cloud base [s]")
            pyplot.grid()
            pyplot.show()

        # Assert
        assert (np.diff(example_output['ql']) >= 0).all()

    @staticmethod
    def test_b(example_output, plot=True):
        # Plot
        if plot:
            for key in GASEOUS_COMPOUNDS.keys():
                pyplot.plot(
                    np.asarray(example_output[f'aq_{key}_ppb']),
                    np.asarray(example_output['t']) - Z_CB * si.s, label='aq')
                pyplot.plot(
                    np.asarray(example_output[f'gas_{key}_ppb']),
                    np.asarray(example_output['t']) - Z_CB * si.s, label='gas')
                pyplot.plot(
                    np.asarray(example_output[f'aq_{key}_ppb']) + np.asarray(example_output[f'gas_{key}_ppb']),
                    np.asarray(example_output['t']) - Z_CB * si.s, label='sum')
                pyplot.legend()
#                pyplot.xlim(0, .21)
                pyplot.xlabel(key + ' [ppb]')
                pyplot.show()

        # Assert
        # assert False  TODO #157

    @staticmethod
    def test_c(example_output, plot=True):
        if plot:
            pyplot.plot(example_output['pH'], np.asarray(example_output['t']) - Z_CB * si.s)
            pyplot.xlabel('pH')
            pyplot.show()

        #  assert False  TODO #157