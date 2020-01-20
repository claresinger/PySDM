"""
Created at 09.01.2020

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
import scipy.integrate
from ._odesystem import _ODESystem, idx_lnv, idx_thd
from PySDM.simulation.physics.constants import rho_w


# class TODO:
#     @staticmethod
#     def step(**args):
#         np.save("C:\\Users\\piotr\\PycharmProjects\\PySDM\\PySDM_tests\\unit_tests\\simulation\\dynamics\\condensation\\test_data.npy", args)
#         return BDF.step(**args)


class Solver:
    def __init__(self, backend, mean_n_sd_in_cell):
        length = 2 * mean_n_sd_in_cell + 2
        self.y = backend.array(length, dtype=float)  # TODO: list

    def step(self,
             v, n, vdry,
             cell_idx,
             dt, kappa,
             thd, qv,
             dthd_dt, dqv_dt,
             m_d_mean, rhod_mean
             ):
        n_sd_in_cell = len(cell_idx)
        y0 = self.y[0:n_sd_in_cell + idx_lnv]
        y0[idx_thd] = thd
        y0[idx_lnv:] = np.log(v[cell_idx])  # TODO: abstract out ln()
        qt = qv + _ODESystem.ql(n[cell_idx], y0[idx_lnv:], m_d_mean)
        y1 = self.solve_ivp(
            _ODESystem(
                kappa,
                vdry[cell_idx],
                n[cell_idx],
                dthd_dt,
                dqv_dt,
                m_d_mean,
                rhod_mean,
                qt
            ),
            t_range=(0., dt),
            y0=y0,
            rtol=1e-3,
            atol=1e-3
        )

        m_new = 0
        m_old = 0
        for i in range(n_sd_in_cell):
            x_new = np.exp(y1[idx_lnv + i])
            x_old = v[cell_idx[i]]
            nd = n[cell_idx[i]]
            m_new += nd * x_new * rho_w
            m_old += nd * x_old * rho_w
            v[cell_idx[i]] = x_new

        return m_new, m_old, y1[idx_thd]


class EE(Solver):
    def solve_ivp(self, odesys, t_range,
                  y0,
                  rtol=1e-3,
                  atol=1e-3):
        y = y0
        dt = t_range[1] - t_range[0]
        y += dt * odesys(None, y0)  # TODO: backend.add()
        return y


class ImplicitInSizeExplicitInThermodynamic(Solver):
    def solve_ivp(self, odesys, t_range,
                  y0,
                  rtol=1e-3,
                  atol=1e-3):
        dt = t_range[1] - t_range[0]

        for i in range(idx_lnv, len(y0[idx_lnv:])):
            g = lambda x: y0[i] - x + dt * odesys.derr(x, y0[idx_thd], y0[idx_qv], odesys.rd[i])
            y_left = y0[i]
            g0 = g(y_left)
            y1 = y_left + 2 * g0
            g1 = g(y1)
            if y_left > y1:
                y_left, y1 = y1, y_left
            if g0 * g1 < 0:
                for j in range(50):
                    ys = (y_left + y1) / 2
                    gs = g(ys)
                    if g0 * gs < 0:
                        y1 = ys
                        g1 = gs
                    else:
                        y_left = ys
                        g0 = gs
            else:
                y_left += g0
            y0[i] = y_left

        return y0


class BDF(Solver):
    def __init__(self, backend, mean_n_sd_in_cell):
        super().__init__(backend, mean_n_sd_in_cell)

    def solve_ivp(self, odesys, t_range,
                  y0,
                  rtol=1e-3,
                  atol=1e-3):
        integ = scipy.integrate.solve_ivp(odesys,
                                         t_range,
                                         y0,
                                         method="BDF",
                                         rtol=rtol,
                                         atol=atol,
                                         t_eval=[t_range[-1]]
                                         )
        assert integ.success, integ.message
        return integ.y[:, 0]