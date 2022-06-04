import sys; sys.path.append("/Users/omelchert/Programs/Python/py-fmas/"); sys.path.append("../../src/")
import os
import numpy as np
from fmas.config import FTFREQ, FT, IFT, C0
from fmas.grid import Grid
from fmas.tools import plot_claw, plot_evolution
from fmas.solver import CQE
from GNLStools import GNLS, noise_spectral_synthesis_quantum, noise_direct_sampling


def main(del_G, t0_FWHM=50., s0=0):

    # -- INITIALIZATION STAGE -------------------------------------------------
    # ... COMPUTATIONAL DOMAIN
    t_max = 3500.       # (fs)
    t_num = 2**14       # (-)
    z_max = 0.12*1e6    # (micron)
    z0 = 0.10*1e6       # (micron)
    z_num = 200         # (-)
    z_skip = 1          # (-)
    grid = Grid( t_max = t_max, t_num = t_num)
    t, w = grid.t, grid.w

    # ... INITIAL CONDITION
    P0 = 1e4            # (W)
    t0 = 28.4           # (fs)
    t0 = t0_FWHM/1.763  # (fs)
    w0 = 2.2559         # (rad/fs)
    u0_t = np.sqrt(P0)/np.cosh(t/t0)

    # ... GENERALIZED NONLINEAR SCHROEDINGER EQUATION 
    b2 = -1.1830e-2  # (fs^2/micron)
    b3 = 8.1038e-2   # (fs^3/micron)
    b4 = -0.95205e-1 # (fs^4/micron)
    b5 = 2.0737e-1   # (fs^5/micron)
    b6 = -5.3943e-1  # (fs^6/micron)
    b7 = 1.3486      # (fs^7/micron)
    b8 = -2.5495     # (fs^8/micron)
    b9 = 3.0524      # (fs^9/micron)
    b10 = -1.7140    # (fs^10/micron)
    # ... NONLINEAR PARAMETER
    gamma = 0.11e-6     # (1/W/micron)
    # ... RAMAN RESPONSE
    fR = 0.18           # (-)
    tau1 = 12.2         # (fs)
    tau2 = 32.0         # (fs)
    model = GNLS(w, beta_n = [b2,b3,b4,b5,b6,b7,b8,b9,b10], gamma=gamma, w0=w0, fR=fR, tau1=tau1, tau2=tau2)

    du0_t = noise_spectral_synthesis_quantum(t,w0,s0)

    # ... PROPAGATION ALGORITHM
    solver = CQE(model.Lw, model.Nw, user_action=model.claw_Ph, del_G=del_G)
    solver.set_initial_condition(w, FT(u0_t + du0_t))
    solver.propagate( z_range = z_max, n_steps = z_num, n_skip = z_skip)

    # -- STORE RESULTS --------------------------------------------------------
    z_id = np.argmin(np.abs(solver.z-z0))
    results = {
        "s0": s0,
        "du": du0_t,
        "u0": u0_t,
        "P0": P0,
        "t0": t0,
        "w0": w0,
        "t": grid.t,
        "z": solver.z[z_id],
        "w": solver.w,
        "utz": solver.utz[z_id],
        }

    return results



def wrapper(t0_FWHM=50):
    delG = 1e-8
    os.makedirs('./data_delG%g_t0FWHM%lf'%(delG,t0_FWHM), exist_ok=True)

    for s0 in range(30):
        res = main(delG, t0_FWHM, s0)
        np.savez_compressed('./data_delG%g_t0FWHM%lf/res_CQE_SC_delG%g_t0FWHM%lf_s0%d'%(delG, t0_FWHM, delG, t0_FWHM, s0), **res)


if __name__=='__main__':
    wrapper(50.)
    wrapper(100.)
    wrapper(150.)
