import sys; sys.path.append("/Users/omelchert/Programs/Python/py-fmas/"); sys.path.append("../../src/")
import os
import numpy as np
from fmas.config import FTFREQ, FT, IFT, C0
from fmas.grid import Grid
from fmas.tools import plot_evolution
from fmas.solver import CQE
from GNLStools import GNLS


def main(del_G):

    # -- INITIALIZATION STAGE -------------------------------------------------
    # ... COMPUTATIONAL DOMAIN
    t_max = 7000.       # (fs)
    t_num = 2**14       # (-)
    z_max = 0.15*1e6    # (micron)
    z0 = 0.10*1e6       # (micron)
    z_num = 700         # (-)
    z_skip = 1          # (-)
    grid = Grid( t_max = t_max, t_num = t_num)
    t, w = grid.t, grid.w

    # ... INITIAL CONDITION
    P0 = 1e4            # (W)
    t0 = 28.4           # (fs)
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

    # ... PROPAGATION ALGORITHM
    solver = CQE(model.Lw, model.Nw, user_action=model.claw_Ph, del_G=del_G)
    solver.set_initial_condition(w, FT(u0_t))
    solver.propagate( z_range = z_max, n_steps = z_num, n_skip = z_skip)

    # -- STORE RESULTS --------------------------------------------------------
    res = {
        "P0": P0,
        "t0": t0,
        "w0": w0,
        "del_G": del_G,
        "dz_integration": solver.dz_,
        "t": grid.t,
        "z": solver.z,
        "w": solver.w,
        "utz": solver.utz,
        'dz_a': np.asarray(solver._dz_a),
        'del_rle' : np.asarray(solver._del_rle),
        "Cp": solver.ua_vals
        }

    os.makedirs('./data_no_noise', exist_ok=True)
    np.savez_compressed('./data_no_noise/res_CQE_SC_delG%g'%(del_G), **res)

    # -- SHOW RESULTS ---------------------------------------------------------
    plot_evolution( solver.z, grid.t, solver.utz,
        t_lim = (-500,2200), w_lim = (-1.5,2.5), DO_T_LOG = False)


if __name__=='__main__':
    main(1e-12)
    #main(1e-8)
