'''main_GNLS_RK4IP.py

Solves generalized nonlinear Schrödinger equation (GNLS) using a fourth-order
Runge Kutta in the interaction picture (RK4IP) method.

The propagation scenario considered below is discussed on several occations in
the scientific literature. For instance, in Ref. [1], it is used to demonstrate
numerical simulations in terms of the gneralized nonlinear Schrödinger equation
(GNLS) on supercontinuum generation for an instance of a highly nonlinear
photonic crystal fiber (PCF) with an anomalous group-velocity dispersion
regime. In Ref. [2] it is used to introduce a particular z-propagation
algorithm referred to as the "Runge-Kutta in the interacton picture" (RK4IP)
method.

This script represents supplementary material for Ref. [3], demonstrating how
the GNLS data structure and noise models discussed in [3] can be used with a
z-propagation algorithm. For that purpose, the RK4IP algorithm is implemented
below.

References:
    [1] J. M. Dudley, G. Genty, S. Coen,
    Supercontinuum generation in photonic crystal fiber,
    Rev. Mod. Phys. 78 (2006) 1135,
    http://dx.doi.org/10.1103/RevModPhys.78.1135

    [2] J. Hult, A Fourth-Order Runge–Kutta in the Inter- action Picture
    Method for Simulating Supercontin- uum Generation in Optical Fibers,
    IEEE J. Light- wave Tech. 25 (2007) 3770,
    https://doi.org/10.1109/JLT.2007.909373.

    [3] NN

Author:
  O Melchert, 2022-05-05
'''
import sys; sys.path.append("../../src/")
import numpy as np
import matplotlib.pyplot as plt
from GNLStools import GNLS, noise_model_01

# -- SET COMPUTATIONAL GRID
z, dz = np.linspace(0, 0.1e6, 10000, retstep=True)
t = np.linspace(-3500, 3500, 2**13, endpoint=False)
w = np.fft.fftfreq(t.size, d=t[1]-t[0])*2*np.pi
# -- INSTANTIATE GENERALIZED NONLINEAR SCHRÖDINGER EQUATION 
gnls = GNLS(
    w,               # (rad/fs)
    beta_n = [
        -1.1830e-2,  # (fs^2/micron) beta_2
        8.1038e-2,   # (fs^3/micron) beta_3
        -0.95205e-1, # (fs^4/micron) beta_4
        2.0737e-1,   # (fs^5/micron) beta_5
        -5.3943e-1,  # (fs^6/micron) beta_6
        1.3486,      # (fs^7/micron) beta_7
        -2.5495,     # (fs^8/micron) beta_8
        3.0524,      # (fs^9/micron) beta_9
        -1.7140,     # (fs^10/micron) beta_10
        ],
    gamma=0.11e-6,   # (1/W/micron)
    w0= 2.2559,      # (rad/fs)
    fR = 0.18,       # (-)
    tau1 = 12.2,     # (fs)
    tau2 = 32.0      # (fs)
    )
# -- SPECIFY INITIAL PULSE
ut = np.sqrt(1e4)/np.cosh(t/28.4)
dut = noise_model_01(t, 2.2559, 1)
uw = np.fft.ifft(ut + dut)

# -- RK4IP PULSE PROPAGATION
P = np.exp(gnls.Lw*dz/2)
for n in range(1,z.size):
    uw_I = P*uw
    k1 = P*gnls.Nw(uw)*dz
    k2 = gnls.Nw(uw_I + k1/2)*dz
    k3 = gnls.Nw(uw_I + k2/2)*dz
    k4 = gnls.Nw(P*uw_I + k3)*dz
    uw = P*(uw_I + k1/6 + k2/3 + k3/3) + k4/6

# -- PLOT RESULTS
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
I = np.abs(np.fft.fft(uw))**2
ax1.plot(t, I*1e-3)
ax1.set_xlim(-200,3200); ax1.set_xlabel(r"Time $t$ (fs)")
ax1.set_ylim(0,6); ax1.set_ylabel(r"Intensity $|u|^2$ (kW)")
Iw = np.abs(uw)**2
ax2.plot(2*np.pi*0.29979/(w+2.2559), 10*np.log10(Iw/np.max(Iw)))
ax2.set_xlim(0.45,1.4); ax2.set_xlabel(r"Wavelength $\lambda$ (micron)")
ax2.set_ylim(-60,0); ax2.set_ylabel(r"Spectrum $|u_\lambda|^2$ (dB)")
fig.tight_layout(); plt.show()
