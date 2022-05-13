'''main_GNLS_SSM.py

Solves generalized nonlinear Schrödinger equation (GNLS) using a very basic
split-step method (see Eqs. (15) of Ref. [D2013]). This approach is not very
accurate (see the discussion in Ref. [D2013]) and meant for illustration
purposes only, use with care!

References:
[D2013] R. Deitering, R. Glowinski, H. Oliver, and S. Poole,
    A Reliable Split-Step Fourier Method for the Propagation Equation of
    Ultra-Fast Pulses in Single-Mode Optical Fibers,
    J. Lightwave Tech. 31 (2013) 2008.

Author:
  O Melchert, 2022-05-05
'''
import sys; sys.path.append("../../src/")
import numpy as np
import matplotlib.pyplot as plt
from GNLStools import GNLS

# -- SET COMPUTATIONAL GRID
z, dz = np.linspace(0, 0.1e6, 20000, retstep=True)
t = np.linspace(-3500, 3500, 2**13, endpoint=False)
w = np.fft.fftfreq(t.size, d=t[1]-t[0])*2*np.pi
# -- INSTANTIATE GENERALIZED NONLINEAR SCHRÖDINGER EQUATION 
gnls = GNLS(
    w,               # (rad/fs)
    beta_n = [
        -1.1830e-2,  # (fs^2/micron)
        8.1038e-2,   # (fs^3/micron)
        -0.95205e-1, # (fs^4/micron)
        2.0737e-1,   # (fs^5/micron)
        -5.3943e-1,  # (fs^6/micron)
        1.3486,      # (fs^7/micron)
        -2.5495,     # (fs^8/micron)
        3.0524,      # (fs^9/micron)
        -1.7140,     # (fs^10/micron)
        ],
    gamma=0.11e-6,   # (1/W/micron)
    w0= 2.2559,      # (rad/fs)
    fR = 0.18,       # (-)
    tau1 = 12.2,     # (fs)
    tau2 = 32.0      # (fs)
    )
# -- SPECIFY INITIAL PULSE
ut = np.sqrt(1e4)/np.cosh(t/28.4); uw = np.fft.ifft(ut)

# -- SPLIT-STEP PULSE PROPAGATION
FT, IFT = np.fft.ifft, np.fft.fft
P = np.exp(0.25*gnls.Lw*dz)
for n in range(1,z.size):
    uw = P*uw
    uw += gnls.Nw(uw)*dz
    uw = P*uw

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
