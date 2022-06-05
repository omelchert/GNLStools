'''GNLStools.py

Module implementing data structures and functions enabling numerical simulation
and analysis of the propagation dynamics of ultrashort laser pulses in
nonlinear waveguides.

The modeling approach is based on the generalized nonlinear Schroedinger
equation for the pulse envelope. The provided software implements the effects
of linear dispersion, pulse self-steepening, and the Raman effect. Input pulse
shot noise can be included using commonly adopted quantum noise models
considering both, pure spectral phase noise as well as Gaussian noise, and
coherence properties of the resulting spectra can be calculated.

Author: O. Melchert
Date: March 2022
'''
import numpy as np
import numpy.fft as nfft
from numpy.lib.scimath import sqrt as csqrt
from scipy.special import factorial
from scipy.constants import Planck  as hPlanck

# -- CONVENIENT ABBREVIATIONS
hBar = 1e15*hPlanck/2/np.pi # (J fs) reduced Planck constant
# ... FORWARD AND BACKWARD FOURIER TRANSFORMS
FT = nfft.ifft
IFT = nfft.fft


def noise_model_01(t,w0,s0):
    r"""noise model 01 - Direct sampling

    generates an instance of noise by directly sampling in the time- domain.
    The underlying noise model assumes complex-valued noise amplitudes with
    normally distributed real and imaginary parts.

    Args:
        t (1D numpy-array, floats): time grid
        w0 (float): pulse center frequency
        s0 (int): seed for random number generator

    Returns: (du)
        du (1D numpy-array, cplx floats): instance of time-domain noise
    """
    np.random.seed(s0)
    N01 = np.random.normal
    dt = t[1]-t[0]

    # -- REPRESENTATIVE ENERGY OF PHOTON IN BIN 
    e0 = hBar*w0                        # (J)
    # -- NOISE SCALING FACTOR ACCOUNTING FOR POWER IN WATTES
    sFac = csqrt(1e15*e0/dt/4)      # (sqrt(W) = sqrt(J/s))
    # -- GAUSSIAN NOISE MODEL IN TIME DOMAIN 
    noise_t = sFac*(N01(0,1,size=t.size) + 1j*N01(0,1,size=t.size))

    return noise_t


def noise_model_02(t,w0,s0):
    r"""noise model 02 - Fourier method, pure phase noise

    generates an instance of time-domain noise by sampling its Fourier
    representation. The underlying noise model assumes complex valued spectral
    amplitudes with pure phase noise.

    Args:
        t (1D numpy-array, floats): time grid
        w0 (float): pulse center frequency
        s0 (int): seed for random number generator

    Returns: (du)
        du (1D numpy-array, cplx floats): instance of time-domain noise
    """
    np.random.seed(s0)
    U = np.random.uniform
    w = nfft.fftfreq(t.size,d=t[1]-t[0])*2*np.pi
    T = 2*np.pi/(w[1]-w[0])

    # -- REPRESENTATIVE ENERGY OF PHOTON IN BIN 
    e0 = hBar*(w+w0)          # (J)
    # -- NOISE SCALING FACTOR
    sFac = csqrt(1e15*e0/T)       # (sqrt(W) = sqrt(Js))
    # -- OBTAIN SPECTRAL AMPLITUDES WITH PURE PHASE NOISE
    phi = U(size=t.size)
    noise_w = sFac*np.exp(1j*2*np.pi*phi)

    return IFT(noise_w)             # ( sqrt(Js)*rad/s = sqrt(J/s)*rad)


def noise_model_03(t,w0,s0):
    r"""noise model 03 - Fourier method, Gaussian noise

    generates an instance of time-domain noise by sampling its Fourier
    representation. The underlying noise model assumes complex-valued spectral
    amplitudes with normally distributed real and imaginary parts.

    Args:
        t (1D numpy-array, floats): time grid
        w0 (float): pulse center frequency
        s0 (int): seed for random number generator

    Returns: (du)
        du (1D numpy-array, cplx floats): time-domain representation
    """
    np.random.seed(s0)
    U = np.random.uniform
    w = nfft.fftfreq(t.size,d=t[1]-t[0])*2*np.pi
    T = 2*np.pi/(w[1]-w[0])

    # -- REPRESENTATIVE ENERGY PER BIN 
    e0 = hBar*(w+w0)          # (J)
    # -- NOISE SCALING FACTOR
    sFac = csqrt(1e15*e0/T/4)     # (sqrt(W) = sqrt(Js))
    # -- OBTAIN NOISIFIED SPECTRAL AMPLITUDES USING BOX-MUELLER METHOD 
    phi1, phi2 = U(size=t.size), U(size=t.size)
    noise_w = sFac*np.sqrt(-2*np.log(phi1))*np.exp(2j*np.pi*phi2)

    return IFT(noise_w)             # ( sqrt(Js)*rad/s = sqrt(J/s)*rad)


def number_of_photons(w,uw,w0):
    r"""number of photons in pulse

    computes the total number photons in the pulse according to Eq.~(9d).

    Args:
        w (numpy array, 1D): angular frequency grid
        uw (numpy array, 1D): spectral envelope
        w0 (float): pulse center frequency

    Notes:
        - Squared amplitude is measured in units of Watts (W=J/s), hBar is
          given in units J fs, and w and w0  are given in rad/fs. So as to
          account for the difference in units, an additional factor of 1e-15 is
          needed to get a proper number of photons.

    Returns: (N0)
        N0 (float): total number of photons
    """
    # -- SCALING FACTOR WITH UNITS OF 1/J
    sFac = 2*np.pi/hBar/(w[1]-w[0])
    # --- DIMENSIONLES TOTAL NUMBER OF PHOTONS
    return 1e-15*sFac*np.sum(np.abs(uw)**2/(w + w0))


def number_of_photons_naive(t,u0,w0):
    r"""number of photons in pulse (naive)

    computes the total number photons in the pulse, based on the assumption
    that all photons contribute the same energy (hbar omega_0)

    Args:
        t (numpy array, 1D): time samples
        u0 (numpy array, 1D): complex field
        w0 (float): pulse center frequency

    Notes:
        - Squared amplitude is measured in units of Watts (W=J/s), hBar is
          given in units J fs, and w and w0  are given in rad/fs. So as to
          account for the difference in units, an additional factor of 1e-15 is
          needed to get a proper number of photons.

    Returns: (N0)
        N0 (float): total number of photons
    """
    # -- TOTAL ENERGY IN PULSE
    E = 1e-15*np.trapz(np.abs(u0)**2,x=t) # (W*fs*1e15 = J)
    # -- PHOTON ENERGY
    E0 = hBar*w0                    # (J)
    return E/E0


def coherence_interpulse(w, uw_list):
    r"""first order coherence (interpulse coherence)

    Implements spectrally resolved modulus of first order coherence for zero
    time-lag, see Eq. (3) of Ref. [1]

    Args:
      w (1D array): angular frequency grid
      uw_list (2D array): list of spectral envelopes at same z

    Returns: Gamma
      g12 (float): first order coherence as function of angular frequency

    Refs:
        [1] J. M. Dudley, G. Genty, S. Coen,
        Supercontinuum generation in photonic crystal fiber,
        Rev. Mod. Phys. 78 (2006) 1135,
        http://dx.doi.org/10.1103/RevModPhys.78.1135.
    """
    Iw_av = np.mean(np.abs(uw_list)**2, axis=0)
    nPairs = 0
    tmp = np.zeros(len(uw_list[0]),dtype=complex)
    for i,j in list(itertools.combinations(range(len(uw_list)),2)):
       tmp += uw_list[i]*np.conj(uw_list[j])
       nPairs += 1
    tmp = np.abs(tmp)/nPairs
    return np.real(tmp/Iw_av), Iw_av


def coherence_intrapulse(w, uw_list, w1, w2):
    r"""intrapulse coherence function

    Implements variant of intrapulse coherence function, modified
    from Eq. (3) of Ref. [1]

    Args:
      w (1D array): angular frequency grid
      uw_list (2D array): list of spectral envelopes at same z
      w1 (float): selected frequency for complex conjugated field
      w2 (float): selected frequency for field

    Returns: Gamma
      Gamma (float): intrapulse coherence for the specified frequencies

    NOTE:
      - in Ref. [1], intrapulse coherence with focus on f-to-2f setups was
        considered. Here the intrapulse coherence of Ref. [1] is modified
        by relaxing the f/2f condition. Instead, two general distict
        frequencies are considered.

    Refs:
      [1] Role of Intrapulse Coherence in Carrier-Envelope Phase Stabilization
          N. Raabe, T. Feng, T. Witting, A. Demircan, C. Bree, and G. Steinmeyer,
          Phys. Rev. Lett. 119, 123901 (2017),
          https://doi.org/10.1103/PhysRevLett.119.123901.
    """
    def _helper(w,Ewz,w1,w2):
        w1Id = np.argmin(np.abs(w-w1))
        w2Id = np.argmin(np.abs(w-w2))
        return Ewz[w2Id]*np.conj(Ewz[w1Id])
    tmp = list(map(lambda x: _helper(w, x, w1, w2), uw_list))
    num = np.abs(np.mean(tmp))
    den = np.mean(np.abs(tmp))
    return num/den


def beta_w_coeff2array(beta_n, w):
    r"""Helper function definig the propagation constant

    Args:
        w (:obj:`numpy.ndarray`): Angular frequency grid.

    Returns:
        :obj:`numpy.ndarray` Propagation constant as function of
        frequency detuning.
    """
    beta_n = np.asarray(beta_n)
    n_fac = factorial(np.arange(beta_n.size)+2, exact=True)
    beta_w_fun = np.poly1d(np.pad(beta_n/n_fac,(2,0))[::-1])
    return beta_w_fun(w)


class GNLS(object):
    r"""Generalized nonlinear Schrödinger equation.

    Implements the generalized nonlinear Schrödinger equation (GNLS) [1].

    References:
        [1] G. P. Agrawal, Nonlinear Fiber Optics, Academic Press (2019).

    Args:
        w (:obj:`numpy.ndarray`):
            Angular frequency grid.
        beta_w (:obj:`numpy.ndarray`):
            Propagation constant.
        gamma (:obj:`float`):
            Nonlinear parameter (default=1.0 1/W/micron).
        w0 (:obj:`float`):
            Referrence frequency (default=np.inf rad/fs).
        fR (:obj:`float`):
            Fractional Raman contribution (default=0.).
        hRw (:obj:`float`):
            Frequency domain represeentation of Raman response (default=1.).
    """
    def __init__(self, w, beta_n, gamma=1., w0 = np.inf, fR=0.18, tau1=12.2, tau2=30.0):
        self.w = w
        self.beta_w = beta_w_coeff2array(beta_n, w)
        self.gamma = gamma
        self.w0 = w0
        self.fR = fR
        self.hRw = (tau1**2+tau2**2)/(tau1**2*(1-1j*w*tau2)**2+tau2**2)

    @property
    def Lw(self):
        r"""Frequency-domain representation of the linear dispersion operator.

        Returns: (Lw)
            Lw (:obj:`numpy.ndarray`):
                action of the linear part
        """
        return 1j*self.beta_w

    def Nw(self, uw):
        r"""Frequency-domain representation of the nonlinear operator.

        Args:
            uw (:obj:`numpy.ndarray`):
                Frequency-domain representation of field envelope at current
                :math:`z`-position.

        Returns: (Nw)
            Nw (:obj:`numpy.ndarray`):
                action of the nonlinear part
        """
        # -- STRIP OFF SELF KEYWORD
        w, w0, gamma, fR, hRw = self.w, self.w0, self.gamma, self.fR, self.hRw
        # -- NONLINEAR FUNCTIONAL HANDLING THE RAMAN RESPONSE
        _NR = lambda u: (1-fR)*np.abs(u)**2*u + fR*u*IFT(FT(np.abs(u)**2)*hRw)
        # -- APPLY FULL NONLINEAR OPERATION
        return 1j*gamma*(1.+w/w0)*FT(_NR(IFT(uw)))

    def claw_Ph(self, i, zi, w, uw):
        r"""Conservation law of the propagation model.

        Implements conserved quantity related to the classical analog of the
        photon number [1], defined by

        .. math::
            C_{Ph}(z)= \frac{2\pi}{\hbar \Delta \Omega} \sum_{\Omega} |u_\omega(z)|^2/(\Omega+\omega_0).

        Args:
            i (:obj:`int`):
                Index specifying the current :math:`z`-step.
            zi (:obj:`float`):
                Current :math:`z`-value.
            w (:obj:`numpy.ndarray`):
                Angular frequency mesh.
            uw (:obj:`numpy.ndarray`):
                Freuqency domain representation of the current field.

        References:
            [1] K. J. Blow, D. Wood, Theoretical description of transient
            stimulated Raman scattering in optical fibers.  IEEE J. Quantum
            Electron., 25 (1989) 1159, https://doi.org/10.1109/3.40655.

        Notes:
            - Squared amplitude is measured in units of Watts (W=J/s), hBar is
              given in units J fs, and w and w0  are given in rad/fs. So as to
              account for the difference in units, an additional factor of 1e15
              is needed to get a proper number of photons.

        Returns: (C_Ph)
            CPh (float): Total number of photons.
        """
        # -- STRIP OFF SELF KEYWORD
        w, w0 = self.w, self.w0
        # -- SCALING FACTOR WITH UNITS OF 1/J
        dw = w[1]-w[0]              # (rad/fs)
        sFac = 2*np.pi/(hBar*dw)    # (1/J)
        # --- DIMENSIONLES TOTAL NUMBER OF PHOTONS
        # ... np.abs(uw)**2 HAS UNITS (W=J/s = 1e15 J/fs )
        # ... w+w0 HAS UNITS (rad/fs)
        # ... ADDITIONAL FACTOR 1e-15 RESOLVES THIS ISSUE
        return 1e-15*sFac*np.sum(np.abs(uw)**2/(w + w0))

