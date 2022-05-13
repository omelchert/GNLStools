'''GNLStools.py


Author: O. Melchert
Date: March 2022
'''
import numpy as np
import numpy.fft as nfft
from scipy.special import factorial
from scipy.constants import Planck  as hPlanck

# -- CONVENIENT ABBREVIATIONS
hBar = hPlanck/2/np.pi # (J s)

FT=nfft.ifft
IFT=nfft.fft


def qnoise_spectral_synthesis(ts,w0,s0):
    """quantum noise through spectral synthesis

    generates quantum noise by spectral synthesis. The algorithm
    implements pure phase noise in the Fourier domain.

    Args:
        t (1D numpy-array, floats): time grid
        w0 (float): pulse center frequency
        s0 (int): seed for random number generator

    Returns: (du)
        du (1D numpy-array, cplx floats): time-domain representation
            of quantum noise
    """

    # -- PROPERLY NORMALIZED FOURIER TRANSFORM PAIR
    def _ifft(w, Xw):
        return (w[1]-w[0])*nfft.fft(Xw)/np.sqrt(2.*np.pi)

    def _fft(t, Xt):
        return (t[1]-t[0])*nfft.ifft(Xt)/np.sqrt(2.*np.pi)*t.size

    nn = 1 #16
    tMax = -ts.min()
    Nt = nn*ts.size
    t = np.linspace(-nn*tMax, nn*tMax, Nt, endpoint=False)

    w = nfft.fftfreq(t.size,d=t[1]-t[0])*2*np.pi
    dw = w[1]-w[0]

    np.random.seed(s0)

    # -- REPRESENTATIVE ENERGY OF PHOTON IN BIN 
    #e0 = hBar*w0/1e-15  # (Js*rad/s = J rad)
    #e0 = hBar*(w+w0)/1e-15  # (Js*rad/s = J rad)
    e0 = np.where(w+w0>0, hBar*(w+w0)/1e-15, 0) # (Js*rad/s = J rad)
    # -- NOISE SCALING FACTOR
    sFac = np.sqrt(e0/(dw/1e-15)) # (sqrt(J rad * s/rad) = sqrt(Js))
    # -- RANDOM NOISE IN SPECTRAL PHASE
    noise_w = sFac*np.exp(1j*2*np.pi*np.random.uniform(size=w.size))
    # -- CONVERT BACK TO TIME DOMAIN
    noise_t = _ifft(w,noise_w)/1e-15  # ( sqrt(Js)*rad/s = sqrt(J/s)*rad)

    return noise_t[int((t.size-ts.size)/2): int((t.size+ts.size)/2)]


def qnoise_direct_sampling(t,w0,s0):
    """quantum noise through spectral synthesis

    generates quantum noise by spectral synthesis. The algorithm
    implements pure phase noise in the Fourier domain.

    Args:
        t (1D numpy-array, floats): time grid
        w0 (float): pulse center frequency
        s0 (int): seed for random number generator

    Returns: (du)
        du (1D numpy-array, cplx floats): time-domain representation
            of quantum noise
    """
    np.random.seed(s0)
    _N01 = np.random.normal
    dt = t[1]-t[0]

    # -- REPRESENTATIVE ENERGY OF PHOTON IN BIN 
    e0 = hBar*(w0*1e15) # (J)
    # -- NOISE SCALING FACTOR
    sFac = np.sqrt(e0/(dt*1e-15)/2) # (W = J/s)
    # -- GAUSSIAN NOISE MODEL IN TIME DOMAIN 
    noise_t = sFac*(_N01(0,1,size=t.size) + 1j*_N01(0,1,size=t.size))

    return noise_t


def noise_variance(x):
    """quantum noise variance

    computes variance of complex amplitudes in time domain to
    assess the correctness of the used quantum noise model.

    Note:
       - implements variance consistent with Eq. (27) in Ref. [1]

    Args:
        x (1D numpy array, complex): complex amplitudes specifying noise

    Returns: (var)
        var (float): variance of complex noise increments

    Reference:
        [1] Noise of mode-locked lasers (Part I): numerical model
            R. Paschotta
            Appl. Phys. B 79, 153--162 (2004)
    """
    return np.mean( np.abs(np.real(x))**2 + np.abs(np.imag(x))**2)


def number_of_photons(t,u0,w0):
    """number of photons in pulse

    computes the total number photons in the pulse, based on the assumption
    that all photons contribute the same energy (hbar omega_0)

    Args:
        t (numpy array, 1D): time samples
        u0 (numpy array, 1D): complex field
        w0 (float): pulse center frequency

    Returns: (N0)
        N0 (float): total number of photons
    """
    # -- TOTAL ENERGY IN PULSE
    E = np.trapz(np.abs(u0)**2,x=t) # (J)
    # -- PHOTON ENERGY
    E0 = hBar*(w0*1e15) # (J)
    return E/E0


def coherence_interpulse(w, uw_list):

    Iw_av = np.mean(np.abs(uw_list)**2, axis=0)

    nPairs = 0
    tmp = np.zeros(len(uw_list[0]),dtype=complex)
    for i,j in list(itertools.combinations(range(len(uw_list)),2)):
       tmp += uw_list[i]*np.conj(uw_list[j])
       nPairs += 1
    tmp = np.abs(tmp)/nPairs

    return np.real(tmp/Iw_av), Iw_av


def coherence_intrapulse(w, uw_list, w1, w2):
    r""" intrapulse coherence function

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
          Phys. Rev. Lett. 119, 123901 (2017)
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
        [1] G. P. Agrawal, Nonlinear Fiber Optics, Academic Press (2019)

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

        Implements conserved quantity related to the photon number [1].

        References:
            [1] K. J. Blow, D. Wood, Theoretical description of transient
            stimulated Raman scattering in optical fibers.  IEEE J. Quantum
            Electron., 25 (1989) 1159, https://doi.org/10.1109/3.40655.

        Args:
            i (:obj:`int`):
                Index specifying the current :math:`z`-step.
            zi (:obj:`float`):
                Current :math:`z`-value.
            w (:obj:`numpy.ndarray`):
                Angular frequency mesh.
            uw (:obj:`numpy.ndarray`):
                Freuqency domain representation of the current field.

        Returns: (C_Ph)
            :obj:`float`: total number of photons.
        """
        # -- STRIP OFF SELF KEYWORD
        w, w0 = self.w, self.w0
        # -- SCALING FACTOR WITH UNITS OF 1/J
        sFac = 2*np.pi/hBar/(w[1]-w[0])/1e15
        # --- DIMENSIONLES TOTAL NUMBER OF PHOTONS
        return sFac*np.sum(np.abs(uw)**2/(w + w0))

