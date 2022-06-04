import sys; sys.path.append("../../src/")
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.constants import Planck  as hPlanck
from GNLStools import *

# -- CONVENIENT ABBREVIATIONS
hBar = 1e15*hPlanck/2/np.pi # (J fs)
FT = np.fft.ifft
IFT = np.fft.fft
SHIFT = np.fft.fftshift


class Figure():

    def __init__(self, aspect_ratio = 1.0, fig_format = 'png', fig_basename = 'fig_test'):
        self.fig_format = fig_format
        self.fig_basename = fig_basename
        self.fig = None
        self.set_style(aspect_ratio)

    def set_style(self, aspect_ratio = 1.0):

        fig_width = 3.2 # (inch)
        fig_height = aspect_ratio*fig_width

        params = {
            'figure.figsize': (fig_width,fig_height),
            'legend.fontsize': 6,
            'legend.frameon': False,
            'axes.labelsize': 6,
            'axes.linewidth': 0.8,
            'xtick.labelsize' :6,
            'ytick.labelsize': 6,
            'mathtext.fontset': 'stixsans',
            'mathtext.rm': 'serif',
            'mathtext.bf': 'serif:bold',
            'mathtext.it': 'serif:italic',
            'mathtext.sf': 'sans\\-serif',
            'font.size':  6,
            'font.family': 'serif',
            'font.serif': "Helvetica",
        }
        mpl.rcParams.update(params)


    def save(self):
        fig_format, fig_name = self.fig_format, self.fig_basename
        if fig_format == 'png':
            plt.savefig(fig_name+'.png', format='png', dpi=600)
        elif fig_format == 'pdf':
            plt.savefig(fig_name+'.pdf', format='pdf', dpi=600)
        elif fig_format == 'svg':
            plt.savefig(fig_name+'.svg', format='svg', dpi=600)
        else:
            plt.show()


    def set_subfig_label(self,ax, label1):
            pos = ax.get_position()

            self.fig.text(pos.x0, pos.y1, label1 ,color='white',
                backgroundcolor='k', bbox=dict(facecolor='k', edgecolor='none',
                boxstyle='square,pad=0.1'), verticalalignment='top' )


    def set_layout(self):

        fig = plt.figure()
        self.fig = fig

        plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.98, top = 0.98, wspace = .05, hspace = 0.75)

        gs00 = GridSpec(nrows = 1, ncols = 1)

        gsA = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs00[0,0], wspace=0.07, hspace=0.06)
        axA1 = fig.add_subplot(gsA[0, 0])
        axA2 = fig.add_subplot(gsA[1, 0])
        self.subfigs = [axA1, axA2]


    def figure(self):

        ax2, ax1 = self.subfigs

        w0 = 2.2559         # (rad/fs)
        t_max = 2000.       # (fs)
        t_num = 2**13       # (-)
        t = np.linspace(-t_max, t_max, t_num, endpoint=False)
        dt = t[1]-t[0]
        e0 = hBar*w0/2      # (J)
        P0 = 1e15*e0/dt     # (W=J/s)

        def _sampling(model, Ns=1000, M=25):
            n0 = int(t.size/2)
            p = np.asarray([model(t,w0,s0) for s0 in range(Ns)])
            # -- NOISE AVERAGE
            av = np.mean(p,axis=0)
            # -- NOISE AUTOCORRELATION
            p0 = p[:,n0]
            p *= np.conj(p0)[:,np.newaxis]
            ac = np.real(np.mean(p,axis=0))
            return t/dt, av, ac

        Ns = 10000
        idx_1, av_1, ac_1 = _sampling(noise_model_01, Ns=Ns, M=25)
        idx_2, av_2, ac_2 = _sampling(noise_model_02, Ns=Ns, M=25)
        idx_3, av_3, ac_3 = _sampling(noise_model_03, Ns=Ns, M=25)

        ax1.plot(idx_1, ac_1/P0, lw=1., color='C0', label=r'direct sampling')

        ax1.plot(idx_2, ac_2/P0, lw=1., color='C0', dashes=[3,1], label='Fourier method, phase noise')
        ax1.plot(idx_3, ac_3/P0, lw=1., color='C0', dashes=[1,1], label='Fourier method, Gaussian noise')


        x_lim = (-t_max/dt,t_max/dt)
        x_ticks = (-4000,-3000,-2000,-1000,0,1000,2000,3000,4000)
        ax1.tick_params(axis='x', length=2., pad=2, top=False)
        ax1.set_xlim(x_lim)
        ax1.set_xticks(x_ticks)

        ax1.tick_params(axis='y', length=2., pad=2, top=False)
        ax1.set_ylim(-1.,3.5)
        ax1.set_yticks((-1.,0.,1.,2.,3.))
        ax1.set_ylabel(r"$(\hbar \omega_0/2 \Delta t)^{-1} {\mathsf{Re}}\left[\langle\Delta u_m \Delta u_{0}^{*}\rangle\right]$")
        ax1.set_xlabel(r"Index $m$", labelpad=1)


        dw = 0.4
        dh = 0.6
        axA3 = ax1.inset_axes([1.0-dw-0.015,1.-dh-0.025, dw, dh])

        axA3.plot(idx_1, ac_1/P0, lw=1., color='C0')
        axA3.plot(idx_2, ac_2/P0, lw=1., color='C0', dashes=[3,1])
        axA3.plot(idx_3, ac_3/P0, lw=1., color='C0', dashes=[1,1])
        axA3.axhline(1, color='k', dashes=[2,2], lw=0.75)

        #a0 = t.size*np.pi/(2*t_max)
        #xr = np.linspace(-dt*30,dt*30,2000)
        #jr = 0.5*a0*np.sin(a0*xr)/xr/a0/np.pi
        #axA3.plot(xr/dt, 2*jr , lw=1., color='gray')

        my_col='k'
        axA3.spines['top'].set_color(my_col)
        axA3.spines['bottom'].set_color(my_col)
        axA3.spines['left'].set_color(my_col)
        axA3.spines['right'].set_color(my_col)

        x_lim_ = (-6,6)
        x_ticks_ = (-6,-3,0,3,6)
        axA3.tick_params(axis='x', length=2., pad=2, top=False)
        axA3.set_xlim(x_lim_)
        axA3.set_xticks(x_ticks_)

        axA3.tick_params(axis='y', length=2., pad=2, top=False)
        axA3.set_ylim(-1.25,3.5)
        axA3.set_yticks((-1.,0.,1.,2.,3.))

        ax2.plot(idx_1, np.real(av_1)*1e3, lw=1., color='C0', label = r'noise model 01')

        ax2.plot(idx_2, np.real(av_2)*1e3, lw=1., color='C0', dashes=[3,1], label='noise model 02')
        ax2.plot(idx_3, np.real(av_3)*1e3, lw=1., color='C0', dashes=[1,1], label='noise model 03')

        ax2.tick_params(axis='y', length=2., pad=2, top=False)
        ax2.set_ylim(-2.5,2.5)
        ax2.set_yticks((-2,-1,0,1,2))
        ax2.set_ylabel(r"${\mathsf{Re}}\left[\langle\Delta u_m \rangle\right]\,\times 10^{-3}$")
        ax2.tick_params(axis='x', length=2., pad=2, top=False, labelbottom=False)
        ax2.set_xlim(x_lim)
        ax2.set_xticks(x_ticks)

        ax2.legend(handlelength=1.5, loc=(0.7,0.64))

        self.set_subfig_label(ax1,"(b)")
        self.set_subfig_label(ax2,"(a)")


def main():
    myFig = Figure(aspect_ratio=.75, fig_basename="fig00", fig_format='png')
    myFig.set_layout()
    myFig.figure()
    myFig.save()

if __name__ == '__main__':
    main()
