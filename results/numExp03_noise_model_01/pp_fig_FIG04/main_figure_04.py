"""
Author: O. Melchert
Date: 2020-09-09
"""
import sys
import os
import itertools
import numpy as np
import numpy.fft as nfft
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import sem
from scipy.constants import Planck  as hPlanck

# -- CONVENIENT ABBREVIATIONS
hBar = 1e15*hPlanck/2/np.pi # (J fs)

# -- CONVENIENT ABBREVIATIONS
FT = nfft.ifft
IFT = nfft.fft
SHIFT = nfft.fftshift


def fetch_data(f_list):
    """ fetch data from npz-files

    Args:
      z0 (float): selected propagation distance

    Returns: (w, Ewz_list)
      w (1D array, floats): cropped angular frequency axis
      Ewz_list (2D array, floats): cropped frequency doman arrays
    """

    def _helper(iPath):
        data = np.load(iPath)
        return data['z'], data['t'], data['w'], FT(data['utz'])

    Ewz_list = []
    for fName in f_list:

        # -- FETCH RAW DATA
        (z, t, w, Ewz) = _helper(fName)
        z = z*1e-6   # rescale to m
        w = SHIFT(w)

        Ewz = SHIFT(Ewz,axes=-1)
        Ewz_list += [Ewz]

    return z, w, np.asarray(Ewz_list)


def set_legend(ax, lines, loc=0, ncol=1):
    """set legend

    Function generating a custom legend, see [1] for more options

    Refs:
      [1] https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend.html

    Args:
      ax (object): figure part for which the legend is intended
      lines (list): list of  Line2D objects
    """
    # -- extract labels from lines
    labels = [x.get_label() for x in lines]
    # -- customize legend 
    ax.legend(lines,                # list of Line2D objects
              labels,               # labels 
              title = '',           # title shown on top of legend 
              loc = loc,              # location of the legend
              ncol = ncol,             # number of columns
              labelspacing = 0.3,   # vertical space between handles in font-size units
              borderpad = 0.3,      # distance to legend border in font-size units
              handletextpad = 0.3,  # distance between handle and label in font-size units
              handlelength = 1.5,   # length of handle in font-size units
              frameon = False # remove background patch
              )


def custom_colormap():
    # -- CREATE CUSTOM COLORMAP
    from matplotlib.colors import ListedColormap
    cmap_base = mpl.cm.jet(np.arange(256))
    blank = np.ones((12,4))
    for i in range(3):
       blank[:,i] = np.linspace(1,cmap_base[0,i], blank.shape[0])
    my_cmap = ListedColormap(np.vstack((blank, cmap_base )))
    return my_cmap


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
            'legend.frameon': True,
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


    def set_subfig_label(self,ax, label1, label2):
            pos = ax.get_position()

            self.fig.text(pos.x0, pos.y1+0.0, label1 ,color='white',
                backgroundcolor='k', bbox=dict(facecolor='k', edgecolor='none',
                boxstyle='square,pad=0.1'), verticalalignment='top' )

            self.fig.text(pos.x0+0.04, pos.y1-0.005, label2 ,color='k',
                verticalalignment='top' )


    def set_layout(self):

        fig = plt.figure()
        self.fig = fig

        plt.subplots_adjust(left = 0.14, bottom = 0.09, right = 0.88, top = 0.98, wspace = .075, hspace = 1.)

        gs00 = GridSpec(nrows = 1, ncols = 1)

        gsA = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs00[0,0], wspace=0.07, hspace=0.07)
        axA1 = fig.add_subplot(gsA[0, 0])
        axA2 = fig.add_subplot(gsA[1, 0])
        axA3 = fig.add_subplot(gsA[2, 0])
        self.subfig_1 = [axA1, axA2, axA3]


    def set_subfig_03(self, axB2, z0, f_list):

        _dB = lambda x: np.where(x>1e-20,10.*np.log10(x),10*np.log10(1e-20))

        c0 = 0.29979
        w0 = 2.2559
        lam_lim = (400,1700)
        lam_ticks = (500,800,1100,1400,1600)
        y_lim = (0,1.1)
        y_ticks = (0,0.2,0.4,0.6,0.8,1.0)



        # -- T0 = 100 ---------------------------------------------------------
        #path = '../data_delG1e-08_t0FWHM100.000000/'
        #f_list = [path+f for f in os.listdir(path)[:]]

        dat = np.load(f_list[0])
        w0 = dat['w0']

        z, w, uwz_list = fetch_data(f_list)

        lam = 1e3*2*np.pi*c0/(w0+w) # (nm)
        lam_mask = np.logical_and(lam>lam_lim[0],lam<lam_lim[1])
        Iw_av = np.mean(np.abs(uwz_list)**2, axis=0)
        Iw_av /= np.max(Iw_av)



        #sem = np.var

        Iw_sem = sem(np.abs(uwz_list[:40])**2, axis=0)
        Iw_av = np.mean(np.abs(uwz_list[:40])**2, axis=0)
        #Iw_av /= np.max(Iw_av)
        #Iw_sem /= Iw_av
        axB2.fill_between(lam[lam_mask],0,Iw_sem[lam_mask], lw=0.5, color='lightgrey', label = r'$M=40$')

        Iw_sem = sem(np.abs(uwz_list[:100])**2, axis=0)
        Iw_av = np.mean(np.abs(uwz_list[:100])**2, axis=0)
        #Iw_av /= np.max(Iw_av)
        #Iw_sem /= Iw_av
        axB2.fill_between(lam[lam_mask],0,Iw_sem[lam_mask], lw=0.5, color='darkgrey', label=r'$M=100$')

        Iw_sem = sem(np.abs(uwz_list[:])**2, axis=0)
        Iw_av = np.mean(np.abs(uwz_list[:])**2, axis=0)
        #Iw_av /= np.max(Iw_av)
        #Iw_sem /= Iw_av
        axB2.fill_between(lam[lam_mask],0,Iw_sem[lam_mask], lw=0.5, color='grey', label=r'$M=200$')

    

        #axB2.plot(lam[lam_mask],_dB(Iw_av[lam_mask]), lw=0.5, color='k')

        axB2.tick_params(axis='y', length=2., pad=2, top=False)
        #axB2.set_ylim(-85,5)
        #axB2.set_yticks((-80,-60,-40,-20.,0))
        axB2.set_ylabel(r"Standard error")
        axB2.yaxis.set_label_coords(-0.14,0.5)

        axB2.tick_params(axis='x', length=2., pad=2, top=False)
        axB2.set_xlim(lam_lim)
        axB2.set_xticks(lam_ticks)
        #axB2.set_xlabel(r"Wavelength $\lambda~\mathrm{(nm)}$", labelpad=1)


        ax = axB2.twinx()

        ax.plot(lam[lam_mask],_dB(Iw_av[lam_mask]), lw=0.5, color='C0', label=r'$\langle I_\lambda\rangle~\rightarrow$', zorder=0)

        ax.tick_params(axis='y', length=2., pad=2, top=False)
        ax.set_ylim(-85,5)
        ax.set_yticks((-80,-60,-40,-20.,0))
        ax.set_ylabel(r"Spectrum")

        ax.tick_params(axis='x', length=2., pad=2, top=False)
        ax.set_xlim(lam_lim)
        ax.set_xticks(lam_ticks)
        #ax.set_xlabel(r"Wavelength $\lambda~\mathrm{(nm)}$", labelpad=1)


        axB2.legend(handlelength=1., handletextpad=0.3,  markerfirst=True, title=r'$\leftarrow {\mathrm{stdErr}}(I_\lambda)$', fontsize=5, title_fontsize=5, facecolor='grey',loc=(0.81,0.3), frameon=False)
        ax.legend(handlelength=1., handletextpad=0.3,  fontsize=4,facecolor='grey',edgecolor='white', loc=(0.86,0.85),frameon=False)


def main():

    myFig = Figure(aspect_ratio=.85, fig_basename="fig04", fig_format='png')
    myFig.set_layout()

    axB1, axB2, axB3 = myFig.subfig_1

    path = '../data_delG1e-08_t0FWHM50.000000/'
    f_list = [path+f for f in os.listdir(path)[:]]
    myFig.set_subfig_03(axB1, 0.1, f_list)
    axB1.tick_params(axis='x', length=2., pad=2, top=False, labelbottom=False)
    axB1.set_ylim(0,0.005)
    axB1.set_yticks((0.,0.002,0.004))
    myFig.set_subfig_label(axB1,'(a)', r'$t_0=28.4~\mathrm{fs}$')

    path = '../data_delG1e-08_t0FWHM100.000000/'
    f_list = [path+f for f in os.listdir(path)[:]]
    myFig.set_subfig_03(axB2, 0.1, f_list)
    axB2.tick_params(axis='x', length=2., pad=2, top=False, labelbottom=False)
    axB2.set_ylim(0,0.07)
    axB2.set_yticks((0.,0.02,0.04,0.06))
    myFig.set_subfig_label(axB2,'(b)', r'$t_0=56.7~\mathrm{fs}$')

    path = '../data_delG1e-08_t0FWHM150.000000/'
    f_list = [path+f for f in os.listdir(path)[:]]
    myFig.set_subfig_03(axB3, 0.1, f_list)
    axB3.set_xlabel(r"Wavelength $\lambda~\mathrm{(nm)}$", labelpad=1)
    axB3.set_ylim(0,0.07)
    axB3.set_yticks((0.,0.02,0.04,0.06))
    myFig.set_subfig_label(axB3,'(c)', r'$t_0=85.0~\mathrm{fs}$')

    myFig.save()



if __name__ == '__main__':
    main()
