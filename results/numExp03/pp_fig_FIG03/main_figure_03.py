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
        #Eps_w = SHIFT(Eps_w,axes=-1)

        # -- GET FIELD CORRESPONDING TO SELECTED PROP. DIST.
        #Ewz = Eps_w[getClosestPosition(z,z0)]

        # -- KEEP CROPPED FIELD
        #wMask = np.logical_and(w>-3.,w<3.)
        #Ewz_list += [Ewz[wMask]]
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

        fig_width = 3.4 # (inch)
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


    def set_subfig_label(self,ax, label1, label2):
            pos = ax.get_position()

            self.fig.text(pos.x0, pos.y1+0.01, label1 ,color='white',
                backgroundcolor='k', bbox=dict(facecolor='k', edgecolor='none',
                boxstyle='square,pad=0.1'), verticalalignment='bottom' )

            self.fig.text(pos.x0+0.04, pos.y1+0.01, label2 ,color='k',
                verticalalignment='bottom' )


    def set_layout(self):

        fig = plt.figure()
        self.fig = fig

        plt.subplots_adjust(left = 0.11, bottom = 0.08, right = 0.98, top = 0.95, wspace = .075, hspace = 0.9)

        gs00 = GridSpec(nrows = 5, ncols = 3)

        gsA = GridSpecFromSubplotSpec(1, 1, subplot_spec=gs00[0:2,0], wspace=0.07, hspace=0.1)
        axA1 = fig.add_subplot(gsA[0, 0])
        gsB = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs00[2:,0], wspace=0.07, hspace=0.075)
        axB1 = fig.add_subplot(gsB[0, 0])
        axB2 = fig.add_subplot(gsB[1, 0])
        self.subfig_1 = [axA1, axB1, axB2]

        gsA = GridSpecFromSubplotSpec(1, 1, subplot_spec=gs00[0:2,1], wspace=0.07, hspace=0.1)
        axA1 = fig.add_subplot(gsA[0, 0])
        gsB = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs00[2:,1], wspace=0.07, hspace=0.075)
        axB1 = fig.add_subplot(gsB[0, 0])
        axB2 = fig.add_subplot(gsB[1, 0])
        self.subfig_2 = [axA1, axB1, axB2]

        gsA = GridSpecFromSubplotSpec(1, 1, subplot_spec=gs00[0:2,2], wspace=0.07, hspace=0.1)
        axA1 = fig.add_subplot(gsA[0, 0])
        gsB = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs00[2:,2], wspace=0.07, hspace=0.075)
        axB1 = fig.add_subplot(gsB[0, 0])
        axB2 = fig.add_subplot(gsB[1, 0])
        self.subfig_3 = [axA1, axB1, axB2]



    def set_subfig_01(self, subfig, f_name, z0):
        axA1 = subfig[0]

        t_lim = (-0.2,3.2)
        t_ticks = (0,1,2,3)
        y_lim = (0,10.)
        y_ticks = (0,2,4,6,8,10)
        #y_lim = (0,1.)
        #y_ticks = (0,0.2,0.4,0.6,0.8,1.0)

        dat = np.load(f_name)
        z = dat['z']*1e-6   # (m)
        t = dat['t']*1e-3   # (ps)
        P0 = dat['P0']      # (W)
        utz = dat['utz']    # (W)

        #z_id = np.argmin(np.abs(z-z0))
        #I = np.abs(utz[z_id])**2
        I = np.abs(utz)**2/1000

        axA1.plot(t,I,lw=0.75)
        #axA1.plot(t,I/P0,lw=0.75)

        axA1.tick_params(axis='x', length=2., pad=2, top=False)
        axA1.set_xlim(t_lim)
        axA1.set_xticks(t_ticks)
        axA1.set_xlabel(r"Time $t~\mathrm{(ps)}$", labelpad=1)

        #axA1.tick_params(axis='y', length=2., pad=2, top=False)
        axA1.set_ylim(y_lim)
        axA1.set_yticks(y_ticks)
        #axA1.set_ylabel(r"Intensity $I(t)/P_0$")


    def set_subfig_02(self, subfig, f_list, z0):
        _, axB1, axB2 = subfig

        def firstOrderCoherence(w, Ewz_list):

            Iw_av = np.mean(np.abs(Ewz_list)**2, axis=0)

            nPairs = 0
            XX = np.zeros(len(Ewz_list[0]),dtype=complex)
            for i,j in list(itertools.combinations(range(len(Ewz_list)),2)):
               XX += Ewz_list[i]*np.conj(Ewz_list[j])
               nPairs += 1
            XX = np.abs(XX)/nPairs

            return np.real(XX/Iw_av), Iw_av

        dat = np.load(f_list[0])
        w0 = dat['w0']

        z, w, uwz_list = fetch_data(f_list)
        g12, Iw_av = firstOrderCoherence(w, uwz_list)

        lam_lim = (400,1500)
        lam_ticks = (500,800,1100,1400)
        y_lim = (0,1.04)
        y_ticks = (0,0.2,0.4,0.6,0.8,1.0)

        c0 = 0.29979
        lam = 1e3*2*np.pi*c0/(w0+w) # (nm)
        lam_mask = np.logical_and(lam>lam_lim[0],lam<lam_lim[1])

        _dB = lambda x: np.where(x>1e-20,10.*np.log10(x),10*np.log10(1e-20))

        Iw_av /= Iw_av.max()
        axB1.plot(lam[lam_mask], _dB(Iw_av[lam_mask]), lw=0.75)



        axB2.plot(lam[lam_mask], g12[lam_mask], lw=0.75)

        axB1.tick_params(axis='x', length=2., pad=2, top=False, labelbottom=False)
        axB1.set_xlim(lam_lim)
        axB1.set_xticks(lam_ticks)

        #axB1.tick_params(axis='y', length=2., pad=2, top=False)
        axB1.set_ylim(-85,5)
        axB1.set_yticks((-80,-60,-40,-20.,0))
        #axB1.set_ylim(-80,2)
        #axB1.set_yticks((-70,-50,-30.,-10))
        #axB1.set_ylabel(r"Intensity $I_\Omega~\mathrm{(dB)}$")

        #axB2.tick_params(axis='y', length=2., pad=2, top=False)
        axB2.set_ylim(y_lim)
        axB2.set_yticks(y_ticks)
        #axB2.set_ylabel(r"Coherence $|g_{12}^{(1)}|$")

        axB2.tick_params(axis='x', length=2., pad=2, top=False)
        axB2.set_xlim(lam_lim)
        axB2.set_xticks(lam_ticks)
        axB2.set_xlabel(r"Wavelength $\lambda~\mathrm{(nm)}$", labelpad=1)





def main():

    myFig = Figure(aspect_ratio=.85, fig_basename="fig03", fig_format='png')
    myFig.set_layout()

    path = '../data_delG1e-08_t0FWHM50.000000/'
    f_list = [path+f for f in os.listdir(path)[:]]
    myFig.set_subfig_01(myFig.subfig_1, f_list[0], 0.1)
    myFig.set_subfig_02(myFig.subfig_1, f_list, 0.1)
    axA1, axB1, axB2 = myFig.subfig_1
    axA1.tick_params(axis='y', length=2., pad=2, top=False)
    axA1.set_ylabel(r"Intensity $I(t)~\mathrm{(kW)}$")
    axB1.tick_params(axis='y', length=2., pad=2, top=False)
    axB1.set_ylabel(r"Spectrum $I_\Omega~\mathrm{(dB)}$")
    axB2.tick_params(axis='y', length=2., pad=2, top=False)
    axB2.set_ylabel(r"Coherence $|g_{12}|$")
    myFig.set_subfig_label(axA1,'(a)', r'$t_0=28.4~\mathrm{fs}$')
    axA1.yaxis.set_label_coords(-0.25,0.5)
    axB1.yaxis.set_label_coords(-0.25,0.5)
    axB2.yaxis.set_label_coords(-0.25,0.5)

    path = '../data_delG1e-08_t0FWHM100.000000/'
    f_list = [path+f for f in os.listdir(path)[:]]
    myFig.set_subfig_01(myFig.subfig_2, f_list[0], 0.1)
    myFig.set_subfig_02(myFig.subfig_2, f_list, 0.1)
    axA1, axB1, axB2 = myFig.subfig_2
    axA1.tick_params(axis='y', length=2., pad=2, top=False, labelleft=False)
    axB1.tick_params(axis='y', length=2., pad=2, top=False, labelleft=False)
    axB2.tick_params(axis='y', length=2., pad=2, top=False, labelleft=False)
    myFig.set_subfig_label(axA1,'(b)', r'$t_0=56.7~\mathrm{fs}$')

    path = '../data_delG1e-08_t0FWHM150.000000/'
    f_list = [path+f for f in os.listdir(path)[:]]
    myFig.set_subfig_01(myFig.subfig_3, f_list[0], 0.1)
    myFig.set_subfig_02(myFig.subfig_3, f_list, 0.1)
    axA1, axB1, axB2 = myFig.subfig_3
    axA1.tick_params(axis='y', length=2., pad=2, top=False, labelleft=False)
    axB1.tick_params(axis='y', length=2., pad=2, top=False, labelleft=False)
    axB2.tick_params(axis='y', length=2., pad=2, top=False, labelleft=False)
    myFig.set_subfig_label(axA1,'(c)', r'$t_0=85.0~\mathrm{fs}$')



    myFig.save()

if __name__ == '__main__':
    main()
