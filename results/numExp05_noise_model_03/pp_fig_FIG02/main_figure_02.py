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


    def set_subfig_label(self,ax, label1):
            pos = ax.get_position()

            self.fig.text(pos.x0, pos.y1+0.01, label1 ,color='white',
                backgroundcolor='k', bbox=dict(facecolor='k', edgecolor='none',
                boxstyle='square,pad=0.1'), verticalalignment='bottom' )

            #self.fig.text(pos.x0+0.04, pos.y1+0.01, label2 ,color='k',
            #    verticalalignment='bottom' )


    def set_layout(self):

        fig = plt.figure()
        self.fig = fig

        plt.subplots_adjust(left = 0.08, bottom = 0.11, right = 0.9, top = 0.93, wspace = .05, hspace = 0.9)

        gs00 = GridSpec(nrows = 1, ncols = 2)

        gsA = GridSpecFromSubplotSpec(6, 1, subplot_spec=gs00[0,0], wspace=0.07, hspace=0.1)
        axA1 = fig.add_subplot(gsA[0, 0])
        axA2 = fig.add_subplot(gsA[1, 0])
        axA3 = fig.add_subplot(gsA[2, 0])
        axA4 = fig.add_subplot(gsA[3, 0])
        axA5 = fig.add_subplot(gsA[4, 0])
        axA6 = fig.add_subplot(gsA[5, 0])
        self.subfig_1 = [axA1, axA2, axA3, axA4, axA5, axA6]

        gsA = GridSpecFromSubplotSpec(6, 1, subplot_spec=gs00[0,1], wspace=0.07, hspace=0.1)
        axA1 = fig.add_subplot(gsA[0, 0])
        axA2 = fig.add_subplot(gsA[1, 0])
        axA3 = fig.add_subplot(gsA[2, 0])
        axA4 = fig.add_subplot(gsA[3, 0])
        axA5 = fig.add_subplot(gsA[4, 0])
        axA6 = fig.add_subplot(gsA[5, 0])
        self.subfig_2 = [axA1, axA2, axA3, axA4, axA5, axA6]



    def set_figs(self, f_list):
        sf1 = self.subfig_1

        t_lim = (-0.2,3.2)
        t_ticks = (0,1,2,3)
        y_lim = (0,10.)
        y_ticks = (0,4,8)

        idx_list = np.arange(len(f_list))
        idx_list = np.random.permutation(idx_list)
        print(idx_list)
        idx_list = [ 9 , 1, 10,  3, 12, 16 , 2 , 6,  7, 11,  5, 13, 19, 14, 18, 15,  8, 17,  0 , 4]


        for idx, ax in enumerate(sf1):

            dat = np.load(f_list[idx_list[idx]])
            z = dat['z']*1e-6   # (m)
            t = dat['t']*1e-3   # (ps)
            w = dat['w']        # (rad/fs)
            w0 = dat['w0']      # (rad/fs)
            P0 = dat['P0']      # (W)
            utz = dat['utz']    # (W)

            I = np.abs(utz)**2/1000

            ax.plot(t,I,lw=0.75)

            ax.set_xlim(t_lim)
            ax.set_xticks(t_ticks)
            ax.set_ylim(y_lim)
            ax.set_yticks(y_ticks)

            if idx==2:
                ax.set_ylabel(r"Intensity $I(t)~\mathrm{(kW)}$", y=0)
            ax.tick_params(axis='y', length=2., pad=2, top=False)


            if idx==len(sf1)-1:
                ax.spines.right.set_color('none')
                ax.spines.top.set_color('none')
                ax.tick_params(axis='x', length=2., pad=2, top=False)
                ax.set_xlabel(r"Time $t~\mathrm{(ps)}$", labelpad=1)
            else:
                ax.spines.right.set_color('none')
                ax.spines.top.set_color('none')
                #ax.spines.bottom.set_color('none')
                ax.tick_params(axis='x', length=2., pad=2, top=False, bottom=True)
                #ax.tick_params(axis='x', length=2., pad=2, top=False, labelbottom=False, bottom=False)


        sf2 = self.subfig_2

        lam_lim = (400,1500)
        lam_ticks = (500,800,1100,1400)
        Iw_lim = (0,1.04)
        Iw_ticks = (0,0.2,0.4,0.6,0.8,1.0)

        for idx, ax in enumerate(sf2):

            dat = np.load(f_list[idx_list[idx]])
            z = dat['z']*1e-6   # (m)
            t = dat['t']*1e-3   # (ps)
            w = dat['w']        # (rad/fs)
            w0 = dat['w0']      # (rad/fs)
            P0 = dat['P0']      # (W)
            utz = dat['utz']    # (W)

            w = SHIFT(w)

            c0 = 0.29979
            lam = 1e3*2*np.pi*c0/(w0+w) # (nm)
            lam_mask = np.logical_and(lam>lam_lim[0],lam<lam_lim[1])
            _dB = lambda x: np.where(x>1e-20,10.*np.log10(x),10*np.log10(1e-20))

            Iw = SHIFT(np.abs(FT(utz))**2)
            Iw /= Iw.max()

            ax.plot(lam[lam_mask], _dB(Iw[lam_mask]), lw=0.75)

            ax.set_ylim(-85,5)
            ax.set_yticks((-70,-40,-10))
            #ax.set_ylabel(r"Intensity $I_\Omega~\mathrm{(dB)}$")

            #axB2.tick_params(axis='y', length=2., pad=2, top=False)
            #ax.set_ylim(y_lim)
            #ax.set_yticks(y_ticks)
            #axB2.set_ylabel(r"Coherence $|g_{12}^{(1)}|$")

            #ax.tick_params(axis='x', length=2., pad=2, top=False)
            ax.set_xlim(lam_lim)
            ax.set_xticks(lam_ticks)

            if idx==2:
                ax.set_ylabel(r"Spectrum $I_\Omega~\mathrm{(dB)}$",y=0)
                ax.yaxis.set_label_position('right')
            ax.tick_params(axis='y', length=2., pad=2, left=False, right=True, labelleft=False, labelright=True)

            if idx==len(sf1)-1:
                ax.spines.left.set_color('none')
                ax.spines.top.set_color('none')
                ax.tick_params(axis='x', length=2., pad=2, top=False)
                ax.set_xlabel(r"Wavelength $\lambda~\mathrm{(nm)}$", labelpad=1)
            else:
                ax.spines.left.set_color('none')
                ax.spines.top.set_color('none')
                #ax.spines.bottom.set_color('none')
                ax.tick_params(axis='x', length=2., pad=2, top=False, bottom=True)
                #ax.tick_params(axis='x', length=2., pad=2, top=False, labelbottom=False, bottom=False)

        self.set_subfig_label(sf1[0],"(a)")
        self.set_subfig_label(sf2[0],"(b)")




def main():
    #path = sys.argv[1]
    path = '../data_delG1e-08_t0FWHM150.000000/'
    f_list = [path+f for f in os.listdir(path)[:]]

    myFig = Figure(aspect_ratio=.85, fig_basename="fig02", fig_format='png')
    myFig.set_layout()

    myFig.set_figs(f_list)

    myFig.save()

if __name__ == '__main__':
    main()
