"""
Author: O. Melchert
Date: 2020-09-09
"""
import sys
import os
import numpy as np
import numpy.fft as nfft
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import h5py

__author__ = 'Oliver Melchert'
__date__ = '2020-09-09'


# -- CONVENIENT ABBREVIATIONS
FT = nfft.ifft
IFT = nfft.fft

def _beta_fun():
    r"""Helper function definig the propagation constant

    Implements group-velocity dispersion with expansion coefficients
    listed in Tab. I of Ref. [1]. Expansion coefficients are valid for
    :math:`lambda = 835\,\mathrm{nm}`, i.e. for :math:`\omega_0 \approx
    2.56\,\mathrm{rad/fs}`.

    References:
        [1] J. M. Dudley, G. Genty, S. Coen,
        Supercontinuum generation in photonic crystal fiber,
        Rev. Mod. Phys. 78 (2006) 1135,
        http://dx.doi.org/10.1103/RevModPhys.78.1135

    Args:
        w (:obj:`numpy.ndarray`): Angular frequency grid.

    Returns:
        :obj:`numpy.ndarray` Propagation constant as function of
        frequency detuning.
    """
    # ... EXPANSION COEFFICIENTS DISPERSION
    b2 = -1.1830e-2  # (fs^2/micron)
    b3 = 8.1038e-2  # (fs^3/micron)
    b4 = -0.95205e-1  # (fs^4/micron)
    b5 = 2.0737e-1  # (fs^5/micron)
    b6 = -5.3943e-1  # (fs^6/micron)
    b7 = 1.3486  # (fs^7/micron)
    b8 = -2.5495  # (fs^8/micron)
    b9 = 3.0524  # (fs^9/micron)
    b10 = -1.7140  # (fs^10/micron)
    # ... PROPAGATION CONSTANT
    beta_w_fun = np.poly1d([b10/3628800, b9/362880, b8/40320, b7/5040, b6/720,
    b5/120, b4/24, b3/6, b2/2, 0.0, 0.0])
    return beta_w_fun


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
            'axes.linewidth': 1.,
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


    def set_subfig_label(self,ax, label, loc=1):
            pos = ax.get_position()

            if loc==1:
                self.fig.text(pos.x0, pos.y1, label ,color='white',
                    backgroundcolor='k', bbox=dict(facecolor='k', edgecolor='none',
                    boxstyle='square,pad=0.1'), verticalalignment='top' )

            elif loc==2:
                self.fig.text(pos.x0, pos.y0, label ,color='white',
                    backgroundcolor='k', bbox=dict(facecolor='k', edgecolor='none',
                    boxstyle='square,pad=0.1'), verticalalignment='bottom' )

            else:
                print("check label position")
                exit()


    def set_layout(self):

        fig = plt.figure()
        self.fig = fig

        plt.subplots_adjust(left = 0.1, bottom = 0.04, right = 0.89, top = 0.9, wspace = .5, hspace = 3.)



        gs00 = GridSpec(nrows = 8, ncols = 1)
        gsA = GridSpecFromSubplotSpec(3, 3, subplot_spec=gs00[:5,0], wspace=0.07, hspace=0.1)
        axA1 = fig.add_subplot(gsA[1:, 0])
        axA2 = fig.add_subplot(gsA[1:, 1:])
        axB1 = fig.add_subplot(gsA[0, 0])
        axB2 = fig.add_subplot(gsA[0, 1:])
        self.subfig_1 = [axA1, axA2, axB1, axB2]

        gsC = GridSpecFromSubplotSpec(5, 1, subplot_spec=gs00[5:,0], wspace=0.1, hspace=1.)
        axC = fig.add_subplot(gsC[0:4,0])
        self.subfig_2 = axC


    def set_subfig_01(self):
        axA1, axA2, axB1, axB2 = self.subfig_1

        #f_name = sys.argv[1]
        f_name = '../data_no_noise/res_CQE_SC_delG1e-12.npz'
        z, t, w, w0, utz, dz_int, Cp, del_G = fetch_data(f_name)
        t = t*1e-3 # fs -> ps
        z = z*1e-4 # micron -> cm
        w += w0

        t_lim = (-0.5,3.5)
        t_ticks = (0,1,2,3)
        z_lim = (0,14)
        z_ticks  = (0,4,8,12)
        w_lim = (1,4.2)
        w_ticks = (1,2,2.415,3,4)
        v_ticks = np.asarray([200, 300, 400, 500, 600])

        _norm = lambda x: x/x[0].max()
        _truncate = lambda x: np.where(x>1.e-20,x,1.e-20)
        _dB = lambda x: np.where(x>1e-20,10.*np.log10(x),10*np.log10(1e-20))

        def set_colorbar_lin(fig, img, ax, ax2, label='text', dw=0.):
            # -- EXTRACT POSITION INFORMATION FOR COLORBAR PLACEMENT
            refPos = ax.get_position()
            x0, y0, w, h = refPos.x0, refPos.y0, refPos.width, refPos.height
            h2 = ax2.get_position().height
            y2 = ax2.get_position().y0
            # -- SET NEW AXES AS REFERENCE FOR COLORBAR
            colorbar_axis = fig.add_axes([x0+dw, y2 + h2 + .03*h, w-dw, 0.04*h])
            # -- SET CUSTOM COLORBAR
            colorbar = fig.colorbar(img,        # image described by colorbar
                    cax = colorbar_axis,        # reference axex
                    orientation = 'horizontal', # colorbar orientation
                    extend = 'max'             # ends with out-of range values
                    )
            colorbar.ax.tick_params(
                    color = 'k',                # tick color 
                    labelcolor = 'k',           # label color
                    bottom = False,             # no ticks at bottom
                    labelbottom = False,        # no labels at bottom
                    labeltop = True,            # labels on top
                    top = True,                 # ticks on top
                    direction = 'out',          # place ticks outside
                    length = 2,                 # tick length in pts. 
                    labelsize = 5.,             # tick font in pts.
                    pad = 1.                    # tick-to-label distance in pts.
                    )
            fig.text(x0, y2+h2+0.03*h, label, horizontalalignment='left', verticalalignment='bottom', size=6)
            return colorbar

        def set_colorbar(fig, img, ax, ax2, label='text', dw=0.):
            # -- EXTRACT POSITION INFORMATION FOR COLORBAR PLACEMENT
            refPos = ax.get_position()
            x0, y0, w, h = refPos.x0, refPos.y0, refPos.width, refPos.height
            h2 = ax2.get_position().height
            y2 = ax2.get_position().y0
            # -- SET NEW AXES AS REFERENCE FOR COLORBAR
            colorbar_axis = fig.add_axes([x0+dw, y2 + h2 + .03*h, w-dw, 0.04*h])
            # -- SET CUSTOM COLORBAR
            colorbar = fig.colorbar(img,        # image described by colorbar
                    cax = colorbar_axis,        # reference axex
                    orientation = 'horizontal', # colorbar orientation
                    extend = 'both'             # ends with out-of range values
                    )
            colorbar.ax.tick_params(
                    color = 'k',                # tick color 
                    labelcolor = 'k',           # label color
                    bottom = False,             # no ticks at bottom
                    labelbottom = False,        # no labels at bottom
                    labeltop = True,            # labels on top
                    top = True,                 # ticks on top
                    direction = 'out',          # place ticks outside
                    length = 2,                 # tick length in pts. 
                    labelsize = 5.,             # tick font in pts.
                    pad = 1.                    # tick-to-label distance in pts.
                    )
            fig.text(x0, y2+h2+0.03*h, label, horizontalalignment='left', verticalalignment='bottom', size=6)
            return colorbar

        #my_cmap = custom_colormap()
        my_cmap = mpl.cm.get_cmap('jet')

        # -- PROPAGATION CHARACERISTICS 
        # ... BOTTOM PLOT

        Itz = np.abs(utz)**2
        #Itz = _truncate(_norm(np.abs(utz)**2))

        I0 = np.max(Itz[0])
        #for i in range(z.size):
        #    print(z[i], np.max(Itz[i])/I0)
        #exit()


#        img = axA1.pcolorfast(t, z, _dB(Itz[:-1,:-1]),
#                              vmin=-40, vmax=0.,
#                              cmap = my_cmap
#                              )
#        cb = set_colorbar(self.fig, img, axA1, axB1, label='$|\mathcal{u}|^2\,\mathrm{(dB)}$', dw=0.1)
#        cb.set_ticks((-40,-20,0))

        img = axA1.pcolorfast(t, z, Itz[:-1,:-1]/1e3,
                              vmin=0, vmax=30,
                              cmap = my_cmap
                              )
        cb = set_colorbar_lin(self.fig, img, axA1, axB1, label='$|\mathcal{u}|^2\,\mathrm{(kW)}$', dw=0.1)
        cb.set_ticks((0,10,20,30,40))

        axA1.tick_params(axis='x', length=2., pad=2, top=False)
        axA1.set_xlim(t_lim)
        axA1.set_xticks(t_ticks)
        axA1.set_xlabel(r"Time $t~\mathrm{(ps)}$", labelpad=1)

        axA1.tick_params(axis='y', length=2., pad=2, top=False)
        axA1.set_ylim(z_lim)
        axA1.set_yticks(z_ticks)
        axA1.set_ylabel(r"Propagation distance $z~\mathrm{(cm)}$")


        # ... UPPER PLOT - real field 
        z_id = np.argmin(np.abs(z-14))
        I = np.abs(utz[z_id])**2
        I0 = np.abs(utz[0])**2
        axB1.plot(t, I/1e3, color='#1f77b4', linewidth=.75)
        #axB1.plot(t, I/1e3, color='k', linewidth=.75)

        axB1.tick_params(axis='x', length=2., pad=2, top=False, labelbottom=False)
        axB1.set_xlim(t_lim)
        axB1.set_xticks(t_ticks)

        axB1.tick_params(axis='y', length=2., pad=2, top=False, labelleft=True, left=True)
        axB1.set_ylim((0,6))
        axB1.set_yticks((0,2,4,6))
        axB1.set_ylabel(r"$|\mathcal{u}|^2\,\mathrm{(kW)}$")

        # -- FREQUENCY DOMAIN
        # ... BOTTOM PLOT
        uwz = FT(utz,axis=-1)
        Iwz_s = nfft.fftshift(_truncate(_norm(np.abs(uwz)**2)),axes=-1)
        w_s  = nfft.fftshift(w)

        w_s_mask = np.logical_and(w_s>w_lim[0], w_s<6) #w_lim[1])
        Iwz_s_f = Iwz_s[:,w_s_mask]
        w_s_f = w_s[w_s_mask]
        c0 = 0.29979
        _lam = lambda w: 2*np.pi*c0/w
        _w = lambda x: 2*np.pi*c0/x
        lam = _lam(w_s)*1e3

        l_mask = np.logical_and(lam>400,lam<1500)
        img = axA2.pcolorfast(lam[l_mask], z, _dB(Iwz_s[:,l_mask][:-1,:-1]),
                              vmin=-40, vmax=0.,
                              cmap = my_cmap
                              )
        cb = set_colorbar(self.fig, img, axA2, axB2, label='$|\mathcal{u}_\Omega|^2\,\mathrm{(dB)}$', dw=0.1)
        cb.set_ticks((-40,-30,-20,-10,0))

        w_ZDW = 2.415
        #axA2.axvline( 1e3*2*np.pi*c0/w_ZDW, color='k', dashes=[2,2], linewidth=1)

        axA2.tick_params(axis='x', length=2., pad=2, top=False)

        lam_lim = (450,1350)
        lam_ticks = (500,700,900,1100,1300)
        axA2.set_xlim(lam_lim)
        axA2.set_xticks(lam_ticks)
        #axA2.set_xlim(w_lim)

        #v2w = lambda v: v*2*np.pi/1000.
        #axA2.set_xticks(v2w(v_ticks))
        #axA2.set_xticklabels( v_ticks  )
        #axA2.set_xticklabels( (1,2,r'$\omega_{\rm{Z}}$',3,4)  )
        #axA2.set_xlabel(r"Angular frequency $\omega~\mathrm{(rad/fs)}$")
        #axA2.set_xlabel(r"Frequency $\nu = (\Omega+\omega_0)/2 \pi~\mathrm{(THz)}$",labelpad=1)
        axA2.set_xlabel(r"Wavelength $\lambda = 2 \pi c/(\omega_0+\Omega)~\mathrm{(nm)}$",labelpad=1)
        #axA2.set_xlabel(r"Wavelength $\lambda~\mathrm{(nm)}$",labelpad=1)

        axA2.tick_params(axis='y', length=2., pad=2, top=False, labelleft=False)
        axA2.set_ylim(z_lim)
        axA2.set_yticks(z_ticks)

        #axA1.axhline(0.56, color='magenta', dashes=[2,2], lw=1)
        #axA2.axhline(0.56, color='magenta', dashes=[2,2], lw=1)
        #axA1.axhline(0.73, color='cyan', dashes=[2,2], lw=1)
        #axA2.axhline(0.73, color='cyan', dashes=[2,2], lw=1)

        # -- INSET AXES ----
        axA3 = axA1.inset_axes([0.45,0.025, 0.5,0.5])

        axA3.pcolorfast(t, z, _norm(Itz[:-1,:-1]),
                              vmin=0, vmax=3.,
                              cmap = my_cmap
                              )


        x_min, x_max = -0.065, 0.12
        y_min, y_max = 0.2, 1.8
        axA3.tick_params(axis='x', length=2., pad=1, top=False, bottom=False, labelbottom=False)
        axA3.set_xlim(x_min, x_max)

        axA3.tick_params(axis='y', length=2., pad=1, top=False, left=False, labelleft=False)
        axA3.set_ylim(y_min, y_max)

        my_col = 'white'
        box_x = [x_min, x_min, x_max, x_max, x_min]
        box_y = [y_min, y_max, y_max, y_min, y_min]
        axA1.plot(box_x, box_y, color=my_col, lw=0.75)

        axA3.spines['top'].set_color(my_col)
        axA3.spines['bottom'].set_color(my_col)
        axA3.spines['left'].set_color(my_col)
        axA3.spines['right'].set_color(my_col)

        f_name = 'res_spectrum_pyNLO_dz40micron_z14cm.dat'
        dat = np.loadtxt(f_name)
        lam_2 = dat[:,0]
        Iw_2 = dat[:,1]
        w_2 = 2*np.pi*0.29970/lam_2
        l2 = axB2.plot(lam_2, _dB(Iw_2/np.max(Iw_2)), color='lightgray', linewidth=1.5, label=r'B')

        #f_name = 'res_gnlse_Trevors_z14m.dat'
        #dat = np.loadtxt(f_name)
        #w_3 = dat[:,0]
        #Iw_3 = dat[:,1]
        #l3 = axB2.plot(w_3/1000, _dB(Iw_3/np.max(Iw_3)), color='lightgray', linewidth=1.5, label=r'B')

        z_id = np.argmin(np.abs(z-14))
        Iw = Iwz_s[z_id]
        #l1 = axB2.plot(lam[l_mask] , _dB(Iw[l_mask]/np.max(Iw)), color='k', linewidth=0.75, dashes=[2,2], label=r'A')
        l1 = axB2.plot(lam[l_mask] , _dB(Iw[l_mask]/np.max(Iw)), color='#1f77b4', linewidth=0.75, dashes=[2,2], label=r'A')

        set_legend(axB2, l1+l2, loc=(0.4,0.0), ncol=1)

        w_ZDW = 2.415
        #axB2.axvline( 1e3*2*np.pi*c0/w_ZDW, color='k', dashes=[2,2], linewidth=1)

        axB2.tick_params(axis='x', length=2., pad=2, top=False, labelbottom=False)
        axB2.set_xlim(lam_lim)
        #axB2.set_xlim(w_lim)
        #axB2.set_xticks(w_ticks)
        axB2.set_xticks(lam_ticks)
        #axB2.set_xticks(v2w(v_ticks))
        #axA2.set_xticklabels( v_ticks  )

        axB2.tick_params(axis='y', length=2., pad=2, top=False, labelleft=False, left=False, right=True, labelright=True)
        axB2.set_ylim((-90,5))
        axB2.set_yticks((-80,-40,0))
        axB2.set_ylabel('$|\mathcal{u}_\Omega|^2~\mathrm{(dB)}$')
        axB2.yaxis.set_label_position('right')

        self.set_subfig_label(axA1,'(c)',loc=1)
        self.set_subfig_label(axA2,'(d)',loc=1)
        self.set_subfig_label(axB1,'(a)',loc=1)
        self.set_subfig_label(axB2,'(b)',loc=1)


        # -- PHOTON NUMBER CONSERVATION ---------------------------------------
        ax = self.subfig_2

        l1 = ax.plot(z,(1-Cp/Cp[0])*1e8,lw=.75,color='#1f77b4',label=r"$C_{\rm{Ph}}$")

        ax.tick_params(axis='x', length=2., pad=2, top=False)
        ax.set_xlim(0,12)
        ax.set_xlabel(r"Propagation distance $z~\mathrm{(cm)}$")

        ax.tick_params(axis='y', length=2., pad=2, top=False)
        ax.set_ylim(-1,1)
        ax.set_ylabel(r"$1-C_{\rm{Ph}}(z)/C_{\rm{Ph}}(0)~\mathrm{(\times 10^{-8})}$")


        #ax.add_patch(plt.Rectangle(xy=(0.4,-0.15), width=0.75,height=0.2, fill=False, edgecolor='k', lw=1  ))

        #ax.add_patch(plt.Rectangle(xy=(7.,-0.53), width=0.75,height=0.2, fill=False, edgecolor='k', lw=1  ))

        #pos = ax.get_position()
        #self.fig.text(pos.x0+0.027, pos.y0+0.13, r"A" ,color='k', horizontalalignment='left', verticalalignment='top' )


        ax2 = ax.twinx()

        E = np.sum(np.abs(uwz)**2,axis=-1)
        l2 = ax2.plot(z,1-E/E[0], lw=0.75, dashes = [2,2],color='#1f77b4',label=r"$E$")

        ax2.tick_params(axis='y', length=2., pad=2, top=False)
        ax2.set_ylim(0,0.09)
        ax2.set_yticks((0,0.02,0.04,0.06,0.08))
        ax2.set_ylabel(r"$1-E(z)/E(0)$")

        set_legend(ax, l1+l2, loc=4, ncol=1)

        self.set_subfig_label(ax,'(e)',loc=1)

        axA1.yaxis.set_label_coords(-0.25,0.5)
        axB1.yaxis.set_label_coords(-0.25,0.5)
        ax.yaxis.set_label_coords(-0.09,0.5)



def fetch_data(file_path):
    dat = np.load(file_path)
    utz = dat['utz']
    t = dat['t']
    w = dat['w']
    w0 = dat['w0']
    z = dat['z']
    dz_int = dat['dz_integration']
    dz_a = dat['dz_a']
    Cp = dat['Cp']
    del_G = dat['del_G']
    return z,t,w,w0,utz, dz_a, Cp, del_G


def main():

    myFig = Figure(aspect_ratio=1.1, fig_basename='fig01', fig_format='png')
    myFig.set_layout()
    myFig.set_subfig_01()
    myFig.save()

if __name__ == '__main__':
    main()
