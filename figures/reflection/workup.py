import WrightTools as wt
import matplotlib.pyplot as plt
import numpy as np
import os
import fit_reflection as fr
import tmm_lib as lib


__here__ = os.path.abspath(os.path.dirname(__file__))

moA = 1.845 # eV
moB = 2.040
wA = 1.935

# slit parameters
mpix = 1333 #middle pixel
deltax = 4
lpix_im = mpix-deltax
rpix_im = mpix+deltax

# import data, save to wt5
if False:
    root = wt.Collection(name='root')
    sample = wt.Collection(name='sample', parent=root)
    
    temp = wt.data.from_Solis(os.path.join(__here__, 'lamp_image.asc'))
    temp.copy(parent=sample, name='optical')
    temp = wt.data.from_Solis(os.path.join(__here__, '620nm_cwl_50um_slit.asc'))
    temp.copy(parent=sample, name='spectrum')
    root.print_tree()
    # root.save(os.path.join(__here__, 'data.wt5'))

if True:  # ddk: creating spatial map similar to manuscript
    root = wt.open("data.wt5")
    root.print_tree()

    lamp = root.sample.optical.split(
        "yindex", [850, 1300]
    )[1].split("xindex", [1050, 1500])[1]
    lamp.create_variable(
        "ydist", values=(lamp.yindex[:] - lamp.yindex[:].mean()) * 0.3, units="um"
    )
    lamp.create_variable(
        "xdist", values=(lamp.xindex[:] - mpix) * 0.3, units="um"
    )
    lamp.transform("xdist", "ydist")


    data = root.sample.spectrum.split(
        "yindex", [850, 1300]
    )[1]
    data.smooth((10,0))
    data.create_variable(
        "energy", values=np.linspace(2.57, 1.45, data.wm[:].size).reshape(data.wm.shape), units="eV"
    )
    data.create_variable(
        "ydist", values=(data.yindex[:] - data.yindex[:].mean()) * 0.3, units="um"
    )

    substrate_low = data.split("yindex", [900])[0].signal[:].mean(axis=1)[:, None]
    substrate_high = data.split("yindex", [1250])[1].signal[:].mean(axis=1)[:, None]
    # interpolate between top and bottom substrate spectra 
    z = data.yindex[:].copy()
    s = (z - 900) / 350
    s[s>1] = 1
    s[s<0] = 0
    substrate = (1-s) * substrate_low + s * substrate_high

    data.create_channel(
        "contrast", values=(data.signal[:] - substrate) / substrate, signed=True
    )
    data.transform("energy", "ydist")
    data.contrast.clip(-0.35, 0.35)

    # fig 1: 20x results
    fig, gs = wt.artists.create_figure(width="dissertation", nrows=1, cols=[1,1,"cbar"])

    ax0 = plt.subplot(gs[0, 0])
    ax0.set_title("image (20x) \n ")
    ax0.pcolormesh(lamp, cmap="gist_gray")
    ax0.grid()

    ax1 = plt.subplot(gs[0, 1])
    ax1.set_title("reflection contrast \n (y=0)")
    ax1.pcolormesh(data, channel="contrast")
    plt.yticks(visible=False)
    ax1.grid()

    cax = plt.subplot(gs[0, 2])
    wt.artists.plot_colorbar(
        cax,
        cmap="signed",
        ticks=np.linspace(-0.3, 0.3, 7),
        clim=[-0.35, 0.35],
        label=r"$(R-R_0) / R_0$"
    )
    
    wt.artists.set_ax_labels(ax1, xlabel=r"$\hbar\omega \ \left(\mathsf{eV}\right)$")
    wt.artists.set_ax_labels(ax0, xlabel=r"$\mathsf{x \ position \ (\mu m)}$", ylabel=r"$\mathsf{y \ position \ (\mu m)}$")
    wt.artists.corner_text("a", ax=ax0)
    wt.artists.corner_text("b", ax=ax1)
    wt.artists.savefig(__here__ + "/contrast.1.png")

    # fig 2: comparison of different NAs
    fig, gs = wt.artists.create_figure(width="dissertation", nrows=1, cols=[1, 1, 0.3])

    ax2 = plt.subplot(gs[0, 0])
    ax3 = plt.subplot(gs[0, 1], sharey=ax2)
    plt.yticks(visible=False)
    ax2.set_title(r"MoS$_2$")
    ax3.set_title(r"WS$_2$")
    ax2.grid()
    ax3.grid()
    wt.artists.set_ax_labels(
        ax2,
        xlabel=r"$\hbar\omega \ \left(\mathsf{eV}\right)$",
        ylabel=r"$(R-R_0) / R_0$",
    )
    wt.artists.set_ax_labels(
        ax3,
        xlabel=r"$\hbar\omega \ \left(\mathsf{eV}\right)$",
    )

    # ax2.set_title("Comparison with Fresnel Effects")
    y_ws2 = data.split("ydist", [15, 20])[1].contrast[:].mean(axis=1)
    y_mos2 = data.split("ydist", [31, 35])[1].contrast[:].mean(axis=1)
    x = data.energy.points[:]

    d2 = 298e-7

    if True:  # apply offsets to MX2 optical constants
        def nmos2(hw):
            return fr.n_mos2_ml(hw + 0.04)
        def nws2(hw):
            return fr.n_ws2_ml(hw + 0.07)
    else:
        nmos2 = fr.n_mos2_ml
        nws2 = fr.n_ws2_ml

    # MoS2
    blank = lib.FilmStack([fr.n_air, fr.n_fused_silica, fr.n_Si], [d2])
    sample = lib.FilmStack([fr.n_air, nmos2, fr.n_fused_silica, fr.n_Si], [fr.d_mono, d2])
    r_blank = blank.RT(fr.hw)[0]
    r_sample = sample.RT(fr.hw)[0]
    contrast = (r_sample - r_blank) / r_blank
    ax2.plot(data.energy.points[:], y_mos2, label=r"MoS$_2$", color="k")
    ax2.plot(fr.hw, contrast, linewidth=4, alpha=0.5, color="k", label=r"MoS$_2$ (theoretical)")

    # WS2
    blank = lib.FilmStack([fr.n_air, fr.n_fused_silica, fr.n_Si], [d2])
    sample = lib.FilmStack([fr.n_air, nws2, fr.n_fused_silica, fr.n_Si], [fr.d_mono, d2])
    r_blank = blank.RT(fr.hw)[0]
    r_sample = sample.RT(fr.hw)[0]
    contrast = (r_sample - r_blank) / r_blank
    ax3.plot(data.energy.points[:], y_ws2, label=r"WS$_2$", color="k")
    ax3.plot(fr.hw, contrast, linewidth=4, alpha=0.5, color="k", label=r"WS$_2$ (theoretical)")

    x100 = wt.open(os.path.join(__here__, "reflection.wt5"))
    rc2 = x100.spectrum
    y_ws2_100x = rc2.split("y", [15, 20])[1].contrast[:].mean(axis=1)
    y_mos2_100x = rc2.split("y", [-20, -15])[1].contrast[:].mean(axis=1)
    ax2.plot(rc2.wm.points[:], y_mos2_100x, label=r"MoS$_2$ (100x)", color="k", linestyle=":")
    ax3.plot(rc2.wm.points[:], y_ws2_100x, label=r"WS$_2$ (100x)", color="k", linestyle=":")

    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color="k", ls=":"),
        Line2D([0], [0], color="k"),
        Line2D([0], [0], color="k", lw=4, alpha=0.5),
    ]

    ax3.legend(custom_lines, ['NA 0.95', 'NA 0.46', 'NA ~ 0 \n (theory)'], fontsize=16, loc=[1.1, 0.5])
    wt.artists.corner_text("a", ax=ax2, fontsize=12)
    wt.artists.corner_text("b", ax=ax3, fontsize=12)
    wt.artists.savefig(__here__ + "/contrast.2.png")


if False:  # overview figure
    data = wt.open('data.wt5')
    opt = data.sample[0] # optical image
    spectrum = data.sample[1]
    spectrum.smooth([9,1], channel='signal', verbose=True)
    E = np.linspace (1.45, 2.57, spectrum.wm.points.size) # for center photon energy of 2 eV
    
    fig, gs = wt.artists.create_figure(nrows=2, cols=[1]*3 + ['cbar'],
                                       width='double', wspace=0.6, hspace=0.4)
    ax0 = plt.subplot(gs[0,0])
    cmap = plt.get_cmap('Greys_r')
    
    ax0.pcolormesh(opt.xindex.points, opt.yindex.points, opt.signal[:].T, cmap=cmap)
    ax0.set_title('full image')
    ax0.set_xlabel('x position')
    ax0.set_ylabel('y position')
    
    opt = opt.split('xindex', [1070, 1497], verbose=False)[1]
    opt.xindex.units = opt.yindex.units = None
    opt = opt.split('yindex', [868, 1320], verbose=False)[1]
    
    ax1 = plt.subplot(gs[0,1])
    ax1.pcolormesh(opt.xindex.points, opt.yindex.points, opt.signal[:].T, cmap=cmap)
    ax1.axvspan(xmin=lpix_im, xmax=rpix_im, ymin=0, ymax=1, color='k', alpha=0.2)
    ax1.set_title('Structure of interest')
    ax1.set_xlabel('x position')
    
    spec = spectrum.split('yindex', [868, 1320], verbose=False)[1]
    ax2 = plt.subplot(gs[0,2])
    ax2.pcolormesh(E, spec.yindex.points, np.flip(spec.signal[:].T,axis=1), cmap='default')
    ax2.axvline(x=moA, ymin=0, ymax=1, c='r', lw=2.4, alpha=0.6, label='Mo-A')
    ax2.axvline(x=moB, ymin=0, ymax=1, c='b', lw=2.4, alpha=0.6, label='W-A')
    ax2.axvline(x=wA, ymin=0, ymax=1, c='orange', lw=2.4, alpha=0.6, label='Mo-B')
    ax2.set_title('reflection image')
    ax2.legend(loc='upper right')
    cax = plt.subplot(gs[:,-1])
    wt.artists.plot_colorbar(cax=cax, cmap='default',
                             ticks=np.linspace(0, spec.signal[:].max(), 11))
    
    ax3 = plt.subplot(gs[1,0])
    scan_cmap = plt.get_cmap('rainbow')
    n_slices = 11
    color_inds = np.linspace(0, 1, n_slices)
    ymin=965
    ymax=1200
    yinds = np.linspace(ymin, ymax, n_slices)
    baseline = spectrum.split('yindex', [750, 850], verbose=False)[1]
    substrate = baseline.signal[:].mean(axis=1)
    offset = 0
    for i in range(n_slices):
        slyce = spectrum.split('yindex', [yinds[i], yinds[i]+1], verbose=False)[1]
        contrast = (slyce.signal[:] - substrate) / (slyce.signal[:] + substrate)
        ax3.plot(E, np.flip(contrast + offset), c=scan_cmap(color_inds[i]), lw=1.8)
        ax1.plot([lpix_im, rpix_im], [yinds[i], yinds[i]], c=scan_cmap(color_inds[i]))
        offset += 0.1
    ax2.axvline(x=wA, ymin=0, ymax=1, c='orange', lw=2.4, alpha=0.6)
    ax3.set_xlabel('Photon energy (eV)')
    ax3.set_ylabel('Reflection contrast')
    ax3.grid()
    
    if False:
        wt.artists.savefig(os.path.join(__here__, 'refl_contrast.png'))