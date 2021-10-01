from matplotlib.colors import ListedColormap
import numpy as np
import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt
import string


all_plot = False
save = True
fig_type = ".png"

here = pathlib.Path(__file__).resolve().parent
data_dir = here.parent / "data"
p = "ZYZ543.wt5"
root = wt.open(data_dir / p)
shg_pol = wt.open(data_dir / "polarization.wt5")


if False:
    wt.artists.apply_rcparams(kind="publication")


# --- definitions ---------------------------------------------------------------------------------


alph = list(string.ascii_lowercase)

c_Mo = "m"
c_W = "c"
c_S = "y"
c_SiO2 = "#648BDF"
c_Si = "#5764BA"
txt_fs = 12
measurearrowprops = {"width": .2, "headwidth": 3, "headlength": 3, "color": "k"}

label_wm = "$\\mathsf{\\hbar\omega_{m}\\,\\left(eV \\right)}$"
label_Ishg = "$\\mathsf{norm. \\, I_{SHG}}$"
label_fontsize = 18


# --- PL map summary -----------------------------------------------------------------------------


if all_plot:
    import pl_summary
    pl_summary.run(save)


# --- big PL fig slices and moment-----------------------------------------------------------------


if all_plot:
    p = here.parent / "fitting" / "fit_PL.wt5"
    d = root.PL.proc_PL.copy(verbose=False)
    norm_PL = "$\\mathsf{norm. \\,\ PL \\,\ int.}$"
    nrows = 1
    vals = [1.85, 1.90, 1.93, 1.95, 1.97]
    cols = [1] * len(vals) + ["cbar"]

    fig, gs = wt.artists.create_figure(cols=cols, width="double")
    axs_maps = [plt.subplot(gs[i]) for i in range(len(vals))]
    cax = plt.subplot(gs[-1])
    for ax, txt in zip(axs_maps, alph):
        wt.artists.corner_text(
            txt,
            ax=ax,
            distance=.03,
            fontsize=14,
            background_alpha=1,
            bbox=True,
            factor=200,
        )

    # plot specific colors of PL
    for i in range(5):
        ax = axs_maps[i]
        dx = d.chop("x", "y", at={"energy": [vals[i], "eV"]}, verbose=False)[0]
        ax.pcolor(dx, vmax=d.intensity.max())
        dx.constants[0].format_spec = ".2f"
        dx.round_spec = -1
        ax.set_title(dx.constants[0].label, fontsize=12)
    # cax plotting and labeling
    ticks = np.linspace(0, 1, 6)
    wt.artists.plot_colorbar(cax=cax, cmap="default", label=norm_PL, ticks=ticks)
    # map labels
    xlabel = "$\\mathsf{x\\,\\left(\\mu m\\right)}$"
    ylabel = "$\\mathsf{y\\,\\left(\\mu m\\right)}$"
    ticks = [-40, -20, 0, 20, 40]
    wt.artists.set_fig_labels(
        xlabel=xlabel, ylabel=ylabel, col=slice(0, 4, 1), xticks=ticks, yticks=ticks
    )
    # save
    if save:
        p = "PL_mapping_slices" + fig_type
        p = here / p
        wt.artists.savefig(p)
    else:
        plt.show()


# --- junction reflection contrast (100x) ---------------------------------------------------------


if all_plot:
    import rc_vs_y_100x
    rc_vs_y_100x.run(save)


# --- compare PL and Raman maps -------------------------------------------------------------------


if all_plot:
    pl = root.PL.proc_PL.copy()
    pl.smooth((5, 0, 0))
    pl_color = pl.intensity_energy_moment_1[0]
    pl_color[pl_color < 1.81] = 1.81

    raman = root.raman.proc_raman
    raman1 = raman.chop("x", "y", at={"energy": [416, "wn"]})[0]
    raman2 = raman.chop("x", "y", at={"energy": [350, "wn"]})[0]

    fig, gs = wt.artists.create_figure(width="dissertation", cols=[1, 1, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    plt.yticks(visible=False)
    ax2 = plt.subplot(gs[2])
    plt.yticks(visible=False)

    ax1.set_title(r"Raman, WS$_2$ 2LA(M)")
    ax0.set_title(r"Raman, WS$_2$ A$_1$($\mathsf{\Gamma}$)")
    ax2.set_title(r"PL $\langle \hbar \omega \rangle$")

    ax0.pcolormesh(raman1, channel="leveled", cmap="magma")
    ax1.pcolormesh(raman2, channel="leveled", cmap="magma")
    ax2.pcolormesh(pl.axes[0].points, pl.axes[1].points, pl_color.T, cmap="rainbow_r")

    for axi in [ax0, ax1, ax2]:
        axi.grid()
        axi.set_ylim(-25, 25)
        axi.set_xlim(-25, 25)
    wt.artists.set_ax_labels(ax0, xlabel=r"$x \ (\mu\mathsf{m})$", ylabel=r"$y \ (\mu\mathsf{m})$")
    wt.artists.set_ax_labels(ax1, xlabel=r"$x \ (\mu\mathsf{m})$")
    wt.artists.set_ax_labels(ax2, xlabel=r"$x \ (\mu\mathsf{m})$")

    # save
    if save:
        p = "raman_pl_comparison" + fig_type
        p = here / p
        wt.artists.savefig(p)
    else:
        plt.show()


# --- representative Raman spec vs y --------------------------------------------------------------


if all_plot:
    import raman_vs_y
    raman_vs_y.run(save)

# --- maps of raman modes -------------------------------------------------------------------------


if all_plot:
    d = root.raman.proc_raman

    # define masks for MoS2, WS2, and Si
    d_temp = d.chop("x", "y", at={"energy": [383, "wn"]}, verbose=False)[0]
    d_temp.intensity.normalize()
    separator = d_temp.intensity[:]
    MX2_area = separator > 0.2
    WS2_area = separator > 0.36
    MoS2_area = MX2_area & ~WS2_area

    # frequencies picked by hand to represent peak transitions
    WS2_freqs= [174, 198, 211, 350, 378, 397, 416]
    MoS2_freqs = [383, 404, 408]
    Si_freqs = [520]

    row_names = [r"WS$_2$", r"MoS$_2$", "Si"]

    fig, gs = wt.artists.create_figure(
        width="dissertation", nrows=3, cols=[1] * len(WS2_freqs) + ["cbar"], hspace=0.5
    )
    for row, (freqs, mask) in enumerate(
        zip(
            [WS2_freqs, MoS2_freqs, Si_freqs],
            [None, WS2_area, None]  # MoS2 masks WS2 for better contrast
    )):
        for col, freq in enumerate(freqs):
            ax = plt.subplot(gs[row, col])
            if col==0:
                ax.set_ylabel(row_names[row])
            chop = d.chop("x", "y", at={"energy": [freq, "wn"]}, verbose=False)[0]
            z = chop.leveled[:]
            if mask is not None:
                z[mask] = 0
            ax.pcolormesh(chop.x.points, chop.y.points, z.T, cmap="magma", vmin=0)
            ax.set_title(str(freq) + " wn")
            plt.yticks(visible=False)
            plt.xticks(visible=False)
    cax = plt.subplot(gs[:, -1])
    wt.artists.plot_colorbar(cax, cmap="magma", label="Raman intensity (norm.)")

    if save:
        p = f"raman_mode_maps" + fig_type
        p = here / p
        wt.artists.savefig(p)
    else:
        plt.show()


# --- energies vs distance from junction ----------------------------------------------------------


if False:
    pl = root.PL.proc_PL
    arr = pl.intensity[:]
    arr_collapse = np.mean(arr, axis=0)

    mask = np.ones(arr_collapse.shape)
    mask[arr_collapse < arr_collapse.max() * 0.04] = 0
    nanmask = mask.copy()
    nanmask[nanmask == 0] = np.nan
    pl.intensity_energy_moment_1 *= nanmask

    crossover = pl.chop("energy", "y", at={"x":[0, 'um']})[0]

    wt5 = data_dir / "reflection microspectroscopy" / "reflection.wt5"    
    reflection = wt.open(wt5)
    spectrum = reflection.spectrum

    # spectrum.print_tree()
    # crossover.print_tree()
    if True:
        fig, gs = wt.artists.create_figure(cols=[1])
        ax = plt.subplot(gs[0])
        # sin terms apply to the angle of the slice taken
        ax.plot(spectrum.y[0,:] * np.sin(np.pi/6), spectrum.ares[0,:], label="A transition")
        ax.plot(spectrum.y[0,:] * np.sin(np.pi/6), spectrum.bres[0,:], label="B transition")
        ax.plot(
            -crossover.y[0,:] + 11,
            crossover.intensity_energy_moment_1[0,:],
            label=r"PL $\langle \hbar \omega \rangle$")
        ax.grid()
        ax.legend()
        ax.set_ylabel(r"Energy (eV)")
        ax.set_xlabel(r"distance from junction normal ($\mu$m)")
        plt.show()


# --- SHG polarization diagnostic image -----------------------------------------------------------


if False:
    verbose=False

    def raw_subsection_plot(ylims, select_angles, aspect=1):
        slit = shg_pol.imgs.slit.split("yindex", ylims, verbose=verbose)[1]
        lamp = shg_pol.imgs.lamp.split("yindex", ylims, verbose=verbose)[1]
        pump = shg_pol.imgs.pump.split("yindex", ylims, verbose=verbose)[1]

        fig, gs = wt.artists.create_figure(
            width="double",
            cols=[1] + [1] * len(select_angles) + ["cbar"],
            default_aspect=aspect
        )

        y_lines = [866, 789, 680, 550, 432, 324]
        y_lines = [yi for yi in y_lines if (yi > ylims[0] and yi < ylims[1])]

        ax = plt.subplot(gs[0,0])
        ax.pcolormesh(lamp, cmap="gist_gray", vmax=shg_pol.imgs.lamp.signal.max() * 0.8, vmin=shg_pol.imgs.lamp.signal.max() * 0.68)
        ax.contour(pump, levels=np.array([pump.signal.max()/2]), colors=["magenta"], alphas=[0.5])
        ax.contour(slit, levels=np.array([slit.signal.max()/2]), colors=["r"], alphas=[0.5])
        plt.grid(color="k")
        plt.hlines(y=y_lines, xmin=lamp.xindex.min(), xmax=lamp.xindex.max(), colors="goldenrod")
        plt.vlines([1328], ymin=lamp.yindex.min(), ymax=lamp.yindex.max(), colors="goldenrod")

        vlim = [0, shg_pol.polarization.signal.max() * 0.5]
        for i, angle in enumerate(select_angles):
            d = shg_pol.polarization.chop("xindex", "yindex", at={"angle":[angle, None]}, verbose=verbose)[0]
            d = d.split("yindex", ylims, verbose=verbose)[1]
            axi = plt.subplot(gs[0, i+1], sharey=ax, sharex=ax)
            axi.set_title(r"${0}".format(angle) + r"^\circ$")
            axi.pcolormesh(d, vmin=vlim[0], vmax=vlim[1])
            # axi.plot(d.signal_xindex_moment_0[:], d.yindex[0, :])
            plt.hlines(y=[yi for yi in y_lines], xmin=lamp.xindex.min(), xmax=lamp.xindex.max(), colors="goldenrod")
            plt.vlines([1328], ymin=lamp.yindex.min(), ymax=lamp.yindex.max(), colors="goldenrod")
            plt.yticks(visible=False)
            plt.grid(b=True)

        cax = plt.subplot(gs[0,-1])
        wt.artists.plot_colorbar(cax, ticks=np.linspace(*vlim, 6), clim=vlim, label="cps")
        return fig

    fig1 = raw_subsection_plot(
        [200, 1300],
        np.arange(0, 360, 20)[:9],
        aspect=0.2 * shg_pol.polarization.shape[1] / shg_pol.polarization.shape[-1]
    )
    if save:
        wt.artists.savefig(here / "polarization_data_raw_view.png")

    fig2 = raw_subsection_plot(
        [324, 550],
        [60],
        aspect=1
    )
    if save:
        wt.artists.savefig(here / "low_junction_raw_view.png")

    fig3 = raw_subsection_plot(
        [775, 925],
        [60],
        aspect=1
    )
    if save:
        wt.artists.savefig(here / "high_junction_raw_view.png")


# --- shg polarization summary image - SI ---------------------------------------------------------


if all_plot:
    verbose=False
    from scipy.interpolate import interp1d
    um_per_pixel = 0.3

    def pixel_to_um(var):
        out = var * um_per_pixel
        out -= out.mean()
        return out

    def raw_angle_to_physical(deg):
        """ 
        Relate recorded angle to the image axes.
        Raw number is off by an offset
        0 deg corresponds to H-pol in the lab frame, which is V-pol for our image in this figure
        """
        offset = 49.
        return deg - offset

    def get_power_function(angle, power):
        """power interpolation for angle
        """
        x = list(angle)
        y = list(power)
        x.append(360)
        y.append(y[0])
        x = np.array(x)
        f = interp1d(x, np.array(y))
        return f

    def get_bs_transmission_function():
        """plate beamsplitter corrections
        based on transmission measurements for both H-pol and V-pol
        """
        x = np.linspace(0, 360)
        y = (0.5 * np.sin(x * np.pi / 180))**2 + (.83 * np.cos(x * np.pi / 180))**2
        f = interp1d(x, np.array(y))
        return f

    ylims = [200, 950]
    slit = shg_pol.imgs.slit.split("yindex", ylims, verbose=verbose)[1]
    lamp = shg_pol.imgs.lamp.split("yindex", ylims, verbose=verbose)[1]
    pump = shg_pol.imgs.pump.split("yindex", ylims, verbose=verbose)[1]

    f1 = get_power_function(shg_pol.pump_power.raw_angle[:], shg_pol.pump_power.power[:])
    f2 = get_bs_transmission_function()
    slit.signal.normalize()
    for data in [slit, lamp, pump]:
        data.transform("yindex", "xindex")

    shg_pol.polarization.transform("yindex", "angle")

    temp = shg_pol.polarization.split("yindex", ylims)[1]
    # pump.create_channel(name="signal_square", values=pump.signal[:]**2)
    # pump.moment("xindex", channel="signal_square", moment=0)

    temp.signal_xindex_moment_0[:] /= pump.signal_xindex_moment_0[0, 0, :][None, :, None]
    temp.signal_xindex_moment_0[:] /= f1(shg_pol.polarization.angle[:])**2
    temp.signal_xindex_moment_0[:] /= f2(
        raw_angle_to_physical(shg_pol.polarization.angle[:]) % 360
    )
    temp.signal_xindex_moment_0.normalize()

    for d in [temp, lamp, pump]:
        d.create_variable("xdistance", values=pixel_to_um(d["xindex"][:]), units="um")
        d.create_variable("ydistance", values=pixel_to_um(d["yindex"][:]), units="um")
        d.transform("ydistance", "xdistance")


    fig, gs = wt.artists.create_figure(
        width="dissertation",
        nrows = 4,
        cols = [1, "cbar"],
        default_aspect=1/6.
    )

    ax1 = plt.subplot(gs[0, 0])
    plt.xticks(visible=False)
    ax2 = plt.subplot(gs[1:, 0], sharex=ax1)

    ax1.pcolormesh(lamp, cmap="gist_gray", vmax=shg_pol.imgs.lamp.signal.max() * 0.8, vmin=shg_pol.imgs.lamp.signal.max() * 0.68)
    pump_cm = ListedColormap(np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0.2]
    ]))

    pump.signal[:] /= pump.signal_xindex_moment_0[:]
    pump.signal.normalize()
    ax1.contourf(
        pump,
        levels=np.array([0.5, 1]), cmap=pump_cm
    )
    # ax1.contourf(
    #     slit.yindex[0, :], slit.xindex[:, 0], slit.signal[:],
    #     levels=np.array([0.5, 1]), cmap=slit_cm
    # )

    angle = temp["angle"]
    angle[:] = raw_angle_to_physical(angle[:])
    temp.transform("ydistance", "angle")
    ax2.pcolormesh(temp, channel="signal_xindex_moment_0")
    ax2.set_yticks(np.linspace(-30, 270, 6))
    for ax in [ax1, ax2]:
        ax.grid(color="k", linestyle=":")

    # ax1.set_xlim(200, 950)
    wt.artists.set_ax_labels(ax=ax2, xlabel=r"$x \ \left(\mu \mathsf{m}\right)$", ylabel=r"$\theta_{\mathsf{out}} = \theta_{\mathsf{in}} \ (\mathsf{deg})$")
    wt.artists.set_ax_labels(ax=ax1, ylabel=r"$y \ \left(\mu \mathsf{m}\right)$")
    cax = plt.subplot(gs[1:,1])
    wt.artists.plot_colorbar(cax=cax, label="SHG Intensity (a.u.)")
    wt.artists.corner_text("a", ax=ax1)
    wt.artists.corner_text("b", ax=ax2)

    if save:
        wt.artists.savefig(here / "polarization_summary.SI.png")
    else:
        plt.show()


# --- shg polarization summary image - manuscript -------------------------------------------------


if all_plot:
    verbose=False
    from scipy.interpolate import interp1d
    um_per_pixel = 0.3  # 20x

    def raw_angle_to_physical(deg):
        """ 
        Relate recorded angle to the image axes.
        Raw number is off by an offset
        0 deg corresponds to H-pol in the lab frame, which is V-pol for our image in this figure
        """
        offset = 49.
        return deg - offset


    # ylims = [200, 950]
    ylims = [750, 950]
    slit = shg_pol.imgs.slit.split("yindex", ylims, verbose=verbose)[1]
    lamp = shg_pol.imgs.lamp.split("yindex", ylims, verbose=verbose)[1]
    pump = shg_pol.imgs.pump.split("yindex", ylims, verbose=verbose)[1]

    def pixel_to_um(var):
        out = var * um_per_pixel
        out -= out.mean()
        return out

    def get_power_function(angle, power):
        """power interpolation for angle
        """
        x = list(angle)
        y = list(power)
        x.append(360)
        y.append(y[0])
        x = np.array(x)
        f = interp1d(x, np.array(y))
        return f

    def get_bs_transmission_function():
        """plate beamsplitter corrections
        based on transmission measurements for both H-pol and V-pol
        """
        x = np.linspace(0, 360)
        y = (0.5 * np.sin(x * np.pi / 180))**2 + (.83 * np.cos(x * np.pi / 180))**2
        f = interp1d(x, np.array(y))
        return f

    f1 = get_power_function(shg_pol.pump_power.raw_angle[:], shg_pol.pump_power.power[:])
    f2 = get_bs_transmission_function()
    slit.signal.normalize()
    for data in [slit, lamp, pump]:
        data.transform("yindex", "xindex")
    
    shg_pol.polarization.transform("yindex", "angle")

    temp = shg_pol.polarization.split("yindex", ylims)[1]
    # pump.create_channel(name="signal_square", values=pump.signal[:]**2)
    # pump.moment("xindex", channel="signal_square", moment=0)

    temp.signal_xindex_moment_0[:] /= pump.signal_xindex_moment_0[0, 0, :][None, :, None]
    temp.signal_xindex_moment_0[:] /= f1(shg_pol.polarization.angle[:])**2
    temp.signal_xindex_moment_0[:] /= f2(raw_angle_to_physical(shg_pol.polarization.angle[:])%360)
    temp.signal_xindex_moment_0.normalize()
    temp.moment("angle", channel="signal_xindex_moment_0", moment=0)  # signal at all angles

    for d in [temp, lamp, pump]:
        d.create_variable("xdistance", values=pixel_to_um(d["xindex"][:]), units="um")
        d.create_variable("ydistance", values=pixel_to_um(d["yindex"][:]), units="um")
        d.transform("ydistance", "xdistance")

    fig, gs = wt.artists.create_figure(
        width=6.66,
        nrows = 2,
        cols = [1],
        default_aspect = 100 / (ylims[1] - ylims[0]),
        margin=[0.8, 0.15, 0.8, 0.8]
    )

    ax1 = plt.subplot(gs[0, 0])
    plt.xticks(visible=False)
    ax2 = plt.subplot(gs[1, 0], sharex=ax1)

    ax1.pcolormesh(
        lamp, cmap="gist_gray", vmax=shg_pol.imgs.lamp.signal.max() * 0.8, vmin=shg_pol.imgs.lamp.signal.max() * 0.68
    )
    pump_cm = ListedColormap(np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0.2]
    ]))

    pump.signal[:] /= pump.signal_xindex_moment_0[:]
    pump.signal.normalize()
    ax1.contourf(
        pump.ydistance[0, :], pump.xdistance[:, 0], pump.signal[:],
        levels=np.array([0.5, 1]), cmap=pump_cm
    )
    ax1.text(0.38, 0.65, r"MoS$_2$", transform=ax1.transAxes, fontsize=16)
    ax1.text(0.7, 0.23, r"WS$_2$", transform=ax1.transAxes, fontsize=16)

    angle = temp["angle"]
    angle[:] = raw_angle_to_physical(angle[:])
    # temp.print_tree()
    if True:
        yi = temp.signal_xindex_moment_0[0, :, 6]
        yi /= yi.max()
        ax2.plot(temp.ydistance.points, yi, linewidth=3, color="k")
        ax2.set_ylabel("SHG intensity (a.u.)", fontsize=18)
        # ax2.plot(temp, channel="signal_xindex_moment_0_angle_moment_0")
    else:
        temp.transform("ydistance", "angle")
        ax2.pcolormesh(temp, channel="signal_xindex_moment_0")
        ax2.set_ylabel(r"$\theta_{\mathsf{out}} = \theta_{\mathsf{in}} \ (\mathsf{deg})$")
    # ax2.set_yticks(np.linspace(-30, 270, 6))
    for ax in [ax1, ax2]:
        ax.grid(color="k", linestyle=":")

    # ax1.set_xlim(*ylims)
    ax2.set_xlabel(r"$x \ (\mathsf{\mu m})$", fontsize=18)
    ax1.set_ylabel(r"$y \ (\mathsf{\mu m})$", fontsize=18)
    # cax = plt.subplot(gs[1:,1])
    # wt.artists.plot_colorbar(cax=cax, label="SHG Intensity (a.u.)")
    wt.artists.corner_text("a", ax=ax1)
    wt.artists.corner_text("b", ax=ax2)

    if save:
        wt.artists.savefig(here / "polarization_summary.main.png")
    else:
        plt.show()


# --- shg polar plots ------------------------------------------------------------------------------


if False:
    from scipy.interpolate import interp1d
    center_x = 1340

    lower_WS2_y = [450, 475]
    lower_junction_y = [428, 432]
    lower_MoS2_y = [420, 425]

    upper_WS2_y = [880, 900]
    upper_junction_y = [860, 868]
    upper_MoS2_y = [840, 860]


    def get_power_function(angle, power):
        x = list(angle * np.pi / 180)
        y = list(power)
        x.append(2 * np.pi)
        y.append(y[0])
        x = np.array(x)
        f = interp1d(x, np.array(y))
        return f


    pump_power = wt.open(os.path.join(__here__, "pump_power.wt5"))
    f = get_power_function(pump_power.raw_angle[:], pump_power.power[:])

    for junction_name, WS2_y, junction_y, MoS2_y in [
        ["Upper", upper_WS2_y, upper_junction_y, upper_MoS2_y],
        ["Lower", lower_WS2_y, lower_junction_y, lower_MoS2_y]
    ]:

        WS2 = pol.split("xindex", [center_x-3, center_x+3], verbose=verbose)[1].split("yindex", WS2_y, verbose=verbose)[1]
        MoS2 = pol.split("xindex", [center_x-3, center_x+3], verbose=verbose)[1].split("yindex", MoS2_y, verbose=verbose)[1]
        junction = pol.split("xindex", [center_x-3, center_x+3], verbose=verbose)[1].split("yindex", junction_y, verbose=verbose)[1]
        # pump_power.print_tree()

        fig, gs = wt.artists.create_figure()
        ax = plt.subplot(gs[0], projection="polar")
        ax.title("{0} Junction".format(junction_name))
        offset = 49 * np.pi / 180
        for d, name in zip([WS2, MoS2, junction], ["WS2", "MoS2", "junction"]):
            x = d.angle.points * np.pi / 180
            y = d.signal[:].mean(axis=0).mean(axis=0)
            plt.plot(-(x - offset), y / f(x)**2, label=name)
        # plt.plot(pump_power.raw_angle[:] * np.pi / 180, pump_power.power[:])
        ax.legend()
        wt.artists.savefig(os.path.join(__here__, "{0}_junction.png".format(junction_name.lower())))
        # plt.show()


# --- reflection contrast 20x and comparison with NA -----------------------------------------------


if all_plot:
    import rc_20x_rc_vs_na as rc
    rc.run(save)


# --- reflection contrast vs SiO2 thickness --------------------------------------------------------


if all_plot:
    import rc_vs_SiO2_thickness as mod
    mod.run(save)


# --- raman SI figure ------------------------------------------------------------------------------


if all_plot:
    import raman_si as mod
    mod.run(save)

