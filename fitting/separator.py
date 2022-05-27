"""accurately define the MoS2 WS2, and junction domains using Raman features
potential features to consider for mapping:
- A1g intensity
- E' intensity
- for WS2, nonres background (eg 100-125 wn)
"""


import numpy as np
import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib import cm


ref_cmap = cm.get_cmap("turbo_r")
colors = ref_cmap(np.linspace(0, 1, 4))

here = pathlib.Path(__file__).resolve().parent
root = wt.open(here.parent / "data" / "data.wt5")

def main(save=True):
    raman = root.raman.proc
    raman.smooth((2,0,0))
    raman.level("leveled", 0, 30)
    pl = root.pl.proc
    screen = wt.Data(name="screen")
    screen.create_variable(name="x", values=pl.x[0])
    screen.create_variable(name="y", values=pl.y[0])
    screen.transform("x", "y")

    separators = {
        "MoS2_E2g": [381.7, 385.7],
        "MoS2_A1g": [403.7, 409.7],
        "WS2_A1g": [415, 418],
        "WS2_nr": [540, 550],
        "WS2_2LA": [349, 352],
        "WS2_LA": [170, 178],
    }

    for name, window in separators.items():
        d_temp = raman.split("energy", window, verbose=False)[1]
        d_temp.moment(channel="leveled", axis="energy", moment=0)
        d_temp.leveled_energy_moment_0.normalize()
        screen.create_channel(name, values=d_temp.leveled_energy_moment_0[0])

    substrate = (screen["MoS2_E2g"][:] < 0.06) & (screen["MoS2_A1g"][:] < 0.06)
    mos2 = np.logical_not(substrate) \
        & (screen["MoS2_E2g"][:] < 0.45) \
        & (screen["MoS2_A1g"][:] < 0.55) \
        & (screen["WS2_A1g"][:] < 0.12)  # 0.17
    # mos2 = np.logical_not(substrate) & \
    #     (screen["MoS2_E2g"][:] < 0.25) & (screen["MoS2_A1g"][:] < 0.41) \
    #     & (screen["WS2_A1g"][:] < 0.18)
    # mos2_edge = mos2 & ((screen["MoS2_E2g"][:] < 0.137) | (screen["MoS2_A1g"][:] < 0.27))
    mos2_edge = mos2 & (screen["MoS2_A1g"][:] < 0.36)
    mos2_core = mos2 & np.logical_not(mos2_edge)
    junction = np.logical_not(substrate + mos2) & (screen["MoS2_E2g"][:] < 0.73) & (screen["MoS2_A1g"][:] < 0.56)
    # junction = np.logical_not(substrate + mos2) & (screen["WS2_2LA"][:] < 0.6) & (screen["WS2_A1g"][:] < 0.6)
    # ws2_core = np.logical_not(junction + substrate + mos2)
    ws2 = np.logical_not(substrate + mos2 + junction)
    junctiona = junction & (screen["WS2_2LA"][:] < 0.6)
    junctionb = junction & np.logical_not(junctiona) # & (screen["MoS2_A1g"][:] < 0.4)

    fig, gs = wt.artists.create_figure(
        width="double", cols=[1, 1], wspace=0.75, hspace=0.75, nrows=2,
    )

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    for ax, separators, lim in zip(
        [ax0, ax1],
        [["MoS2_A1g", "MoS2_E2g"],
        ["WS2_A1g", "WS2_2LA"]],
        [140, 40]
    ):
        separator1 = screen[separators[0]]
        separator2 = screen[separators[1]]

        axx = wt.artists.add_sideplot(ax, "x", pad=0.05)
        axx.hist(separator1[:].flatten(), bins=100, color="k")
        axy = wt.artists.add_sideplot(ax, along="y", pad=0.05)
        axy.hist(separator2[:].flatten(), bins=100, color="k", orientation="horizontal")
        axx.set_ylim(None, lim)
        axy.set_xlim(None, lim)
        axx.set_facecolor("gray")
        axy.set_facecolor("gray")

        for i, mask in enumerate([substrate, mos2_edge, mos2_core, junctiona, junctionb, ws2]):
            x = separator1[:][mask].flatten()
            y = separator2[:][mask].flatten()
            ax.scatter(x, y, alpha=0.3, color=ref_cmap(i/5))
        
        ax.set_xlabel(separator1.natural_name)
        ax.set_ylabel(separator2.natural_name)
        ax.grid(color="k")
        ax.set_xticks(ax.get_yticks())

    # rolling because ygrid of raman and pl differ by 1 (see workup)
    screen.create_channel("substrate", values=substrate)
    screen.create_channel("mos2_core", values=mos2_core)
    screen.create_channel("mos2_edge", values=mos2_edge)
    screen.create_channel("junctiona", values=junctiona)
    screen.create_channel("junctionb", values=junctionb)
    # screen.create_channel("ws2_core", values=ws2_core)
    screen.create_channel("ws2", values=ws2)
    screen.print_tree()

    ax2 = plt.subplot(gs[2])
    ax3 = plt.subplot(gs[3])

    # --- calibration curve for determining mos2 edge content ---
    x0 = screen["MoS2_A1g"][:][mos2_core].mean()
    y0 = screen["MoS2_E2g"][:][mos2_core].mean()
    x1 = screen["MoS2_A1g"][:][substrate].mean()
    y1 = screen["MoS2_E2g"][:][substrate].mean()
    ax0.plot([x0, x1], [y0, y1], color=ref_cmap(1/5), ls="--", lw=3)

    # --- calibration curve for determining junction content ---
    x0 = screen["WS2_A1g"][:][mos2_core].mean()
    y0 = screen["WS2_2LA"][:][mos2_core].mean()
    x1 = screen["WS2_A1g"][:][ws2].mean()
    y1 = screen["WS2_2LA"][:][ws2].mean()
    ax1.plot([x0, x1], [y0, y1], color=ref_cmap(3/5), ls="--", lw=3)


    patches = []
    mean_spectra = []
    for i, (label, name) in enumerate({
        r"$\mathsf{substrate}$": "substrate",
        r"$\mathsf{MoS}_2 \ \mathsf{(edge)}$": "mos2_edge",
        r"$\mathsf{MoS}_2$": "mos2_core",
        r"$\mathsf{Mo}_x \mathsf{W}_{1-x}\mathsf{S}_2$": "junctiona",
        r"$\mathsf{WS}_2 \ \mathsf{(edge)}$": "junctionb",
        r"$\mathsf{WS}_2 \ \mathsf{(core)}$": "ws2",
    }.items()):
        zone = screen[name][:]
        color = ref_cmap(i/5)
        cmap = ListedColormap([[0,0,0,0], color])
        patches.append(mpatches.Patch(color=color, label=label))
        ax2.pcolormesh(screen.x.points, screen.y.points, zone.T, cmap=cmap)
        raman.create_variable(name=name, values=zone[None, :, :])
        split = raman.split(name, [0.5])[1]
        y = np.nanmean(split.leveled[:], axis=(1,2))
        ax3.plot(split.energy.points, y, color=color, lw=2, alpha=0.8)
        mean_spectra.append(y)

    legend = ax2.legend(handles=patches, framealpha=0.7)
    ax2.grid()
    r = np.linspace(0, 30)


    def rot(r, deg):
        x = r * np.cos(deg * np.pi / 180)
        y = r * np.sin(deg * np.pi / 180)
        return x, y


    for third in [0, 120, 240]:
        for deg in [9, 49]:
            ax2.plot(*rot(r, third + deg), "k:")

    ax3.set_xlim(100, 500)
    ax3.grid(True)
    ax3.set_ylim(0, None)

    ax0.set_facecolor("gray")
    ax1.set_facecolor("gray")
    ax3.set_facecolor("gray")

    ax2.set_xlabel("x (um)")
    ax2.set_ylabel("y (um)")

    ax3.set_xlabel("Raman shift (wn)")
    ax3.set_ylabel("Intensity (a.u.)")

    if save:
        wt.artists.savefig(here / "separator.v2.png", fig=fig)
        screen.save(here / "screen.wt5", overwrite=True)
    else:
        # interactive mode:  display selected lineshapes by selecting from 2D spatial map pixels
        from matplotlib.patches import Rectangle

        class Slice:
            def __init__(self, fig):
                self.fig = fig
                self.x = 0
                self.y = 0
                self.visible = False
                self.rect = Rectangle(
                    [0, 0], 2, 2, ec=[0,0,0,1], fc=[0,0,0,0], lw=2, visible=False
                )
                self.spectrum, = ax3.plot([0], [0], color="k", linewidth=2)
                self.scatter_mos2, = ax0.plot(
                    [0], [0],
                    marker="o", ls="", visible=False, color=[0,0,0,1]
                )
                self.scatter_ws2, = ax1.plot(
                    [0], [0],
                    marker="o", ls="", visible=False, color=[0,0,0,1]
                )
                ax2.add_patch(self.rect)

            def __call__(self, info):
                if info.key in ["left", "right", "up", "down"]:  # key press
                    dx = dy = 0
                    if info.key == "left":
                        dx -= 2
                    elif info.key == "right":
                        dx += 2
                    elif info.key == "up":
                        dy += 2
                    elif info.key == "down":
                        dy -= 2
                    self.x += dx
                    self.y += dy         
                elif info.inaxes == ax2:  # mouse press
                    new_idx = np.abs(raman.x.points - info.xdata).argmin()
                    new_idy = np.abs(raman.y.points - info.ydata).argmin()
                    newx = raman.x.points[new_idx]
                    newy = raman.y.points[new_idy]
                    if not self.visible:  # reactivate
                        self.visible=True
                    elif newx == self.x and newy == self.y:
                        self.visible=False
                    self.x = newx
                    self.y = newy
                else:
                    import matplotlib as mpl
                    mpl.backend_bases.key_press_handler(info, fig.canvas, fig.canvas.toolbar)
                    self.fig.canvas.draw_idle()
                self.rect.set_x(self.x - 1)
                self.rect.set_y(self.y - 1)
                self.rect.set(visible=self.visible)
                self.spectrum.set(visible=self.visible)
                self.scatter_mos2.set(visible=self.visible)
                self.scatter_ws2.set(visible=self.visible)
                self._plot_spectrum(self.x, self.y)
                self._plot_scatter(self.x, self.y)
                self.fig.canvas.draw_idle()

            def _plot_scatter(self, x, y):
                screen.print_tree()
                chopped = screen.chop(at={"x": [x, "um"], "y": [y, "um"]})[0]
                xm = chopped["MoS2_A1g"][:]
                ym = chopped["MoS2_E2g"][:]
                xw = chopped["WS2_A1g"][:]
                yw = chopped["WS2_2LA"][:]
                self.scatter_mos2.set_data([[xm], [ym]])
                self.scatter_ws2.set_data([[xw], [yw]])

                print(xm, ym, xw, yw)

            def _plot_spectrum(self, x, y):
                spec = raman.chop("energy", at={"x": [x, "um"], "y": [y, "um"]})[0]
                self.spectrum.set_data(spec.energy.points, spec.leveled[:])

        event_handler = Slice(fig)
        cid = fig.canvas.mpl_connect('button_press_event', event_handler)
        cid = fig.canvas.mpl_connect('key_release_event', event_handler)

        plt.show()
        return cid


if __name__ == "__main__":
    from sys import argv
    if len(argv) > 1:
        save = argv[1] != "0"
    else:
        save = True
    out = main(save)
