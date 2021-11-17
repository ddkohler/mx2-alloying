"""accurately define the MoS2 WS2, and junction domains"""


import numpy as np
import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt
from fitlib import fom

here = pathlib.Path(__file__).resolve().parent
root = wt.open(here.parent / "data" / "data.wt5")

raman = root.raman.proc
pl = root.pl.proc
screen = wt.Data(name="screen")
screen.create_variable(name="x", values=pl.x[0])
screen.create_variable(name="y", values=pl.y[0])

# --- separator 1 ---
# MoS2 e2g mode; WS2 also gives signal in this range
d_temp = raman.split("energy", [381, 385])[1]
d_temp.moment(channel="intensity", axis="energy", moment=0)
d_temp.intensity_energy_moment_0.normalize()
screen.create_channel("separator1", values=d_temp.intensity_energy_moment_0[0])

# --- separator 2 ---
# non-resonant signal from WS2; avoids some electronic enhancement effects of WS2 raman modes
d_temp = raman.split("energy", [403, 409])[1]
d_temp.moment(channel="intensity", axis="energy", moment=0)
d_temp.intensity_energy_moment_0.normalize()
screen.create_channel(name="separator2", values=d_temp.intensity_energy_moment_0[0])

fig, gs = wt.artists.create_figure(
    width="double", cols=[1, 1], wspace=0.75, hspace=0.75, nrows=2,
    aspects=[[[0, 0], 1.], [[1, 0], 0.4]]
)
ax0 = plt.subplot(gs[0])
ax0.scatter(screen.separator1[:].flatten(), screen.separator2[:].flatten(), alpha=0.1)
ax0.set_xlabel("I(384 wn)")
ax0.set_ylabel("I(405 wn)")
ax0.grid(color="k")
ax0.set_xticks(ax0.get_yticks())

axx = wt.artists.add_sideplot(ax0, "x", pad=0.05)
axx.hist(screen.separator1[:].flatten(), bins=100)
axy = wt.artists.add_sideplot(ax0, along="y", pad=0.05)
axy.hist(screen.separator2[:].flatten(), bins=100, orientation="horizontal")
axx.set_ylim(None, 140)
axy.set_xlim(None, 50)

substrate = (screen.separator1[:] < 0.05) & (screen.separator2[:] < 0.05)
mos2 = np.logical_not(substrate) & (screen.separator2[:] < 0.405) & (screen.separator1[:] < 0.25)
junction = np.logical_not(substrate + mos2) & (screen.separator1[:] < 0.775) & (screen.separator2[:] < 0.745)
ws2 = np.logical_not(substrate + mos2 + junction)
# ws2_edge = (screen.separator1[:] > 0.76) & (screen.separator2[:] > 0.72) & np.logical_not(ws2_core)
# ws2 = ws2 + ws2_edge

# rolling because ygrid of raman and pl differ by 1 (see workup)
screen.create_channel("substrate", values=np.roll(substrate, 0, axis=1))
screen.create_channel("mos2", values=np.roll(mos2, 0, axis=1))
screen.create_channel("junction", values=np.roll(junction, 0, axis=1))
screen.create_channel("ws2", values=np.roll(ws2, 0, axis=1))
# screen.create_channel("ws2_edge", values=np.roll(ws2_edge, 0, axis=1))
screen.create_channel("ws2", values=np.roll(ws2, 0, axis=1))
screen.print_tree()

ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2:])

# --- calibration curve for determining mx2 content ---
x0 = screen.separator1[:][mos2].mean()
y0 = screen.separator2[:][mos2].mean()
x1 = screen.separator1[:][ws2].mean()
y1 = screen.separator2[:][ws2].mean()

from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib import cm
ref_cmap = cm.get_cmap("magma")

ax0.plot([x0, x1], [y0, y1], color=ref_cmap(1/3), ls="--", lw=3)

patches = []
mean_spectra = []
for i, (label, name) in enumerate({
    # "substrate": substrate,
    r"$\mathsf{MoS}_2$": "mos2",
    r"$\mathsf{Mo}_x \mathsf{W}_{1-x}\mathsf{S}_2$": "junction",
    # r"$\mathsf{WS}_2 \ \mathsf{(edge)}$": "ws2_edge",
    r"$\mathsf{WS}_2 \ \mathsf{(core)}$": "ws2"
}.items()):
    zone = screen[name][:]
    color = ref_cmap(i/3)
    cmap = ListedColormap([[0,0,0,0], color])
    patches.append(mpatches.Patch(color=color, label=label))
    ax1.pcolormesh(screen.x.points, screen.y.points, zone.T, cmap=cmap)
    raman.create_variable(name=name, values=zone[None, :, :])
    split = raman.split(name, [0.5])[1]
    y = np.nanmean(split.intensity[:], axis=(1,2))
    # std = np.nanstd(split.intensity[:], axis=(1,2))
    # ax2.fill_between(split.energy.points, y-std, y+std, color=color, alpha=0.5, lw=0)
    ax2.plot(split.energy.points, y, color=color, lw=2, alpha=0.8)
    mean_spectra.append(y)

# ax2.plot(split.energy.points, 0.5 * mean_spectra[0] + 0.5 * mean_spectra[2], c="red")

ax1.legend(handles=patches)
ax1.grid()

ax2.set_xlim(100, 500)
ax2.grid(True)
ax2.set_ylim(0, None)
ax2.set_facecolor("gray")

# plt.show()
wt.artists.savefig(here / "separator.v2.png", fig=fig)
screen.save(here / "screen.wt5")
