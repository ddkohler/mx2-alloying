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
d_temp = raman.split("energy", [125])[0]
d_temp.moment(channel="intensity", axis="energy", moment=0)
d_temp.intensity_energy_moment_0.normalize()
screen.create_channel(name="separator2", values=d_temp.intensity_energy_moment_0[0])

fig, gs = wt.artists.create_figure(width="double", cols=[1, 1], wspace=0.75)
ax0 = plt.subplot(gs[0])
ax0.scatter(screen.separator1[:].flatten(), screen.separator2[:].flatten(), alpha=0.05)
ax0.set_xlabel("I(384 wn)")
ax0.set_ylabel("I(350 wn)")
ax0.grid(color="k")
ax0.set_xticks(ax0.get_yticks())

axx = wt.artists.add_sideplot(ax0, "x", pad=0.05)
axx.hist(screen.separator1[:].flatten(), bins=100)
axy = wt.artists.add_sideplot(ax0, along="y", pad=0.05)
axy.hist(screen.separator2[:].flatten(), bins=100, orientation="horizontal")
axx.set_ylim(None, 140)
axy.set_xlim(None, 30)

substrate = (screen.separator1[:] < 0.1) & (screen.separator2[:] < 0.1)
mos2 = np.logical_not(substrate) & (screen.separator2[:] < 0.155) & (screen.separator1[:] < 0.25)
ws2_core = (screen.separator1[:] > 0.756) & (screen.separator2[:] > 0.64)
ws2_edge = (screen.separator1[:] > 0.65) & (screen.separator2[:] > 0.65) & np.logical_not(ws2_core)
ws2 = ws2_core + ws2_edge
junction = np.logical_not(substrate) & np.logical_not(mos2) & np.logical_not(ws2)

# rolling because ygrid of raman and pl differ by 1 (see workup)
screen.create_channel("substrate", values=np.roll(substrate, 0, axis=1))
screen.create_channel("mos2", values=np.roll(mos2, 0, axis=1))
screen.create_channel("junction", values=np.roll(junction, 0, axis=1))
screen.create_channel("ws2", values=np.roll(ws2, 0, axis=1))
screen.create_channel("ws2_edge", values=np.roll(ws2_edge, 0, axis=1))
screen.create_channel("ws2_core", values=np.roll(ws2_core, 0, axis=1))
screen.print_tree()

ax1 = plt.subplot(gs[1])

# --- calibration curve for determining mx2 content ---
x0 = screen.separator1[:][mos2].mean()
y0 = screen.separator2[:][mos2].mean()
x1 = screen.separator1[:][ws2_edge].mean()
y1 = screen.separator2[:][ws2_edge].mean()

from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib import cm
ref_cmap = cm.get_cmap("magma")

ax0.plot([x0, x1], [y0, y1], color=ref_cmap(1/3), ls="--", lw=3)

patches = []
for i, (label, name) in enumerate({
    # "substrate": substrate,
    r"$\mathsf{MoS}_2$": "mos2",
    r"$\mathsf{Mo}_x \mathsf{W}_{1-x}\mathsf{S}_2$": "junction",
    r"$\mathsf{WS}_2 \ \mathsf{(edge)}$": "ws2_edge",
    r"$\mathsf{WS}_2 \ \mathsf{(core)}$": "ws2_core"
}.items()):
    zone = screen[name][:]
    color = ref_cmap(i/3)
    cmap = ListedColormap([[0,0,0,0], color])
    patches.append(mpatches.Patch(color=color, label=name))
    ax1.pcolormesh(screen.x.points, screen.y.points, zone.T, cmap=cmap)

ax1.legend(handles=patches)
wt.artists.savefig("separator.png", fig=fig)
screen.save(here / "screen.wt5")
