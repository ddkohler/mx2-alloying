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
ref_cmap = cm.get_cmap("magma")
colors = ref_cmap(np.linspace(0, 1, 4))

here = pathlib.Path(__file__).resolve().parent
root = wt.open(here.parent / "data" / "data.wt5")

raman = root.raman.proc
pl = root.pl.proc
screen = wt.Data(name="screen")
screen.create_variable(name="x", values=pl.x[0])
screen.create_variable(name="y", values=pl.y[0])

separators = {
    "MoS2_E2g": [381, 385],
    "MoS2_A1g": [403, 409],
    "WS2_A1g": [415, 417],
    "WS2_nr": [540, 550],
    "WS2_2LA": [349, 352]
}

for name, window in separators.items():
    d_temp = raman.split("energy", window)[1]
    d_temp.moment(channel="intensity", axis="energy", moment=0)
    d_temp.intensity_energy_moment_0.normalize()
    screen.create_channel(name, values=d_temp.intensity_energy_moment_0[0])

substrate = (screen["MoS2_E2g"][:] < 0.05) & (screen["MoS2_A1g"][:] < 0.05)
mos2 = np.logical_not(substrate) & (screen["MoS2_E2g"][:] < 0.25) & (screen["MoS2_A1g"][:] < 0.41) \
    & (screen["WS2_A1g"][:] < 0.18)
mos2_edge = mos2 & ((screen["MoS2_E2g"][:] < 0.137) | (screen["MoS2_A1g"][:] < 0.27))
mos2_core = mos2 & np.logical_not(mos2_edge)
junction = np.logical_not(substrate + mos2) & (screen["MoS2_E2g"][:] < 0.76) & (screen["MoS2_A1g"][:] < 0.755)
# junction = np.logical_not(substrate + mos2) & (screen["WS2_2LA"][:] < 0.6) & (screen["WS2_A1g"][:] < 0.6)
# ws2_core = np.logical_not(junction + substrate + mos2)
ws2 = np.logical_not(substrate + mos2 + junction)
junctiona = junction & (screen["WS2_2LA"][:] < 0.6)
junctionb = junction & np.logical_not(junctiona)

fig, gs = wt.artists.create_figure(
    width="double", cols=[1, 1], wspace=0.75, hspace=0.75, nrows=2,
)

ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
for ax, separators, lim in zip(
    [ax0, ax1],
    [["MoS2_E2g", "MoS2_A1g"],
    ["WS2_A1g", "WS2_2LA"]],
    [140, 40]
):
    separator1 = screen[separators[0]]
    separator2 = screen[separators[1]]

    axx = wt.artists.add_sideplot(ax, "x", pad=0.05)
    axx.hist(separator1[:].flatten(), bins=100)
    axy = wt.artists.add_sideplot(ax, along="y", pad=0.05)
    axy.hist(separator2[:].flatten(), bins=100, orientation="horizontal")
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

# --- calibration curve for determining mos2 edge content
x0 = screen["MoS2_E2g"][:][mos2_core].mean()
y0 = screen["MoS2_A1g"][:][mos2_core].mean()
x1 = screen["MoS2_E2g"][:][substrate].mean()
y1 = screen["MoS2_A1g"][:][substrate].mean()
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
    r"$\mathsf{WS}_2 \ \mathsf{(odd)}$": "junctionb",
    r"$\mathsf{WS}_2 \ \mathsf{(core)}$": "ws2",
}.items()):
    zone = screen[name][:]
    color = ref_cmap(i/5)
    cmap = ListedColormap([[0,0,0,0], color])
    patches.append(mpatches.Patch(color=color, label=label))
    ax2.pcolormesh(screen.x.points, screen.y.points, zone.T, cmap=cmap)
    raman.create_variable(name=name, values=zone[None, :, :])
    split = raman.split(name, [0.5])[1]
    y = np.nanmean(split.intensity[:], axis=(1,2))
    # std = np.nanstd(split.intensity[:], axis=(1,2))
    # ax3.fill_between(split.energy.points, y-std, y+std, color=color, alpha=0.5, lw=0)
    ax3.plot(split.energy.points, y, color=color, lw=2, alpha=0.8)
    mean_spectra.append(y)

ax3.plot(
    split.energy.points, 0.5 * mean_spectra[2] + 0.5 * mean_spectra[5],
    c="red",
    label="0.5 WS2 + 0.5 MoS2"
)
ax3.legend(framealpha=0.7)

legend = ax2.legend(handles=patches, framealpha=0.7)
ax2.grid()

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

# plt.show()
# 1/0
wt.artists.savefig(here / "separator.v2.png", fig=fig)
screen.save(here / "screen.wt5", overwrite=True)
