import WrightTools as wt
import numpy as np
import pathlib
import re  # for string substitution
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import string
from PIL import Image

wt.close()

here = pathlib.Path(__file__).resolve().parent
data_dir = here.parent / "data"
p = "ZYZ543.wt5"
root = wt.open(data_dir / p)

try:
    from . import optical_model as om
except (ModuleNotFoundError, ImportError):
    import optical_model as om

try:
    from . import fitting as ft
except (ModuleNotFoundError, ImportError):
    import fitting as ft


all_plot = False
save = True

fig_type = ".png"

d = root.raman.proc_raman
x0 = 12  # x-slice to examine further (um)

# define masks for MoS2, WS2, and Si
d_temp = d.chop("x", "y", at={"energy": [383, "wn"]}, verbose=False)[0]
d_temp.intensity.normalize()
separator = d_temp.intensity[:]
MX2_area = separator > 0.2
WS2_area = separator > 0.4
MoS2_area = MX2_area & ~WS2_area

d_temp_2 = d.chop("x", "y", at={"energy": [404, "wn"]}, verbose=False)[0]
separator_2 = d_temp_2.intensity[:]
separator_2[WS2_area] = 0
separator_2 /= np.nanmax(separator_2)
MoS2_edge = separator_2 > 0.77
MoS2_non_edge = MoS2_area & ~MoS2_edge

d_temp_3 = d.chop("x", "y", at={"energy": [350, "wn"]}, verbose=False)[0]
separator_3 = d_temp_3.intensity[:]
separator_3[MoS2_area] = 0
separator_3 /= np.nanmax(separator_3)
WS2_edge = separator_3 > 0.9
WS2_non_edge = WS2_area & ~WS2_edge

d.create_channel("leveled", values=d.intensity[:])
d.level("leveled", 0, -20)


if False:  # inspect the mask regions
    fig, gs = wt.artists.create_figure(cols=[1, 1, 1])
    ax0=plt.subplot(gs[0])
    ax1=plt.subplot(gs[1])
    ax2=plt.subplot(gs[2])
    ax0.pcolormesh(d.x.points, d.y.points, WS2_area.T)
    ax1.pcolormesh(d.x.points, d.y.points, MoS2_edge.T)
    ax2.pcolormesh(d.x.points, d.y.points, MoS2_area.T)
    plt.show()

if True:  # interact2D
    # d.level("intensity", 0, -20)
    # root.print_tree(2)
    d.intensity.clip(max=300)
    d.smooth((1, 0, 0))
    # d.intensity **= 0.5

    d.transform("x", "y", "energy")
    out = wt.artists.interact2D(d, channel='intensity', yaxis="y", xaxis="energy")
    plt.show()

if False:  # spectral averaging for representative spectra
    # TODO: combine fits with data
    # d.level("intensity", 0, -2)

    fig, gs = wt.artists.create_figure(nrows=2, default_aspect=0.5, hspace=0.3)
    ax0 = plt.subplot(gs[0])
    plt.xticks(visible=False)
    plt.grid()
    ax1 = plt.subplot(gs[1], sharex=ax0)
    plt.grid()
    ax0.set_title(r"WS$_2$")
    ax1.set_title(r"MoS$_2$")
    ax0.set_ylabel(r"Intensity")
    ax1.set_ylabel(r"Intensity")
    ax1.set_xlabel(r"Raman shift (cm $^{-1}$)")
    z = d.intensity[:]

    z_MoS2 = np.nanmean(z[:, MoS2_area], axis=1)
    z_MoS2_edge = np.nanmean(z[:, MoS2_edge], axis=1)
    z_MoS2_non_edge = np.nanmean(z[:, MoS2_non_edge], axis=1)
    z_WS2 = np.nanmean(z[:, WS2_area], axis=1)
    z_WS2_edge = np.nanmean(z[:, WS2_edge], axis=1)
    z_WS2_non_edge = np.nanmean(z[:, WS2_non_edge], axis=1)

    ax0.plot(d.energy.points, z_WS2_edge,
        linewidth=2, alpha=0.7, label="WS2 edge", color='k', linestyle=":"
    )
    ax0.plot(d.energy.points, z_WS2_non_edge,
        linewidth=2, alpha=0.7, label="WS2", color='k'
    )
    ax1.plot(d.energy.points, z_MoS2_edge,
        linewidth=2, alpha=0.7, label="MoS2 thin", color='k', linestyle=":"
    )
    ax1.plot(d.energy.points, z_MoS2_non_edge,
        linewidth=2, alpha=0.7, label="MoS2", color='k'
    )

    ax0a = ax0.inset_axes([0.1, 0.6, 0.15, 0.3])
    ax0a.pcolormesh(WS2_edge.T, cmap="gray_r")
    ax0b = ax0.inset_axes([0.3, 0.6, 0.15, 0.3])
    ax0b.pcolormesh(WS2_non_edge.T, cmap="gray_r")

    ax1a = ax1.inset_axes([0.1, 0.6, 0.15, 0.3])
    ax1a.pcolormesh(MoS2_edge.T, cmap="gray_r")
    ax1b = ax1.inset_axes([0.3, 0.6, 0.15, 0.3])
    ax1b.pcolormesh(MoS2_non_edge.T, cmap="gray_r")

    for ax in [ax0a, ax0b, ax1a, ax1b]:
        ax.set_xticks([])
        ax.set_yticks([])

    ax1.set_ylim(None, 50)
    # ax0.set_ylim(0, None)
    ax1.set_xlim(None, 500)
    wt.artists.savefig(here / "raman_spectra.png", facecolor="white")
    # plt.show()
