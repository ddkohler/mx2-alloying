import pathlib
import WrightTools as wt
import numpy as np
import matplotlib.pyplot as plt
import tmm_lib as lib
import figlib as fl


here = pathlib.Path(__file__).resolve().parent
data = here.parent / "data" / "reflection_microspectroscopy" / "reflection.wt5"
root = wt.open(data)

d = root["spectrum"]
hw = np.linspace(1.6, 2.7, 201)  # eV

d_mono = fl.d_mono
n_fused_silica = fl.n_fused_silica
n_air = fl.n_air
n_Si = fl.n_Si

wt.artists.apply_rcparams(kind="publication")


# -------------------------------------------------------------------------------------------------


def run(save):
    # d2 = 75e-7
    fig, gs = wt.artists.create_figure(width="dissertation", cols=[1, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    # plt.title(f"Reflection Contrast vs. SIO2 Thickness ")
    for i, d2i in enumerate(np.linspace(240, 340, 6)):
        # nmos2 = n_semi_1D(E0s_mos2, Gs_mos2, As_mos2, 0, offset_mos2)
        def nmos2(hw):
            return fl.n_mos2_ml(hw + 0.04)
        label = str(d2i)
        if i==0:
            label += " nm"
        blank = lib.FilmStack([n_air, n_fused_silica, n_Si], [d2i * 1e-7])
        sample = lib.FilmStack([n_air, nmos2, n_fused_silica, n_Si], [d_mono, d2i * 1e-7])
        r_blank = blank.RT(hw)[0]
        r_sample = sample.RT(hw)[0]
        contrast = (r_sample - r_blank) / r_blank
        ax1.plot(hw, contrast, linewidth=3, alpha=0.5, label=label)
        ax0.plot(hw, r_blank, linewidth=3, alpha=0.5, label=label)

    for y in [-15]:
        speci = d.chop("wm", at={"y":[y, "um"]})[0]
        # speci.print_tree()
        ax1.plot(
            speci.wm[:], speci.contrast[:],
            color="k", label="experiment \n" + r"($x=0 \ \mu \mathsf{m}, y=-16 \ \mu \mathsf{m}$)"
        )
    l = ax0.legend(fontsize=12, framealpha=0.7, bbox_to_anchor=(1.1, 1.2, 1.5, ), ncol=3)
    wt.artists.corner_text("a", ax=ax0)
    wt.artists.corner_text("b", ax=ax1)
    wt.artists.set_ax_labels(
        ax1,
        xlabel=r"$\hbar \omega \ \left(\mathsf{eV} \right)$",
        ylabel=r"$(R - R_0) / R_0$"
    )
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()
    wt.artists.set_ax_labels(
        ax0,
        xlabel=r"$\hbar \omega \ \left(\mathsf{eV} \right)$",
        ylabel=r"$R_0$"
    )
    ax0.grid()
    ax1.grid()
    if save:
        wt.artists.savefig(here / "reflection_constrast_vs_SiO2_thickness.png")
    else:
        plt.show()


if __name__ == "__main__":
    from sys import argv
    if len(argv) > 1:
        save = argv[1] != "0"
    else:
        save = True
    run(save)
