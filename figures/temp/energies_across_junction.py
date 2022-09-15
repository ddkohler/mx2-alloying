import numpy as np
import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt


def run(save):
    here = pathlib.Path(__file__).resolve().parent
    data_dir = here.parent / "data"
    root = wt.open(data_dir / "heterostructure.wt5")
    pl = root.pl.proc

    arr = pl.intensity[:]
    arr_collapse = np.mean(arr, axis=0)

    mask = np.ones(arr_collapse.shape)
    mask[arr_collapse < arr_collapse.max() * 0.04] = 0
    nanmask = mask.copy()
    nanmask[nanmask == 0] = np.nan
    pl.intensity_energy_moment_1 *= nanmask

    crossover = pl.chop("energy", "y", at={"x":[0, 'um']})[0]

    spectrum = root.reflection.x100

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

    if save:
        p = f"energies_across_junction.png"
        p = here / p
        wt.artists.savefig(p)
    else:
        plt.show()


if __name__ == "__main__":
    from sys import argv
    if len(argv) > 1:
        save = argv[1] != "0"
    else:
        save = True
    run(save)
