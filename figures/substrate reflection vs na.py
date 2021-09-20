# using TMM to examine various consequences of reflections from off-axis rays
# mostly considering the Si/SiO2 substrate

import pathlib
import WrightTools as wt
import numpy as np
import matplotlib.pyplot as plt
import tmm_lib as lib
import optical_model as om

here = pathlib.Path(__file__).parent
hw = np.linspace(1.5, 3, 201)


def from_refractiveindex_info(url, **kwargs):
    arr = np.genfromtxt(url, **kwargs, unpack=True)
    from scipy.interpolate import interp1d
    func = interp1d(1e4 / arr[0] / 8065.5, arr[1] + 1j * arr[2], kind='quadratic')
    return func


n_Si = from_refractiveindex_info(
    r"https://refractiveindex.info/database/data/main/Si/Schinke.yml",
    skip_header=11, skip_footer=2
)

n_fused_silica = lib.n_fused_silica
n_air = lib.n_air
d2 = 300e-7

for i, angle in enumerate(np.linspace(0, 80, 9)):
    blank = lib.FilmStack([n_air, n_fused_silica, n_Si], [d2], angle=angle * np.pi / 180)
    plt.plot(np.ones(hw.shape)*angle + 3 * (hw - hw.mean()), blank.RT(hw)[0], label=str(angle))
plt.title("angle of incedence vs. reflection")
plt.legend()
plt.grid()
plt.show()

