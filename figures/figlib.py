import numpy as np
from scipy.interpolate import interp1d
from matplotlib import cm
import pathlib

here = pathlib.Path(__file__).resolve().parent

# --- colors --------------------------------------------------------------------------------------


ref_cmap = cm.get_cmap("magma")
colors = ref_cmap(np.linspace(0, 1, 6))
grid_kwargs = {
    "ls": ":",
    "lw": "1",
    "c": "k"
}

# --- parameters ----------------------------------------------------------------------------------


d_SiO2 = 297e-7  # thickness of SiO2 layer, cm
d_mono = 7.5e-8  # thickness of MX2 monolayer, cm

# spectral shift (eV) to apply to literature refractive index of MX2
# positive values shift features to lower energy (redshift)
offset_mos2 = 0.04
offset_ws2 = 0.06


# --- refractive index data -----------------------------------------------------------------------


def from_refractiveindex_info(url, **kwargs) -> object:
    arr = np.genfromtxt(url, **kwargs, unpack=True)
    func = interp1d(1e4 / arr[0] / 8065.5, arr[1] + 1j * arr[2], kind='quadratic')
    return func


n_mos2_ml = from_refractiveindex_info(
    here.parent / "data" / "refractive index" / "MoS2" / "Islam-1L.yml",
    # r"https://refractiveindex.info/database/data/main/MoS2/Islam-1L.yml",
    skip_header=10, skip_footer=5,
)

n_ws2_ml = from_refractiveindex_info(
    here.parent / "data" / "refractive index" / "WS2" / "Ermolaev.yml",
    # r"https://refractiveindex.info/database/data/main/WS2/Ermolaev.yml", 
    # r"https://refractiveindex.info/database/data/main/WS2/Islam-1L.yml",  # no longer hosted?
    skip_header=10, skip_footer=5
)

n_Si = from_refractiveindex_info(
    here.parent / "data" / "refractive index" / "Si" / "Schinke.yml",
    # r"https://refractiveindex.info/database/data/main/Si/Schinke.yml",
    skip_header=11, skip_footer=2
)


def n_mos2(hw):
    return n_mos2_ml(hw + offset_mos2)


def n_ws2(hw):
    return n_ws2_ml(hw + offset_ws2)


def n_air(w):
    if isinstance(w, float) or isinstance(w, int):
        return 1
    return np.ones(w.shape, dtype=complex)


def n_fused_silica(w):
    if isinstance(w, float) or isinstance(w, int):
        return 1.46
    return np.ones(w.shape, dtype=complex) * 1.46

