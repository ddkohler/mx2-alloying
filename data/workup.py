import numpy as np
import matplotlib.pyplot as plt
import WrightTools as wt
import os
import pathlib
import time
from scipy.ndimage import median_filter


all_import = True
data_dir = pathlib.Path(__file__).parent
root = wt.Collection(name="root")
verbose = False
talkback = True

tstart = time.time()
if talkback:
    print("starting import")


# --- define --------------------------------------------------------------------------------------


def from_Horiba(p, energy_units="nm", name=None, parent=None):
    # grab arrays
    arr = np.genfromtxt(p, unpack=True, skip_header=1)
    indxx = arr[0]
    indxy = arr[1]
    channel_arr = arr[2:, :]
    energy = np.genfromtxt(p, unpack=True, skip_footer=arr.shape[1])
    # shape arrays
    length = int(np.sqrt(indxx.size))
    indxx = indxx.reshape((length, length))[:, 0][None, :, None]
    indxy = indxy.reshape((length, length))[0, :][None, None, :]
    energy = energy.reshape((energy.size, 1, 1))
    channel_arr = channel_arr.reshape((energy.size, length, length))
    # create data object
    d = wt.Data(name=name, parent=parent)
    d.create_variable("x", values=indxx, units="um", label="x")
    d.create_variable("y", values=indxy, units="um", label="y")
    d.create_variable("energy", values=energy, units=energy_units, label="m")
    d.create_channel("intensity", values=channel_arr)
    d.transform("x", "y", "energy")
    return d


def from_Solis(
    path, name=None, parent=None, cosmic_filter=True, background_subtract=True, sumx=False
):
    temp = wt.data.from_Solis(path, verbose=verbose)
    if cosmic_filter:
        temp.signal[:] = median_filter(temp.signal[:], size=3)
    if background_subtract:
        background = temp.split("xindex", [1300], verbose=False)[0]
        temp.signal[:] -= background.signal[:].mean()
    temp = temp.split("yindex", [200, 1300], verbose=False)[1]
    if sumx:
        temp.moment("xindex", moment=0)
    temp = temp.copy(name=name, parent=parent, verbose=False)
    return temp


# --- PL and Raman --------------------------------------------------------------------------------


# PL
if all_import:
    # raw PL data
    PL = root.create_collection(name="PL")
    p = "B-PLmapping-532-D1-300-300G-50x-1-05-1.txt"
    p = data_dir / "characterization" / p
    d = from_Horiba(p, "nm", "raw_PL", parent=PL)
    # processed
    d = d.copy(parent=PL, name="proc_PL", verbose=verbose)
    d.convert("eV", verbose=verbose, convert_variables=True)
    d.level("intensity", 0, 2, verbose=verbose)
    d.level("intensity", 1, 2, verbose=verbose)
    d.smooth([2, 0, 0], verbose=verbose)

    d.moment("energy", moment=1)
    arr = d.intensity[:]
    arr_collapse = np.mean(arr, axis=0)
    mask = np.ones(arr_collapse.shape)
    mask[arr_collapse < arr_collapse.max() * 0.04] = 0
    nanmask = mask.copy()
    nanmask[nanmask == 0] = np.nan

    # datasets to plot
    d.intensity_energy_moment_1 *= nanmask[None, :]
    d.channels[-1].clip(d.energy.min(), d.energy.max())
    d.channels[-1].null = d.energy.min()

    yvar = d["y"]
    yvar += 8

    if talkback:
        print("PL done", time.time() - tstart)

# Raman
if all_import:
    # raw
    raman = root.create_collection(name="raman")
    p = "B-Ramanmapping-532-D1-300-2400G-50x-1-05-1.txt"
    p = data_dir / "characterization" / p
    d = from_Horiba(p, "wn", "raw_raman", parent=raman)
    # processed
    d = d.copy(parent=raman, name="proc_raman", verbose=verbose)
    # bad pixels
    for idx in [247, 248, 364, 365, 638, 639]:
        d.intensity[idx] = np.nan
    d.transform("energy", "x", "y")
    d.heal(method="nearest")
    d.transform("x", "y", "energy")
    yvar = d["y"]
    yvar += 6  # tweaked by one pixel to match with PL
    d.create_channel("leveled", values=d.intensity[:])
    d.level("leveled", 0, -20)

    # d.smooth([2, 0, 0], verbose=verbose)
    # d.level("intensity", 0, 2, verbose=verbose)
    # d.level("intensity", 1, 2, verbose=verbose)

    if talkback:
        print("Raman done", time.time() - tstart)

# high-contrast optical image
if all_import:
    um_per_pixel = 100 / 275  # from scale bar
    um_per_pixel *= 0.83 # unfortunately, the um_per_pixel conversion seems to be off (or else PL map is wrong)
    p = "microscope1 - enhanced contrast - pl windowed.tiff"
    p = data_dir / "characterization" / p
    from PIL import Image
    img = np.asarray(Image.open(p))
    s = img.shape[0]
    origin = [236, 306]  # x, y
    # extent = np.array([origin[0]-s, origin[0], origin[1] - s, origin[1]]) * um_per_pixel
    # extent *= 0.83  
    d = wt.data.Data(name="om", parent=PL)

    for i, color in enumerate(["r", "g", "b"]):
        d.create_channel(name=color, values=img[..., i])
    l = np.arange(s) * um_per_pixel
    d.create_variable(name="x", values=(l - l[-origin[0]])[:, None])
    d.create_variable(name="y", values=(l - l[-origin[1]])[None, :])
    d.transform("x", "y")

    if talkback:
        print("OM high contrast done", time.time() - tstart)


# --- Microscope ----------------------------------------------------------------------------------


if all_import:
    from PIL import Image

    p = "microscope1.tif"
    p = data_dir / "characterization" / p
    img = Image.open(p).convert("P")
    img_arr = np.asarray(img, dtype=float).T

    x = np.linspace(0, 12.25, img_arr.shape[0]) * 100 / 2.125  # rough conversion to um
    y = np.linspace(0, 9.25, img_arr.shape[1]) * 100 / 2.125  # rough conversion to um
    # clip outside of image
    img_arr = img_arr[280:1250, 140:965]
    x = x[280:1250]
    y = y[140:965]
    x -= x.min()
    y -= y.min()
    # create data object
    micro = root.create_collection(name="micro")
    d = wt.Data(name="micro", parent=micro)
    d.create_variable("x", values=x[:, None], units="um")
    d.create_variable("y", values=y[None, :], units="um")
    d.create_channel("intensity", values=img_arr)
    d.transform("x", "y")

    if talkback:
        print("microscope done", time.time() - tstart)


# --- SHG polarization ----------------------------------------------------------------------------


if all_import:
    # need to offset axes to have images agree with SHG
    # offsets found by superposing SHG and lamp image patterns
    # image coords + offset = SHG coords
    y_offset = 12
    x_offset = 11

    # SHG
    raw = wt.Collection(name="raw")
    for i, entry in enumerate(os.scandir(data_dir / "shg microscopy" / "polarization")):
        if entry.name.endswith(".asc"):
            angle = float(entry.name.split(" ")[0])
            temp = from_Solis(entry.path, name=str(i), parent=raw, sumx=True)
            yindex = temp["yindex"]
            xindex = temp["xindex"]
            yindex -= y_offset
            xindex -= x_offset
            temp.create_variable(name="angle", values=angle * np.ones([1 for i in range(temp.ndim)]))
            temp.transform(*temp.axis_names, "angle")
    shg_pol = wt.Collection(name="Jan29")
    pol = wt.data.join([d for d in raw.values()])
    pol = pol.copy(parent=shg_pol, name="polarization")

    # images
    shg_pol.create_collection(name="imgs")
    from_Solis(
        data_dir / "shg microscopy" / "lamp and pump v2.asc",
        name="lamp_and_pump", parent=shg_pol.imgs,
        cosmic_filter=False, background_subtract=False
    )
    from_Solis(
        data_dir / "shg microscopy" / "lamp unfiltered without slit.asc",
        name="lamp", parent=shg_pol.imgs,
        cosmic_filter=False, background_subtract=False
    )
    from_Solis(
        data_dir / "shg microscopy" / "pump - slit 200um - concatenated image.asc",
        name="pump", parent=shg_pol.imgs,
        cosmic_filter=False, background_subtract=True, sumx=True,
    )
    from_Solis(
        data_dir / "shg microscopy" / "400 nm lamp slit transmission.asc",
        name="slit", parent=shg_pol.imgs,
        cosmic_filter=False, background_subtract=False
    )

    # power dependence with angle
    offset = 49 * np.pi / 180
    # data transcribed from notes
    xy = np.array([
        [0, 1.42],
        [20, 1.68],
        [40, 2.11],
        [60, 2.47],
        [80, 2.64],
        [100, 2.51],
        [120, 2.06],
        [140, 1.63],
        [160, 1.41],
        [180, 1.48],
        [200, 1.74],
        [220, 2.15],
        [240, 2.49],
        [260, 2.60],
        [280, 2.44],
        [300, 2.06],
        [320, 1.64],
        [340, 1.39],
    ])

    angles = xy[:, 0]
    powers = xy[:, 1]

    power = shg_pol.create_data(name="pump_power")
    power.create_variable(name="raw_angle", values=angles)
    power.create_variable(name="angle", values=angles - offset)
    power.create_channel(name="power", values=powers)
    power.transform("raw_angle")

    # shg_pol.print_tree()
    shg_pol.save(data_dir / "polarization.wt5")


# --- Finish --------------------------------------------------------------------------------------


if all_import:
    p = "ZYZ543.wt5"
    root.save(data_dir / p, overwrite=True)

    if talkback:
        print("DONE! Total time", time.time() - tstart)
