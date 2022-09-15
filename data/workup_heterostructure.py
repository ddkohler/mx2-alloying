import numpy as np
import matplotlib.pyplot as plt
import WrightTools as wt
import os
import pathlib
import time
from scipy.ndimage import median_filter


all_import = True
here = pathlib.Path(__file__).resolve().parent
data_dir = here / "heterostructure" 
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
    PL = root.create_collection(name="pl")
    p = data_dir / "characterization" / "B-PLmapping-532-D1-300-300G-50x-1-05-1.txt"
    d = from_Horiba(p, "nm", "raw", parent=PL)
    # processed
    d = d.copy(name="temp", verbose=verbose)
    d.convert("eV", verbose=verbose, convert_variables=True)
    d.level("intensity", 0, 2, verbose=verbose)  # remove fluctuating baseline noise
    d.level(0, 2, 2, verbose=verbose)  # remove pixel-specific offsets
    d.smooth([2, 0, 0], verbose=verbose)

    d.moment("energy", moment=1)

    arr = d.intensity[:]
    arr_collapse = np.mean(arr, axis=0)
    mask = np.ones(arr_collapse.shape)
    mask[arr_collapse < arr_collapse.max() * 0.05] = 0
    nanmask = mask.copy()
    nanmask[nanmask == 0] = np.nan

    # datasets to plot
    d.intensity_energy_moment_1 *= nanmask[None, :]
    d.channels[-1].clip(d.energy.min(), d.energy.max())
    d.channels[-1].null = d.energy.min()

    yvar = d["y"]
    yvar += 8
    d = d.split("y", [-40, 56])[1]
    d = d.copy(parent=PL, name="proc", verbose=verbose)

    if talkback:
        print("PL done", time.time() - tstart)

# Raman
if all_import:
    # raw
    raman = root.create_collection(name="raman")
    p = data_dir / "characterization" / "B-Ramanmapping-532-D1-300-2400G-50x-1-05-1.txt"
    d = from_Horiba(p, "wn", "raw", parent=raman)
    # processed
    d = d.copy(name="temp", verbose=verbose)
    # bad pixels
    for idx in [247, 248, 364, 365, 638, 639]:
        d.intensity[idx] = np.nan
    idx = np.unravel_index(np.nanargmax(d.intensity[:]), d.shape)
    d.intensity[:, idx[1], idx[2]] = np.nan
    d.transform("energy", "x", "y")
    d.heal(method="nearest")
    d.transform("x", "y", "energy")
    # normalized by Si contribution
    d.level("intensity", 2, 5)
    # d.level("intensity", 0, 5) # need to level on energy axis to avoid removing Si peaks
    evar = d["energy"]
    evar += 0.7  # based on calibration of Si peak; see "calibrate_raman.py"
    yvar = d["y"]
    yvar += 6  # tweaked height to match with PL
    d = d.split("y", [-40, 56])[1]
    d = d.copy(parent=raman, name="proc", verbose=verbose)
    d.create_channel("leveled", values=d.intensity[:])
    d.level("leveled", 0, -20)

    if talkback:
        print("Raman done", time.time() - tstart)


# --- high-contrast optical image -----------------------------------------------------------------


if all_import:
    um_per_pixel = 100 / 275  # from scale bar
    um_per_pixel *= 0.83 # the um_per_pixel conversion seems to be slightly off (or else PL map is wrong)
    p = data_dir / "characterization" / "microscope1 - enhanced contrast - pl windowed.tiff"
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


# --- SHG polarization ----------------------------------------------------------------------------


if all_import:
    from scipy.interpolate import interp1d
    # need to offset axes to have images agree with SHG
    # offsets found by superposing SHG and lamp image patterns
    # image coords + offset = SHG coords
    y_offset = 12
    x_offset = 11
    um_per_pixel = 0.3  # 20x
    ylims = [200, 950]  # truncate image

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
    shg_pol = wt.Collection(name="shg", parent=root)
    pol = wt.data.join([d for d in raw.values()])
    pol = pol.split("yindex", ylims)[1]

    # images
    shg_pol.create_collection(name="imgs")
    for filename, name, kwargs in zip(
        [
            "lamp and pump v2",
            "lamp unfiltered without slit",
            "pump - slit 200um - concatenated image",
            "pump - slit 200um - concatenated image"
        ],
        ["lamp_and_pump", "lamp", "pump", "slit"],
        [
            dict(cosmic_filter=False, background_subtract=False),
            dict(cosmic_filter=False, background_subtract=False),
            dict(cosmic_filter=False, background_subtract=True, sumx=True),
            dict(cosmic_filter=False, background_subtract=False)
        ]
    ):
        temp = from_Solis(
            data_dir / "shg microscopy" / (filename + ".asc"), name=name, **kwargs
        ).split("yindex", ylims, verbose=verbose)[1]
        temp.copy(parent=shg_pol.imgs, name=name)
        temp.close()

    # center
    def pixel_to_um(var):
        out = var * um_per_pixel
        out -= out.mean()
        return out

    for d in [pol, shg_pol.imgs.lamp, shg_pol.imgs.pump, shg_pol.imgs.slit]:
        d.create_variable("xdistance", values=pixel_to_um(d["xindex"][:]), units="um")
        d.create_variable("ydistance", values=pixel_to_um(d["yindex"][:]), units="um")
        d.print_tree()
        d.transform("ydistance", "xdistance")

    def raw_angle_to_physical(deg):
        """ 
        Relate recorded angle to the image axes.
        Raw number is off by an offset
        0 deg corresponds to H-pol in the lab frame, which is V-pol for our image in this figure
        """
        offset = 49.
        return deg - offset

    # power dependence with angle
    # offset = 49 * np.pi / 180
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
    power.create_variable(name="angle", values=raw_angle_to_physical(angles))
    power.create_channel(name="power", values=powers)
    power.transform("raw_angle")


    def get_power_function(angle, power):
        """power interpolation for angle
        """
        x = list(angle)
        y = list(power)
        x.append(360)
        y.append(y[0])
        x = np.array(x)
        f = interp1d(x, np.array(y))
        return f

    def get_bs_transmission_function():
        """plate beamsplitter corrections
        based on transmission measurements for both H-pol and V-pol
        """
        x = np.linspace(0, 360)
        y = (0.5 * np.sin(x * np.pi / 180))**2 + (.83 * np.cos(x * np.pi / 180))**2
        f = interp1d(x, np.array(y))
        return f

    f1 = get_power_function(shg_pol.pump_power.raw_angle[:], shg_pol.pump_power.power[:])
    f2 = get_bs_transmission_function()

    pol.signal_xindex_moment_0[:] /= shg_pol.imgs.pump.signal_xindex_moment_0[0, 0, :][None, :, None]
    pol.signal_xindex_moment_0[:] /= f1(pol.angle[:])**2
    pol.signal_xindex_moment_0[:] /= f2(raw_angle_to_physical(pol.angle[:]) % 360)
    pol.signal_xindex_moment_0.normalize()
    pol.transform("ydistance", "xdistance", "angle")
    pol.print_tree()
    pol.moment("angle", channel="signal_xindex_moment_0", moment=0)  # signal at all angles
    pol.copy(parent=shg_pol, name="polarization")


# --- Reflection spectroscopy ---------------------------------------------------------------------


if all_import:
    # --- 20x
    um_per_pixel = 0.3  # 20x
    mpix = 1333  # middle pixel

    reflection = root.create_collection("reflection")
    p = data_dir / "reflection_microspectroscopy"

    # --- --- lamp
    temp = wt.data.from_Solis(p / '20x image.asc').split("yindex", [850, 1300])[1].split("xindex", [1050, 1500])[1]
    temp.create_variable(
        "ydist", values=(temp.yindex[:] - temp.yindex[:].mean()) * um_per_pixel, units="um"
    )
    temp.create_variable(
        "xdist", values=(temp.xindex[:] - mpix) * um_per_pixel, units="um"
    )
    temp.transform("xdist", "ydist")
    temp.copy(parent=reflection, name="x20_image")

    # --- --- spectrum
    temp = wt.data.from_Solis(p / '20x spectrum.asc').split("yindex", [850, 1300])[1]
    temp.create_variable(
        "energy", 
        values=np.linspace(440.1, 800.9, temp.wm[:].size).reshape(temp.wm.shape),
        units="nm"
    )
    temp.energy.convert("eV")
    temp.create_variable(
        "ydist",
        values=(temp.yindex[:] - temp.yindex[:].mean()) * um_per_pixel,
        units="um"
    )
    temp.smooth((10, 0))

    substrate_low = temp.split("yindex", [900])[0].signal[:].mean(axis=1)[:, None]
    substrate_high = temp.split("yindex", [1250])[1].signal[:].mean(axis=1)[:, None]
    # interpolate between top and bottom substrate spectra 
    z = temp.yindex[:].copy()
    s = (z - 900) / 350
    s[s>1] = 1
    s[s<0] = 0
    substrate = (1-s) * substrate_low + s * substrate_high

    temp.create_channel(
        "contrast", values=(temp.signal[:] - substrate) / substrate, signed=True
    )

    temp.copy(parent=reflection, name='x20')

    # --- 100x ------------------------------------------------------------------------------------
    um_per_pixel = 0.06
    tweak_contrast = True  # tweak the reflection intensities to match at substrate
    y0 = 1055  # pixel that marks the heterostructure transition (y=0)
    x0 = 1338  # slit center pixel

    img_sig = wt.data.from_Solis(
        os.path.join(p, "target area 100x image.asc")
    )
    # image is cropped:  set correct y index values
    bottom = int(img_sig.attrs["{left,right,bottom,top}"].split(",")[2])
    y = img_sig["yindex"]  # the x reference is necessary to use setitem (*=, +=, etc.) syntax on a variable
    y += bottom 

    spectrum_sig = wt.data.from_Solis(
        os.path.join(p, "target area 100x spectrum.asc")
    )
    # slight vertical axis adjustment to account for grating misalignment (known issue)
    y = spectrum_sig["yindex"]
    y -= 12

    spectrum_blank = wt.data.from_Solis(
        os.path.join(p, "blank 100x spectrum.asc")
    )

    scalar = 0.88 if tweak_contrast else 1.
    spectrum_sig.create_channel(
        name="contrast",
        # ddk: adding tweak factor to account for lamp fluctuations
        # guiding principle is to minimize delta R at substrate region
        values=scalar * spectrum_sig.signal[:] / spectrum_blank.signal[:] - 1,
        signed=True
    )

    spectrum_sig.smooth(5, channel="contrast")

    img_sig.create_variable(name="y", values=(img_sig.yindex[:]-y0) * um_per_pixel, units="um")
    img_sig.create_variable(name="x", values=(img_sig.xindex[:]-x0) * um_per_pixel, units="um")
    spectrum_sig.create_variable(name="y", values=(spectrum_sig.yindex[:]-y0) * um_per_pixel, units="um")

    spectrum_sig = spectrum_sig.split("wm", [1.7, 2.5], units="eV")[1]
    spectrum_sig = spectrum_sig.split("y", [-30, 30])[1]
    img_sig = img_sig.split("y", [-30, 30])[1]
    img_sig = img_sig.split("x", [-30, 30])[1]

    img_sig.transform("x", "y")
    spectrum_sig.transform("wm", "y")
    spectrum_sig.convert("eV")

    mos2, ws2 = spectrum_sig.split("y", [-0.36]).values()
    mos2s = mos2.split("wm", [1.9, 2.2], units="eV")
    ws2s = ws2.split("wm", [2, 2.4], units="eV")
    ares = []
    bres = []
    for split in [mos2s, ws2s]:
        a, b = [split[0], split[1]]
        if split == ws2s:  # try to avoid room light feature
            bs = b.split("wm", [2.266, 2.3], units="eV")
            b = wt.data.join([bs[0], bs[2]])
            b.convert("eV")
        ares.append(a.wm[:][a.contrast[:].argmin(axis=0)][:, 0])
        bres.append(b.wm[:][b.contrast[:].argmin(axis=0)][:, 0])
    ares = spectrum_sig.create_channel(name="ares", values=np.concatenate(ares)[None,:])
    bres = spectrum_sig.create_channel(name="bres", values=np.concatenate(bres)[None,:])

    spectrum_sig = spectrum_sig.copy(name="x100", parent=reflection)
    img_sig = img_sig.copy(name="x100_image", parent=reflection)


# --- Finish --------------------------------------------------------------------------------------


if all_import:
    root.print_tree(depth=2)
    p = "heterostructure.wt5"
    root.save(here / p, overwrite=True)

    if talkback:
        print("DONE! Total time", time.time() - tstart)
