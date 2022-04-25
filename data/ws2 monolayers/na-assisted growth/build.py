import WrightTools as wt
import pathlib
import os
from scipy.ndimage import median_filter


HERE = pathlib.Path(__file__).resolve().parent
UM_PER_PIXEL = 0.3  # 20x


def add_space_axes(data:wt.Data):
    """ spatial units for image axes
    """
    axes = list(_ for _ in data.axis_names)
    if "xindex" in axes:
        data.create_variable("x", values=(data.xindex[:] - data.xindex[:].mean()) * UM_PER_PIXEL)
        axes[axes.index("xindex")] = "x"
    if "yindex" in axes:
        data.create_variable("y", values=(data.yindex[:] - data.yindex[:].mean()) * UM_PER_PIXEL)        
        axes[axes.index("yindex")] = "y"
    data.transform(*axes)


def main():
    """ Import files as wt5
    """
    root = wt.Collection(name="root")

    # reflection contrast info
    spectrum_fname = "20x reflection spectrum {0}.asc"
    temp = wt.data.from_Solis(HERE / spectrum_fname.format("reference"))
    temp = temp.split("yindex", [750, 1750])[1]
    temp.copy(name="spectrum", parent=root)
    for sample in ["spot1", "spot2"]:
        temp = wt.data.from_Solis(HERE / spectrum_fname.format(sample))
        temp = temp.split("yindex", [750, 1750])[1]
        root.spectrum.create_channel(sample, values=temp.signal[:])

    for obj in os.scandir(HERE):
        if obj.name.endswith(".asc"):
            name = obj.name[:-4].replace(" ", "_")
            temp = wt.data.from_Solis(obj.path)            
            if "reflection_spectrum" in name:
                continue
            if "image" in name:
               temp = temp.split("xindex", [700, 1900])[1]
            if "fluorescence" in name:
                if "spectrum" in name:
                    temp = temp.split("wm", [600, 700])[1].split("yindex", [750, 1750])[1]
                else:
                    temp = temp.split("yindex", [100, 2100])[1]
                temp.signal[:] = median_filter(temp.signal[:], size=3)
            temp.copy(name=name, parent=root)

    for data in root.values():
        add_space_axes(data)

    root.print_tree()
    root.save(HERE / "root.wt5", overwrite=True)


if __name__ == "__main__":
    main()

