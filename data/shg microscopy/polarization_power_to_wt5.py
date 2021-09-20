import numpy as np
import WrightTools as wt
import os

__here__ = os.path.abspath(os.path.dirname(__file__))
offset = 49 * np.pi / 180

# data transcribed from (digital) notes
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


if __name__ == "__main__":  # save as wt5
    if False:
        d = wt.Data(name="pump power")
        d.create_variable(name="raw_angle", values=angles)
        d.create_channel(name="power", values=powers)
        d.transform("raw_angle")
        d.print_tree()
        d.save(os.path.join(__here__, "pump_power.wt5"))

    if True:
        import matplotlib.pyplot as plt
        plt.figure()
        ax = plt.subplot(111, projection='polar')
        plt.title("Power vs. Polarization")
        ax.plot(-(angles - offset) * np.pi / 180, powers)
        wt.artists.savefig(os.path.join(__here__, "pump_power.png"))

