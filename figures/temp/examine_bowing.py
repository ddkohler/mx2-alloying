import numpy as np
import matplotlib.pyplot as plt
import WrightTools as wt

x = np.linspace(0,1)

E2a = 1.84
E2b = 1.98
E1a = 1.97
E1b = 2.4

fig, gs = wt.artists.create_figure()
ax0 = plt.subplot(gs[0])

for b_a in [0.21, 0.25, 0.29]:
    for b_b in [0.13, 0.19, 0.25]:
        Exa = E1a * x + E2a * (1-x) - b_a * x * (1-x)
        Exb = E1b * x + E2b * (1-x) - b_b * x * (1-x)
        dExba = Exb - Exa

        ax0.plot(Exa, Exb)

plt.show()