import matplotlib.pyplot as plt
import numpy as np

"""
Dice -- brats2020
"""
# species = ("Anand et al", "Messaoudi et al", "Ding et al", "Zhang et al", "Ali et al", "Liu et al", "Wang et al", "Ours")
# penguin_means = {
#     'ET': (0.710, 0.654, 0.615, 0.700, 0.748, 0.573, 0.758, 0.720),
#     'TC': (0.740, 0.681, 0.782, 0.740, 0.748, 0.730, 0.840, 0.843),
#     'WT': (0.880, 0.882, 0.870, 0.880, 0.871, 0.832, 0.899, 0.917),
#     'Mean': (0.777, 0.739, 0.756, 0.773, 0.789, 0.712, 0.819, 0.827),
# }
#
# color_map = ["#04966b", "#f3511c", "#0070b5", "#2c4663"]
# x = np.arange(len(species))  # the label locations
# y = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# width = 0.1  # the width of the bars
# multiplier = 0


"""
HD -- brats2020
"""
# species = ("Anand et al", "Zhang et al", "Ali et al", "Wang et al", "Ours")
# penguin_means = {
#     'ET': (38.310, 38.600, 3.929, 5.296, 3.125),
#     'TC': (32.000, 30.180, 10.090, 5.517, 3.839),
#     'WT': (6.880, 6.950, 9.428, 5.076, 4.307),
#     'Mean': (25.73, 25.24, 7.816, 5.296, 3.757),
# }
#
# color_map = ["#04966b", "#f3511c", "#0070b5", "#2c4663"]
# x = np.arange(len(species))  # the label locations
# y = [5, 10, 15, 20, 25, 30, 35, 40, 45]
# width = 0.1  # the width of the bars
# multiplier = 0


"""
Dice -- Brats2019
"""
# species = ("Zhou et al", "Shen et al", "Li et al", "Liu et al", "Zhu et al", "Ours")
# penguin_means = {
#     'ET': (0.706, 0.724, 0.802, 0.768, 0.838, 0.728),
#     'TC': (0.775, 0.788, 0.834, 0.811, 0.892, 0.850),
#     'WT': (0.897, 0.875, 0.867, 0.893, 0.900, 0.910),
#     'Mean': (0.793, 0.796, 0.834, 0.824, 0.877, 0.829),
# }
#
# color_map = ["#04966b", "#f3511c", "#0070b5", "#2c4663"]
# x = np.arange(len(species))  # the label locations
# y = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# width = 0.1  # the width of the bars
# multiplier = 0

"""
HD -- Brats2019
"""
species = ("Zhou et al", "Shen et al", "Li et al", "Liu et al", "Zhu et al", "Ours")
penguin_means = {
    'ET': (7.400, 5.970, 6.140, 5.180, 3.080, 2.964),
    'TC': (9.300, 11.470, 6.750, 7.231, 5.118, 3.586),
    'WT': (6.700, 9.350, 4.920, 8.219, 5.644, 4.180),
    'Mean': (7.800, 8.930, 5.937, 6.877, 4.614, 3.577),
}

color_map = ["#04966b", "#f3511c", "#0070b5", "#2c4663"]
x = np.arange(len(species))  # the label locations
y = [0, 2, 4, 6, 8, 10]
width = 0.1  # the width of the bars
multiplier = 0


fig, ax = plt.subplots(layout='constrained', figsize=(8, 5))

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=color_map[multiplier])
    # ax.bar_label(rects, padding=3)
    multiplier += 1

size = 12
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('HD(mm)', fontsize=size)
# ax.set_ylabel('Dice', fontsize=size)
ax.set_title('BraTS2019', color='#535353', fontsize=size)
# ax.set_title('BraTS2020', color='#535353', fontsize=size)
ax.set_xticks(x + width, species, rotation=30, color='#535353', fontsize=size)
ax.set_yticks(y, fontsize=size)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)

plt.grid(axis='y', color='#d6d6d6')
plt.show()

