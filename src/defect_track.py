#%%

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from strained_apfc.manage import utils
from strained_apfc.manage import read_write as rw
from strained_apfc.calculations import defect_detection

sim_path = ""

defect_radius_extension = 10
expected_number_of_defects = 2
sigma_mult = 10
filter_0 = False

defect_indeces = range(1, 20)

dpi = 250
color_map_name = "plasma"

#%% Calculations
#######

config = utils.get_config(sim_path)

if defect_indeces is None:
    defect_indeces = range(int(config["numT"] / config["writeEvery"]) - 1)

defects = []
time_indeces = []
velocity_fields = []

for time_index, i in enumerate(defect_indeces):

    eta_path = f"{sim_path}/eta_files/0.0000/"

    G = np.array(config["G"])
    eta_count = len(config["G"])

    x = np.linspace(-config["xlim"], config["xlim"], config["numPtsX"])
    xm, ym = np.meshgrid(x, x)
    r = np.array([xm, ym])

    etas = rw.read_all_etas_at_line(
        eta_path, i, config["numPtsX"], config["numPtsY"], eta_count, complex
    )

    phi = np.zeros(etas[0].shape)
    for i in range(etas.shape[0]):
        phi += np.real(etas[i] * np.conj(etas[i]))
    phi *= 2

    defect_pos = defect_detection.get_defects_center_by_minimum(
        phi, xm, ym, expand_radius=defect_radius_extension
    )

    if defect_pos.shape != (expected_number_of_defects, 2):
        continue

    defects.append(defect_pos)
    time_indeces.append(time_index)

dt = config["writeEvery"] * config["dt"]

distances = []
for i in range(len(defects)):

    dist = (defects[i][0, 0] - defects[i][1, 0]) ** 2
    dist += (defects[i][0, 1] - defects[i][1, 1]) ** 2
    dist = np.sqrt(dist)

    distances.append(dist)

velocities = []
for i in range(1, len(defects)):

    distm1 = (defects[i - 1][0, 0] - defects[i - 1][1, 0]) ** 2
    distm1 += (defects[i - 1][0, 1] - defects[i - 1][1, 1]) ** 2
    distm1 = np.sqrt(distm1)

    vel = np.abs(distances[i] - distm1)
    vel /= dt * (time_indeces[i] - time_indeces[i - 1])

    velocities.append(vel)

velocities = np.array(velocities)
times = np.array(time_indeces[:-1]) * dt
times_full = np.array(time_indeces) * dt

#%% Plots

#############
### PLOTS ###
#############

cmap = cm.get_cmap(color_map_name)

### Positions ###

fig, ax = plt.subplots(1, 1, dpi=dpi)

ax.set_aspect("equal")
ax.set_xlim([-config["xlim"], config["xlim"]])
ax.set_ylim([-config["xlim"], config["xlim"]])

defects_np = np.array(defects)

colors = np.arange(defects_np.shape[0]) / defects_np.shape[0]
colors = np.array([c for c in colors for _ in (0, 1)])

ax.scatter(defects_np[:, :, 0], defects_np[:, :, 1], c=colors, cmap=cmap)

# for i, d in enumerate(defects):
#    color = time_indeces[i] / len(time_indeces)
#    print(color)
#    ax.scatter(d[:, 0], d[:, 1], c=[color, color], cmap=cmap)

### Distances ###

fig, ax = plt.subplots(1, 1, dpi=dpi)
ax.scatter(distances[1:], velocities)
ax.set_xlabel("distances")
ax.set_ylabel("velocities")

plt.show()
