import argparse
from typing import Union
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from plots import fields as field_plots
from plots.other import plot_info
from manage import utils
from manage import read_write as rw


matplotlib.use("Qt5Agg")
sns.set_theme()
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

#####################
## SETUP ARGUMENTS ##
#####################

parser = argparse.ArgumentParser(
    prog="Watcher", description="Watches amplitudes and densities."
)

parser.add_argument(
    "sim_path",
    help=(
        "The path where the config file lies,"
        "and where the output should be generated."
    ),
)

parser.add_argument(
    "-ft",
    "--frametime",
    action="store",
    type=int,
    default=500,
    help="The amount of time between frames.",
)

parser.add_argument(
    "-th",
    "--theta",
    action="store",
    type=str,
    default="0.0000",
    help="Which angle theta should be looked at.",
)

parser.add_argument(
    "-pi",
    "--plotindex",
    action="store",
    type=int,
    help=(
        "Plots only a single frame with this index."
        " Can be negative for default python indexing."
    ),
)

parser.add_argument(
    "-s",
    "--save",
    action="store",
    type=str,
    default=None,
    help=(
        "Will save the animation / picture in the sim_path with this name"
        "and the appropriate file extension."
    ),
)

parser.add_argument(
    "-dpi", "--dpi", action="store", type=int, default=300, help="DPI of the plots."
)

parser.add_argument(
    "-pe",
    "--plotevery",
    action="store",
    type=int,
    default=1,
    help="Skips this many entries per frame.",
)

parser.add_argument(
    "-vv",
    "--varyvalue",
    action="store",
    type=str,
    help=(
        "If the directory contains vary parameters, this specifies the folder."
        "Default is the lowest value found."
    ),
)

args = parser.parse_args()

####################
## PREP VARIABLES ##
####################

config = utils.get_config(args.sim_path)

out_dir = utils.make_path_arg_absolute(args.sim_path)
if config.get("vary", False):

    out_dir = f"{out_dir}/{config['varyParam']}"

    vary_dir_name = utils.get_vary_val_dir_name(utils.get_vary_values(config)[0])
    if args.varyvalue is not None:
        vary_dir_name = args.varyvalue

    out_dir = f"{out_dir}/{vary_dir_name}"
    config = utils.get_config(out_dir)

sim_path = f"{out_dir}/eta_files/{args.theta}"

eta_count = len(config["G"])

eta_it = rw.EtaIterator(
    sim_path,
    config["numPtsX"],
    config["numPtsY"],
    eta_count,
    float,
    config["simType"] == "n0",
)

line_count = eta_it.count_lines()

eta_it = iter(eta_it)

is_single_plot = args.plotindex is not None
plot_i = args.plotindex if is_single_plot else 0

if plot_i < 0:
    plot_i = line_count - np.abs(plot_i)

include_n0 = config["simType"] == "n0"
is_1d = config["numPtsY"] <= 1

#################
## PREP FIGURE ##
#################

if include_n0:
    fig = plt.figure(figsize=(10, 8))
    ax_eta = plt.subplot(221)
    ax_n0 = plt.subplot(222)
    ax_info = plt.subplot(223)
else:
    fig = plt.figure(figsize=(10, 5))
    ax_eta = plt.subplot(121)
    ax_n0 = None
    ax_info = plt.subplot(122)

cbar_cax_eta = None
cbar_cax_n0 = None

if not is_1d:

    ax_eta.set_aspect("equal")
    ax_info.set_aspect("equal")

    div_eta = make_axes_locatable(ax_eta)
    cbar_cax_eta = div_eta.append_axes("right", "5%", "5%")

    if include_n0:

        div_n0 = make_axes_locatable(ax_n0)
        cbar_cax_n0 = div_n0.append_axes("right", "5%", "5%")
        ax_n0.set_aspect("equal")

# This is supposed to be the iterating index for the plots.
# It is wrapped in a list so we can pass a mutable object to the
# FuncAnimation(..) function. This way we can increment the index there.
index = [plot_i]

##########
## PLOT ##
##########


def plot(
    frame,
    config: dict,
    ax_eta: plt.Axes,
    ax_info: plt.Axes,
    theta: Union[str, float],
    index: list[int],
    max_index: int,
    ax_n0: Union[plt.Axes, None] = None,
    cbar_cax_eta=None,
    cbar_cax_n0=None,
    plot_every=1,
):

    if index[0] > max_index:
        index[0] = 0

    etas = rw.read_all_etas_at_line(
        sim_path, index[0], config["numPtsX"], config["numPtsY"], eta_count, complex
    )

    if include_n0:
        n0 = rw.read_arr_at_line(
            f"{sim_path}/n0.txt", index[0], config["numPtsX"], config["numPtsY"]
        )
    else:
        n0 = None

    ax_eta.cla()
    field_plots.plot_eta(etas, config, ax_eta, cbar_cax_eta)

    if n0 is not None and ax_n0 is not None:
        ax_n0.cla()
        field_plots.plot_n0(n0, config, ax_n0, cbar_cax_n0)

    ax_info.cla()
    plot_info(config, index[0] * config["writeEvery"], ax_info, float(theta))

    index[0] += plot_every


plot_fargs = (
    config,
    ax_eta,
    ax_info,
    args.theta,
    index,
    line_count - 1,
    ax_n0,
    cbar_cax_eta,
    cbar_cax_n0,
    args.plotevery,
)

if is_single_plot:

    plot(None, *plot_fargs)
    if args.save is not None:
        plt.savefig(f"{out_dir}/{args.save}.png", dpi=args.dpi)

else:

    animation = FuncAnimation(fig, plot, interval=args.frametime, fargs=plot_fargs)
    if args.save is not None:
        animation.save(f"{out_dir}/{args.save}.gif", dpi=args.dpi)

plt.show()
