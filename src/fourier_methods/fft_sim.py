import json
import os
import sys
import shutil
import numpy as np
import multiprocessing as mp

from manage import utils
from calculations import initialize
from .base_runner import FFTBaseSim
from .n0_runner import FFTN0Sim
from .hapfc_runner import FFTHydroAPFCSim

from .parameter_sets import PARAM_SETS


#: Saves the precentages of progress
#: each thread is at. Will be set by
#: :py:func:`sim.fft_sim.create_thread_list`.
progress_list = []


def are_processes_running(thread_list: list[mp.Process]) -> bool:
    """
    Checks if the processees are still running

    Args:
        thread_list (list[mp.Process]): list of processes

    Returns:
        bool: is at least one of them still running
    """

    for i in range(len(thread_list)):
        if thread_list[i].is_alive():
            return True

    return False


def print_progress_string():
    """
    Takes the global :py:data:`sim.fft_sim.progress_list` and prints
    the values to a string.
    """

    prog_strs = [f"{n:5.1f}%" for n in progress_list]
    prog_strs = " ".join(prog_strs)
    sys.stdout.write(f"{prog_strs}\r")
    sys.stdout.flush()


def initialize_sim_threads(
    sim_path: str,
    calcdb0: bool,
    calc_init_eta: bool,
    calc_init_n0: bool,
    thread_count: int,
    continue_sim: bool,
    param_set: int | None,
    keep_dir: bool,
) -> list[mp.Process]:
    """
    Initializes a simulation with its threads

    Args:
        sim_path (str): folder where the config is located and
            the sim output should be put in.
        calcdb0 (bool): Whether equilibrium :math:`\Delta B^0` should be
            calculated via :math:`\\frac{8 t^2}{135 v}`
        calc_init_eta (bool): Should the initial amplitude be calculated.
            It will always use the
            :py:func:`calculations.initialize.init_eta_height` function with
            :code:`use_pm=True` and :code:`use_n0=False`.
        calc_init_n0 (bool): Whether the average density for the given
            initial amplitude height should be calculated.
        thread_count (int): How many threads should be created.
        continue_sim (bool): Whether the last values should be read from a
            file.
        paramet_set (int | None): If it is not None, the
            :py:data:`sim/parameter_sets.PARAM_SETS`
            with the corresponding index will be applied to the
            config.
        keep_dir (bool): If the simulation is not continued, the directories
            are usually deleted. This options keeps this from happening.

    Returns:
        list[mp.Process]: The list of processes.
    """

    ################
    ## GET CONFIG ##
    ################

    sim_path_ = utils.make_path_arg_absolute(sim_path)
    config_path = f"{sim_path_}/config.json"
    config = utils.get_config(sim_path)

    ###############
    ## PARAM SET ##
    ###############

    if param_set is not None:
        for key in PARAM_SETS[param_set]:
            config[key] = PARAM_SETS[param_set][key]

    ######################
    ## HANDLE ARGUMENTS ##
    ######################

    if calcdb0:
        dB0 = 8.0 * config["t"] ** 2 / (135.0 * config["v"])
        config["dB0"] = dB0

    if calc_init_eta:
        initialize.init_eta_height(config, use_pm=True, use_n0=True)

    if calc_init_n0:
        initialize.init_n0_height(config)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    #####################
    ## SIMULTAION VARS ##
    #####################

    thetas = utils.get_thetas(config)

    eta_path = f"{sim_path}/eta_files"
    if not continue_sim and os.path.exists(eta_path) and not keep_dir:
        shutil.rmtree(eta_path)

    if not os.path.exists(eta_path):
        os.makedirs(eta_path)

    return create_thread_list(
        config, thetas, thread_count, eta_path, continue_sim, config["simType"]
    )


def create_thread_list(
    config: dict,
    thetas: np.array,
    thread_count: int,
    eta_path: str,
    continue_sim: bool,
    sim_type: str,
) -> list[mp.Process]:
    """
    Creates the process list for a simulation. It will
    distribute all thetas onto the given number of processes.

    The funciton tobe executed in each thread is
    :py:func:`sim.fft_sim.theta_thread`.

    Args:
        config (dict): config object
        thetas (np.array): list of all angles
        thread_count (int): how many processes should be created.
        eta_path (str): the path in which all output files should be created.
            Each angle will create one sub folder.
        continue_sim (bool): Whether the simulation should continue
            from an existing state.
        sim_type (str): Type of the simulation. Currently checked are
            :code:`n0`, otherwise the base sim will be run.

    Returns:
        list[mp.Process]: List of processes.
    """

    #########################
    ## Setup progress list ##
    #########################

    global progress_list

    mp_manager = mp.Manager()
    progress_list = [0.0 for _ in range(thread_count)]
    progress_list = mp_manager.list(progress_list)

    ##################################
    ## theta list for each progress ##
    ##################################

    theta_lst = [[] for _ in range(thread_count)]

    for theta_i, theta in enumerate(thetas):
        theta_lst[theta_i % thread_count].append(theta)

    #################
    ## thread list ##
    #################

    thread_lst = []
    for i in range(thread_count):
        process_args = (
            np.array(theta_lst[i]),
            config.copy(),
            eta_path,
            continue_sim,
            sim_type,
            i,
        )
        new_process = mp.Process(target=theta_thread, args=process_args)
        thread_lst.append(new_process)

    return thread_lst


def theta_thread(
    thetas: np.array,
    config: dict,
    eta_path: str,
    continue_sim: bool,
    sim_type: str,
    index: int,
):
    """
    One process for one collection of simulations with different angles.

    Args:
        thetas (np.array): thetas to run
        config (dict): config object
        eta_path (str): path where the angle subfolders should be created
        continue_sim (bool): Whether the simulation should continue
            from the last known state.
        sim_type (str): Type of the simulation. Currently checked are
            :code:`n0`, otherwise the base sim will be run.
        index (int): index in a thread list
    """

    ####################
    ## Get Basic Vars ##
    ####################

    step_count: int = config["numT"]
    write_every_i: int = config["writeEvery"]

    total_steps = thetas.shape[0] * step_count

    G_original = np.array(config["G"])
    config_original = config.copy()

    for theta_i, theta in enumerate(thetas):

        #####################################
        ### create direcotry if not exist ###
        #####################################

        theta_path = f"{eta_path}/{theta:.4f}"
        if not os.path.exists(theta_path):
            os.makedirs(theta_path)

        config = config_original.copy()
        config["simPath"] = theta_path

        ##########################
        ### rotate G in config ###
        ##########################

        # fmt: off
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        # fmt: on

        G = G_original.copy()
        for eta_i in range(G.shape[0]):
            G[eta_i] = rot.dot(G[eta_i])
        config["G"] = G.tolist()

        #####################
        ## setup sim class ##
        #####################

        if sim_type == "n0":
            sim = FFTN0Sim(config, continue_sim)
        elif sim_type == "hydro":
            sim = FFTHydroAPFCSim(config, continue_sim)
        else:
            sim = FFTBaseSim(config, continue_sim)

        ignore_first_write = continue_sim

        if not continue_sim:
            sim.reset_out_files(theta_path)

        ###############
        ### run sim ###
        ###############

        if not continue_sim:
            eq_steps = config.get("eqSteps", 0)
            sim.equilibriate(eq_steps)

        if not ignore_first_write:
            sim.write(theta_path)

        for i in range(step_count + 1):  # +1 to get first and last write

            sim.run_one_step()

            should_write = i % write_every_i == 0 and i != 0
            if not should_write:
                continue

            sim.write(theta_path)

            ##########################
            ## modify progress list ##
            ##########################

            curr_i = theta_i * (step_count + 1) + i + 2
            perc = curr_i / total_steps * 100
            progress_list[index] = perc
