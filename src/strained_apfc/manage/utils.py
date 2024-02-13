import os
import json

import numpy as np
import pandas as pd


def make_path_arg_absolute(path: str) -> str:
    """
    Takes in a path and transforms it into an absolute one based
    on the current working directory.

    Args:
        path (str): path

    Returns:
        str: absolute path
    """

    if path[0] == "/":  # is already absolute
        return path

    if path[0] == ".":  # asssumes starting with "./path/to/file"
        path = path[2:]

    if path[-1] == "/":  # if folder is passed remove trailing "/"
        path = path[:-1]

    return "/".join([os.getcwd(), path])


def build_sim_info_str(
    config: dict, index: int, theta: float | None = None, add_info: str = ""
) -> str:
    """
    Builds a latex string with additional information.
    The purpose is to include this string into plots to give context.

    Args:
        config (dict): The config dictionary
        index (int): Which index of the simulation is displayed
        theta (float | None, optional): Angle. Defaults to None.
        add_info (str, optional): additional info to dislay.
            It will be appended at the end. Defaults to "".

    Returns:
        str: formatted latex string
    """

    theta_str = ""
    if theta is not None:
        theta_str = f"\n$\\theta = {theta:.4f}$\n"

    is_1d = config["numPtsY"] <= 1

    time = index * config["dt"]

    txt = f"""\
        \\begin{{center}}
        time: {time}
        sim iteration: {index} \\vspace{{0.5em}}
        {theta_str}
        $B^x = {config['Bx']:.4f}, n_0 = {config['n0']:.4f}$
        $v = {config['v']:.4f}, t = {config['t']:.4f}$
        $\\Delta B^0 = {config['dB0']:.4f}, \\beta = {config['beta']:.4f}$
        $\\mathrm{{d}}t = {config['dt']:.4f}$ \\vspace{{0.5em}}
        initial Radius: {config['initRadius']:.4f}
        initial Eta in solid: {config['initEta']:.4f}
        interface width: {config['interfaceWidth']:.4f}
        domain: $[-{config['xlim']}, {config['xlim']}]{'' if is_1d else '^2'}$
        points: {config['numPtsX']} x {config['numPtsY']}
        {add_info}
        \\end{{center}}
    """

    txt = "".join(map(str.lstrip, txt.splitlines(1)))

    return txt


def get_thetas(config: dict, use_div: bool = True, endpoint: bool = True) -> np.array:
    """
    Creates an array of thetas based on the config.
    This function is supposed to supply a consistent way to get the same
    angles.

    Args:
        config (dict): config object. Explicitely used keys are:
            `thetaCount` and `thetaDiv` (if `use_div=True`).
        use_div (bool, optional): Whether to divide the interval
            into `thetaDiv` parts. Where `thetaDiv`. Defaults to True.
        endpoint (bool, optional): Whether to include the last
            value in the range. Defaults to True.

    Returns:
        np.array: The equaly spaced thetas
    """

    if use_div:
        thetas = np.linspace(
            0,
            2.0 * np.pi / config["thetaDiv"],
            config["thetaCount"],
            endpoint=endpoint,
        )
    else:
        thetas = np.linspace(0, 2.0 * np.pi, config["thetaCount"], endpoint=endpoint)

    return thetas


def get_vary_values(config: dict):

    values = config.get("varyValues", None)

    if values is None or len(values) == 0:
        return np.linspace(config["varyStart"], config["varyEnd"], config["varyAmount"])

    return np.array(values)


def create_float_scientific_string(val: float) -> str:
    """
    Takes in a value and generates a scientific notation for it.
    2 digits infront of the decimal place will be shown.

    Args:
        val (float): input value

    Returns:
        str: formatted string
    """

    val_str = f"{val:.2e}"
    val_str = val_str.split("e")
    val_str = f"{val_str[0]} \\cdot 10^{{{val_str[1]}}}"

    return val_str


def get_config(sim_path: str) -> dict:
    """
    Reads the config from the sim_path.

    Args:
        sim_path (str): directory where the :code:`config.json` is stored.

    Returns:
        dict: config
    """

    sim_path = make_path_arg_absolute(sim_path)
    config_path = f"{sim_path}/config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def fill(arr: np.array, div: int, add=False) -> np.array:
    """
    Duplicates and stacks an array :code:`div` times. It is assumed that the
    array is cylcic and has both endpoints in there. This function is usefull
    if f.e. you have angle values in the range :math:`[0, 2 \\pi]`. This function
    will not duplicate the endpoints.

    Args:
        arr (np.array): Array to duplicate
        div (int): How often it should be duplicated
        add (bool, optional): Whether the duplicated arrays
            should add :code:`i * max(arr)` to its value. Defaults to False.

    Returns:
        np.array: The filled array
    """

    o_arr = arr.copy()
    do_add_int = int(add)
    max_ = np.max(o_arr)

    for i in range(1, div):
        add_arr = do_add_int * i * max_
        arr = np.hstack([arr, o_arr[1:] + add_arr])

    return arr


def fill_df(
    df: pd.DataFrame, new_columns: list[str], div: int, add=False
) -> pd.DataFrame:
    """
    Similar to :py:func:`manage.utils.fill`. Just that it is applied to
    every row in the dataframe.

    Args:
        df (pd.DataFrame): original dataframe
        new_columns (list[str]): new columns
        div (int): How often it should be duplicated
        add (bool, optional): Whether the duplicated arrays
            should add :code:`i * max(arr)` to its value. Defaults to False.

    Returns:
        pd.DataFrame: filled df
    """

    new_df = df.apply(fill, axis=1, result_type="expand", args=(div, add))
    new_df.columns = new_columns
    return new_df


def get_positive_range(x, arrs, do_second_entry=False):

    # [0] because one array for each dimension is returned
    is_gte_0_i = (x >= 0).nonzero()[0]
    new_x = x[is_gte_0_i]

    if do_second_entry:

        new_arr = np.zeros((arrs.shape[0], is_gte_0_i.shape[0]))
        for arr_i in range(arrs.shape[0]):
            new_arr[arr_i] = arrs[arr_i][is_gte_0_i]

    else:

        new_arr = arrs[is_gte_0_i]

        """new_arr = np.zeros((arrs.shape[0], arrs.shape[1], is_gte_0_i.shape[0]))
        for arr_i in range(arrs.shape[0]):
            new_arr[arr_i] = arrs[arr_i][:,is_gte_0_i]"""

    return new_x, new_arr


def get_vary_val_dir_name(vary_val: float) -> str:
    return f"{vary_val:.6f}"


def read_vary_vals_from_dir(vary_dir: str, vary_val_key: str) -> np.ndarray[float]:

    all_files = os.listdir(
        vary_dir,
    )

    dirs = []
    for sim_dir in all_files:
        full_path = f"{vary_dir}/{sim_dir}"
        if os.path.isdir(full_path):
            dirs.append(full_path)

    vary_vals = np.zeros(len(dirs))
    for sim_dir_i, sim_dir in enumerate(dirs):
        config = get_config(sim_dir)
        vary_vals[sim_dir_i] = config[vary_val_key]

    vary_vals = np.sort(vary_vals)

    return vary_vals
