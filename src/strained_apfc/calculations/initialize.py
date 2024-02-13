"""
This module is a collection of functions needed to initialize different parts
of the simulation.

F.e. :py:func:`single_grain` initializes the domain to have a grain in the
center. :py:func:`init_eta_height` can compute the inital height of the
amplitudes and :py:func:`line_defect_x` and :py:func:`line_defect_y` calculate
the displacement for a given :ref:`burgers vector <ch:burgers_vector>`.
"""

import numpy as np

from manage import read_write as rw

###################################
### Domain Shape / Initializers ###
###################################


def tanhmin(radius: np.array, eps: float) -> np.array:
    """
    Applies the tanh minimization function for initialization.

    .. math::

        \\frac{1}{2} \\left[
            1 + \\text{tanh} \\left(- \\frac{3r}{\\varepsilon} \\right)
        \\right]

    Args:
        radius (np.array): :math:`r` radius
        eps (float): :math:`\\varepsilon` a measure for the intreface width

    Returns:
        np.array: resulting tanhmin
    """
    return 0.5 * (1.0 + np.tanh(-3.0 * radius / eps))


def single_grain(xm: np.array, ym: np.array, config: dict) -> np.array:
    """
    Creates a 2d array with a circular grain in the middle.
    The grain will be generated using the
    :py:meth:`calculations.initialize.tanhmin` function.
    The radius, interface width and height are read from the config.

    .. image:: figures/single_grain_example.png

    Args:
        xm (np.array): x-meshgrid
        ym (np.array): y-meshgrid
        config (dict): config object. Explicitely used entries are:
            `initRadius`, `interfaceWidth`, `initEta`

    Returns:
        np.array: grain mesh
    """

    radius: np.array = np.sqrt(xm**2 + ym**2) - config["initRadius"]
    radius = tanhmin(radius, config["interfaceWidth"])

    return radius * config["initEta"]


def center_line(xm: np.array, config: dict) -> np.array:
    """
    Uses the :py:meth:`calculations.initialize.tanhmin` function to
    initialize a vertical line in the middle of the domain.
    The radius, interface width and initial height are read from the config.

    .. image:: figures/center_line_example.png

    Args:
        xm (np.array): x-meshgrid
        config (dict): config; Explicitely used entries are:
            `initRadius`, `interfaceWidth`, `initEta`

    Returns:
        np.array: center line meshgrid
    """

    eta = tanhmin(np.abs(xm) - config["initRadius"], config["interfaceWidth"])
    return eta * config["initEta"]


######################
### Load from File ###
######################


def load_eta_from_file(shape: tuple[int], config: dict, eta_i: int) -> np.array:
    """
    Searches the `sim_path` in the config file for the
    `"out_{eta_i}.txt"` file, and loads its last entry into an
    array.

    Args:
        shape (tuple[int]): The shape of the resulting array.
        config (dict): config object. Explicitely used entries are:
            `simPath`
        eta_i (int): the index of which eta should be read.

    Returns:
        np.array: resulting array
    """

    return load_from_file(config, f"out_{eta_i}.txt", shape)


def load_n0_from_file(shape: tuple[int], config: dict) -> np.array:
    """
    Searches the `simPath` in the config file for the `"n0.txt"` file,
    to load its last entry.

    Args:
        shape (tuple[int]): The shape of the resulting array.
        config (dict): The config object. Explicitly used keys are:
            `simPath`

    Returns:
        np.array: resulting n0 array.
    """

    return load_from_file(config, "n0.txt", shape)


def load_velocity_from_file(shape: tuple[int], config: dict) -> np.ndarray:
    """
    Loads a velocity from a file. Always reads the last line.

    Args:
        shape (tuple[int]): The shape the velocity needs to have. See
            the error section for more details on the limitations.
        config (dict):

    Raises:
        AttributeError: Due to current limitations in read functions the velocity
            can only be read if it has a shape of (2, x, y) and x = y.
        NotImplementedError: If x != y

    Returns:
        np.ndarray:
    """

    # The velocity is saved flattened in one line.
    # All the read functions always assumed to read a simple scalar field,
    # which is why they take an x-dimension and a y-dimension as an input.
    # This does not work for vector fields like the velocity.
    # --> The trick to loading is to read the x and y component flattened.
    # --> read from file with shape (2, x**2)
    # --> then correct to actual shape
    # The limitation is that x and y need to have the same amount of points.

    if len(shape) != 3 or shape[0] != 2:
        raise AttributeError(
            f"The shape for the velocity needs to be (2, x, y) but is {shape}."
        )

    if shape[1] != shape[2]:
        raise NotImplementedError(
            "Can only read velocity arrays when the domain has equal size points."
            " i.e.: In shape (2, x, y): x = y"
        )

    new_shape = (2, shape[1] ** 2)
    arr = load_from_file(config, "velocity.txt", new_shape)
    arr = arr.reshape(shape)

    return arr


def load_from_file(config: dict, file_name: str, shape: tuple[int]) -> np.array:
    """
    Searches the `sim_path` in the config file for the
    `file_name` file, and loads its last entry into an
    array.

    Args:
        shape (tuple[int]): The shape of the resulting array.
        config (dict): config object. Explicitely used entries are:
            `simPath`
        eta_i (int): the index of which eta should be read.

    Returns:
        np.array: resulting array
    """

    out_path = f"{config['simPath']}/{file_name}"
    type_ = detect_type(out_path)
    arr, _ = rw.read_last_line_to_array(out_path, shape[0], shape[1], type_)

    return arr


def detect_type(file_name: str) -> type:
    """
    Detects whether the values contained in the
    file are complex or float.

    Args:
        file_name (str): _description_

    Returns:
        type: _description_
    """

    with open(file_name, "r") as f:
        first_line = f.readline()

    # even if the value saved is 0, the value will be saved as 0j
    # --> j will always be present if the dtype is complex
    if "j" in first_line:
        return complex
    else:
        return float


####################################################
### Initialize Config / calculate Init Variables ###
####################################################


def init_config(config: dict):
    """
    Sets the `A`, `B`, `C`, `D` values based on the
    `n0`, `t`, `v`, `Bx` and `dB0` values in the config.
    These are calculated according to:
    :eq:`eqn:apfc_flow_constants`

    The values are modified in place.

    Args:
        config (dict): config dictionary
    """

    n0 = config["n0"]
    t = config["t"]
    v = config["v"]
    Bx = config["Bx"]
    dB0 = config["dB0"]

    config["A"] = Bx
    config["B"] = dB0 - 2.0 * t * n0 + 3.0 * v * n0**2
    config["C"] = -t + 3.0 * v * n0
    config["D"] = v


def init_eta_height(config: dict, use_pm: bool = False, use_n0: bool = False):
    """
    Sets the `initEta` key in the config based on the
    `t`, `v`, `n0` and `dB0` values.

    Args:
        config (dict): config dictionary
        use_pm (bool, optional): If True the
            :eq:`eqn:n0_init_eta` equation is used to calculate
            the height. The positive variateion is used if
            :math:`t > n_0`. Otherwise the negative version is used.
            If it is false, it is calculated via
            :math:`\\frac{4 t}{45 v}`. Defaults to False.
        use_n0 (bool, optional): If False, n0 will be set to 0. Defaults to False.
    """

    t = config["t"]
    v = config["v"]
    n0 = config["n0"] if use_n0 else 0.0
    dB0 = config["dB0"]

    if not use_pm:
        config["initEta"] = 4.0 * t / (45.0 * v)
        return

    B = dB0 - 2 * t * n0 + 3 * v * n0**2
    C = -t + 3 * v * n0

    sign = -1 if n0 > t else 1

    config["initEta"] = (-C + sign * np.sqrt(C**2 - 15 * v * B)) / (15 * v)


def init_n0_height(
    config: dict,
    x0: float = 0.0,
    ATOL: float = 1e-5,
    RTOL: float = 1e-5,
    MAXITER: int = 1000000,
):
    """
    Initializes the n0 height using the newton method.
    For more information on how this is done, see the
    :ref:`ch:init_n0` section.

    Sets the `n0` parameter in the config, using the
    `initEta`, `dB0`, `Bx`, `t` and `v` values.

    Args:
        config (dict): config dictionary
        x0 (float, optional): The first guess. Defaults to 0.0.
        ATOL (float, optional): Absolute tolerance for stopping criteria
            Defaults to 1e-5.
        RTOL (float, optional): Relative tolerance for stopping criteria.
            Defaults to 1e-5.
        MAXITER (float, optional): Maximum number of iterations. A warning
            will be printed if it is reached.
            Defaults to 1000000.
    """

    phi = 6 * config["initEta"] ** 2
    p = 4 * config["initEta"] ** 3

    a = config["dB0"] + config["Bx"] + 3 * config["v"] * phi
    c = phi * config["t"]
    d = 3 * config["v"] * p

    def f(n0, a, v, t, c, d):
        return a * n0 - t * n0**2 + v * n0**3 + d - c

    def df(n0, a, v, t):
        return a - 2 * t * n0 + 3 * v * n0**2

    xold = x0 + 100.0
    xnew = x0

    k = 0
    while np.abs(xnew - xold) > np.abs(xnew) * RTOL + ATOL:

        xold = xnew
        fval = f(xold, a, config["v"], config["t"], c, d)
        dfval = df(xold, a, config["v"], config["t"])

        xnew = xold - fval / dfval
        k += 1

        if k > MAXITER:
            print("REACHED MAX ITER ON n0 CALCULATION!")
            break

    config["n0"] = -0.5 * xnew


####################
### Line defects ###
####################


def line_defect_x(
    x: float, y: float, poisson_ratio: float, bx: float, offset: np.array = None
) -> float:
    """
    Calculates :math:`u_x` in eq.
    :eq:`eqn:displacement_edge_disloc` for a line defect.

    Args:
        x (float): x coordinate
        y (float): y coordinate
        poisson_ratio (float):
        bx (float): x element of burgers vector
        offset (None|np.aray): offset for dislocation. If None, then offset is 0

    Returns:
        float: x component of displacement for an edge dislocation
    """

    if offset is None:
        offset = np.array([0, 0])

    x = x - offset[0]
    y = y - offset[1]

    denom = 2 * (1 - poisson_ratio) * (x**2 + y**2)
    ret = np.arctan2(y, x) + x * y / denom

    return bx / (2 * np.pi) * ret


def line_defect_y(
    x: float, y: float, poisson_ratio: float, by: float, offset: np.array = None
) -> float:
    """
    Calculates :math:`u_y` in eq.
    :eq:`eqn:displacement_edge_disloc` for a line defect.

    Args:
        x (float): x coordinate
        y (float): y coordinate
        poisson_ratio (float):
        by (float): y element of burgers vector
        offset (None|np.aray): offset for dislocation. If None, then offset is 0

    Returns:
        float: y component of displacement for an edge dislocation
    """

    if offset is None:
        offset = np.array([0, 0])

    x = x - offset[0]
    y = y - offset[1]

    sum1 = 1 - 2 * poisson_ratio
    sum1 /= 4 * (1 - poisson_ratio)
    sum1 *= np.log(x**2 + y**2)

    sum2 = x**2 - y**2
    sum2 /= 4 * (1 - poisson_ratio) * (x**2 + y**2)

    return -by / (2 * np.pi) * (sum1 + sum2)
