from typing import Union
import numpy as np
import scipy


def A(config: dict) -> float:
    """
    Args:
        config (dict): config object

    Returns:
        float: :math:`A` parameter in eq. :eq:`eqn:apfc_flow_constants`.
    """
    return config["Bx"]


def B(config: dict, n0: Union[None, np.array] = None) -> Union[float, np.array]:
    """
    Args:
        config (dict): config object
        n0 (None | np.array, optional): average density. If left
            :code:`None`, the :code:`n0` key in the config will be used.
            Defaults to None.

    Returns:
        Union[float, np.array]: :math:`B` parameter in eq. :eq:`eqn:apfc_flow_constants`.
    """

    if n0 is None:
        n0 = config["n0"]

    t = config["t"]
    v = config["v"]
    dB0 = config["dB0"]

    return dB0 - 2 * t * n0 + 3 * v * n0**2


def C(config: dict, n0: Union[None, np.array] = None) -> Union[float, np.array]:
    """
    Args:
        config (dict): config object
        n0 (None | np.array, optional): average density. If left
            :code:`None`, the :code:`n0` key in the config will be used.
            Defaults to None.

    Returns:
        Union[float, np.array]: :math:`C` parameter in eq. :eq:`eqn:apfc_flow_constants`.
    """

    if n0 is None:
        n0 = config["n0"]

    t = config["t"]
    v = config["v"]

    return -t + 3 * v * n0


def D(config: dict) -> float:
    """
    Args:
        config (dict): config object

    Returns:
        float: :math:`D` parameter in eq. :eq:`eqn:apfc_flow_constants`.
    """
    return config["v"]


def E(config: dict, n0: Union[None, np.array] = None) -> Union[float, np.array]:
    """
    Args:
        config (dict): config object
        n0 (None | np.array, optional): average density. If left
            :code:`None`, the :code:`n0` key in the config will be used.
            Defaults to None.

    Returns:
        Union[float, np.array]: :math:`E` parameter in eq. :eq:`eqn:apfc_flow_constants`.
    """

    if n0 is None:
        n0 = config["n0"]

    t = config["t"]
    v = config["v"]
    dB0 = config["dB0"]
    Bx = config["Bx"]

    return (Bx + dB0) * n0**2 / 2 - t * n0**3 / 3 + v * n0**4 / 4
