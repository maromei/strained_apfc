import numpy as np
import scipy


def theo_surf_en(thetas: np.array, eps: float, gamma_0: float) -> np.array:
    """
    Calculates the fitted surface energy for the triangular lattice.
    Uses eq. :eq:`eqn:surf_en_theo_triangular` up to first order.

    Args:
        thetas (np.array): angles
        eps (np.array): fitted :math:`\\varepsilon` parameter
        gamma_0 (np.array): fitted :math:`\\gamma_0` parameter

    Returns:
        np.array: fitted surface energy
    """
    return gamma_0 * (1.0 + eps * np.cos(6.0 * thetas))


def theo_surf_en_sec_der(thetas: np.array, eps: float, gamma_0: float) -> np.array:
    """
    Calculates the second derivative of the fitted surface energy
    for the triangular lattice.
    Uses eq. :eq:`eqn:surf_en_theo_triangular` up to first order.

    Args:
        thetas (np.array): angles
        eps (np.array): fitted :math:`\\varepsilon` parameter
        gamma_0 (np.array): fitted :math:`\\gamma_0` parameter

    Returns:
        np.array: fitted surface energy
    """
    return -gamma_0 * eps * 36.0 * np.cos(6.0 * thetas)


def theo_surf_en_der(thetas: np.array, eps: float, gamma_0: float) -> np.array:
    """
    Calculates the first derivative of the  surface energy
    for the triangular lattice.
    Uses eq. :eq:`eqn:surf_en_theo_triangular` up to first order.

    Args:
        thetas (np.array): angles
        eps (np.array): fitted :math:`\\varepsilon` parameter
        gamma_0 (np.array): fitted :math:`\\gamma_0` parameter

    Returns:
        np.array: fitted surface energy
    """
    return -gamma_0 * eps * 6.0 * np.sin(6.0 * thetas)


def wulff_shape_theo(thetas: np.array, eps: float, gamma_0: float) -> np.array:
    """
    Calculates the wulff shape using the fitted surface energy
    parameters for the triangular lattice.
    Uses eq. :eq:`eqn:surf_en_theo_triangular` up to first order.

    Args:
        thetas (np.array): angles
        eps (np.array): fitted :math:`\\varepsilon` parameter
        gamma_0 (np.array): fitted :math:`\\gamma_0` parameter

    Returns:
        np.array: fitted surface energy
    """

    surf = theo_surf_en(thetas, eps, gamma_0)
    surf_der = theo_surf_en_der(thetas, eps, gamma_0)

    x = surf * np.cos(thetas) - surf_der * np.sin(thetas)
    y = surf * np.sin(thetas) + surf_der * np.cos(thetas)

    return x, y


def fit_surf_en(thetas: np.array, surf_en: np.array) -> tuple[float, float]:
    """
    Fits the surface energy to :eq:`eqn:surf_en_theo_triangular` up to
    first order for triangular symmetry.

    Args:
        thetas (np.array): angles
        surf_en (np.array): calculated surface energy

    Returns:
        tuple[float, float]: :math:`\\varepsilon, \\gamma_0`
    """

    if np.sum(surf_en == 0.0) == surf_en.shape[0]:
        return 0.0, 0.0

    gamma0 = (np.max(surf_en) - np.min(surf_en)) / 2
    eps = 1

    if gamma0 < 0.1:
        eps = 0.0

    popt, pcov = scipy.optimize.curve_fit(
        theo_surf_en, thetas, surf_en, p0=[eps, gamma0]
    )

    return popt


def calc_stiffness_fit(thetas: np.array, eps: float, gamma_0: float) -> np.array:
    """
    Calculates the stiffness using fitted surface energy up to first
    order for the triangular symmetry.

    Args:
        thetas (np.array): angles
        eps (float): fit param :math:`\\varepsilon`
        gamma_0 (float): fit param :math:`\\gamma_0`

    Returns:
        np.array: Stiffness
    """

    surf_en = theo_surf_en(thetas, eps, gamma_0)
    return surf_en + theo_surf_en_sec_der(thetas, eps, gamma_0)
