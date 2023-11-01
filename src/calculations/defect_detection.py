"""
The goal of this module is to provide functions for detecting defects, or
generally large variations in 2d fields.

The main function of this module is :py:func:`get_defects_center`, which
provides a very simple interface for getting xy coordinates for every single
defect found in the input array. The input array should have roughly the same
or similar values everywhere, except at points where the defect centers should
be extracted.

Additionally the module contains functions to mark arrays, normalize them
and algorithms for extracting islands.

Example
-------

The input array:

.. code::

      5 | 0 0 0 0 0
      4 | 0 1 0 0 0
    y 3 | 0 0 0 0 0
      2 | 0 1 1 0 0
      1 | 0 1 1 0 0
          ---------
          1 2 3 4 5
              x

Will result in the following coordinates:

.. code:: python

    >>> x = np.array([1, 2, 3, 4, 5])
    >>> xm, ym = np.meshgrid(x, x)
    >>> get_defects_center(arr, xm, ym)
        [[2.5, 1.5]
         [2. , 4. ]]
"""

import math

import scipy
import numpy as np


def get_nn(arr: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Gets the Nerest Neighbors of the cell :code:`arr[i,j]`.
    Uses periodic boundary conditions.

    Args:
        arr (np.ndarray): 2D arry
        i (int): row index
        j (int): column index

    Returns:
        np.ndarray: 3x3 2D matrix with the neirest neighbors arround
            :code:`arr[i,j]`
    """

    row_l = i - 1
    row_u = (i + 1) % arr.shape[0]
    col_l = j - 1
    col_u = (j + 1) % arr.shape[1]

    s = np.array(
        [
            [arr[row_l, col_l], arr[row_l, j], arr[row_l, col_u]],
            [arr[i, col_l], arr[i, j], arr[i, col_u]],
            [arr[row_u, col_l], arr[row_u, j], arr[row_u, col_u]],
        ]
    )

    return s


def mark(arr: np.ndarray) -> np.ndarray:
    """
    Generates an array with 1s where the array is over half to the maximum
    value. 0s everywhere else.

    Args:
        arr (np.ndarray):

    Returns:
        np.ndarray:
    """

    is_over_half = arr >= np.max(arr) / 2
    is_over_half = is_over_half.astype(int)

    return is_over_half


def abs_normalize(arr: np.ndarray) -> np.ndarray:
    """
    Scales and translates the given array into an intervall of
    :math:`[0, 1]`.

    Args:
        arr (np.ndarray):

    Returns:
        np.ndarray:
    """

    arr = arr.astype(float)

    arr = np.abs(arr)
    arr -= np.min(arr)
    arr /= np.max(arr)

    return arr


def grad_scale(arr: np.ndarray):

    arr = abs_normalize(arr)
    dx, dy = np.gradient(arr)
    arr = dx**2 + dy**2
    arr = abs_normalize(arr)

    return arr


def expand_mark_by_radius(arr: np.ndarray, radius: float, dx: float) -> np.ndarray:
    """
    Given a marked array (an array with only 1s and 0s), the islands of 1s
    will be extended by the given radius.

    .. warning::

        This function may be computationally expensive as it involves
        a convolution. It gets more expensive the bigger the radius is.

    Args:
        arr (np.ndarray): marked array
        radius (float):
        dx (float): grid spacing

    Returns:
        np.ndarray: expanded marked array
    """

    num_pts = math.ceil(radius / dx)
    conv_arr = np.ones((num_pts, num_pts))

    out = scipy.signal.convolve2d(
        arr, conv_arr, mode="same", boundary="fill", fillvalue=0
    )
    out = abs_normalize(out)
    out = mark(out)

    return out


def mark_islands(arr: np.ndarray) -> np.ndarray:
    """
    Given a 2D array, this functions marks individual "islands".
    Islands are collections of entries in the array, which are completely
    surrounded by zeros.

    .. warning:: Periodic Boundary Conditions apply!

    Args:
        arr (np.ndarray):

    Returns:
        np.ndarray: Has the same shape as the input array. Each island
        will have a unique integer value. Each cell in the array
        will have the value corresponding to the island. 0 means no island.
        Values other than 0 have no specific meaning, other than
        differentiating the islands.

    Example:

        .. code::

            >>> arr = np.array([
                    [1, 1, 0, 0, 1, 0],
                    [0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                ])
            >>> arr
            [[1 1 0 0 1 0]
             [0 1 0 0 1 0]
             [0 0 1 1 0 0]
             [0 0 0 0 0 1]
             [0 0 0 1 0 0]
             [1 0 0 0 0 0]]
            >>> out = mark_islands(arr)
            >>> out
            [[2 2 0 0 2 0]
             [0 2 0 0 2 0]
             [0 0 2 2 0 0]
             [0 0 0 0 0 3]
             [0 0 0 4 0 0]
             [2 0 0 0 0 0]]

    """

    control = np.zeros(arr.shape, dtype=int)
    counter = 1

    for row_i in range(arr.shape[0]):
        for col_j in range(arr.shape[1]):

            if arr[row_i, col_j] == 0:
                continue

            nn = get_nn(control, row_i, col_j)
            island_numbers = nn[nn != 0]

            if island_numbers.shape[0] == 0:
                control[row_i, col_j] = counter
                counter += 1
                continue

            control[row_i, col_j] = island_numbers[0]
            for n in island_numbers[1:]:
                control[control == n] = island_numbers[0]

    return control


def extract_islands(arr: np.ndarray) -> tuple[np.ndarray]:
    """
    Splits the input into arrays containing only one unique value.
    All other values will be set to 0.

    Args:
        arr (np.ndarray):

    Returns:
        tuple[np.ndarray]: The uple contains one entry for each unique value.
        The values will be given in increasing order.
    """

    island_numbers: np.ndarray = arr[arr != 0].flatten()
    island_numbers: np.ndarray = np.unique(island_numbers)
    island_numbers = np.sort(island_numbers)

    island_list = []
    for num in island_numbers:

        island = arr.copy()
        island[island != num] = 0
        island_list.append(island)

    return tuple(island_list)


def get_center_of_mass(
    arr: np.ndarray, xm: np.ndarray, ym: np.ndarray
) -> tuple[float, float]:
    """
    Calculates the weighted centroid of the array.

    Args:
        arr (np.ndarray): weights
        xm (np.ndarray): x coordinates as a meshgrid
        ym (np.ndarray): y coordinates as a meshgrid

    Returns:
        tuple[float, float]: (x, y) coordinates
    """

    arr_sum = arr.sum().astype(float)

    x = np.sum(xm * arr) / arr_sum
    y = np.sum(ym * arr) / arr_sum

    return (x, y)


def get_defects_center(
    arr: np.ndarray, xm: np.ndarray, ym: np.ndarray, expand_radius: float | None = None
) -> np.ndarray:
    """
    Extracts the position of defects / variations in the original array.

    Args:
        arr (np.ndarray): The array to check for defects / variations.
            The array should have roughly the same value everywhere, but the
            points of intereset which should be extracted.
        xm (np.ndarray): x coordinates as a meshgrid.
        ym (np.ndarray): y coordinates as a meshgrid.
        expand_radius (float | None, optional): All found defects within this
            radius will be counted as one defect. Value can be None if no
            such filter should be applied. Defaults to None.

    Returns:
        np.ndarray: The positions of n defects. Is an array with shape
            (n, 2). The second dimension is for (x, y) coordinates.
    """

    marked_arr = grad_scale(arr)
    marked_arr = mark(marked_arr)

    if expand_radius is not None:
        dx: float = np.diff(xm[0, :])[0]
        marked_arr = expand_mark_by_radius(marked_arr, expand_radius, dx)

    full_island_arr = mark_islands(marked_arr)
    islands = extract_islands(full_island_arr)

    defect_centers = np.zeros((len(islands), 2))
    for i, island in enumerate(islands):

        pos = get_center_of_mass(island, xm, ym)
        defect_centers[i, 0] = pos[0]
        defect_centers[i, 1] = pos[1]

    return defect_centers
