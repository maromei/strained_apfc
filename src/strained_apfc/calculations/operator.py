import numpy as np


def gradient2D_along_axis2(arr, h, axis):

    axis = int(not bool(axis))

    idx = tuple([slice(None)] * arr.ndim)

    def slc(idx, axis, i):
        idx = list(idx)
        idx[axis] = i
        return tuple(idx)

    deriv = np.zeros(arr.shape, dtype=arr.dtype)
    for i in range(arr.shape[axis]):

        p1_i = (i + 1) % arr.shape[axis]

        deriv[slc(idx, axis, i)] = (
            arr[slc(idx, axis, p1_i)] - arr[slc(idx, axis, i - 1)]
        ) / (2 * h)

    return deriv


def gradient2D_along_axis(arr, h, axis):

    deriv = np.zeros(arr.shape, dtype=arr.dtype)
    if axis == 0:

        length = arr.shape[1]

        for i in range(length):

            lhs = i - 1
            rhs = (i + 1) % length

            deriv[:, i] = arr[:, rhs] - arr[:, lhs]

    else:

        length = arr.shape[0]

        for i in range(length):

            lhs = i - 1
            rhs = (i + 1) % length

            deriv[i, :] = arr[rhs, :] - arr[lhs, :]

    return deriv / (2 * h)


def gradient2D_along_axis3(arr, h, axis):

    axis = int(not bool(axis))
    return np.gradient(arr, h, axis=axis)


def gradient_periodic_BC(arr, h=1, axis=None):

    if axis is not None:

        # axis = int(not bool(axis))

        if axis >= 2:
            raise NotImplementedError("Axis cannot be >= 2. for 2D calculation.")

        d = gradient2D_along_axis(arr, h, axis)
        return d

    else:

        d = (gradient2D_along_axis(arr, h, i) for i in range(len(arr.shape)))

        return tuple(d)
