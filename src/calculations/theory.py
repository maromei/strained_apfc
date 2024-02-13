import numpy as np


def stress_field_non_singular(x, y, v, b, mu, c):

    fac = -mu * b
    fac /= 2 * np.pi * (1 - v)
    fac /= (c**2 + x**2 + y**2) ** 2

    xx = y * (3 * c**2 + 3 * x**2 + y**2)
    xx *= fac

    yy = y * (c**2 - x**2 + y**2)
    yy *= fac

    xy = x * (3 * c**2 + x**2 - y**2)
    xy *= -fac

    return xx, yy, xy
