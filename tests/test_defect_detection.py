import sys

sys.path.insert(0, "src/")

import numpy as np
from calculations import defect_detection


def test_mark_islands():

    arr = np.array(
        [
            [1, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ]
    )

    expected = np.array(
        [
            [2, 2, 0, 0, 2, 0],
            [0, 2, 0, 0, 2, 0],
            [0, 0, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 3],
            [0, 0, 0, 4, 0, 0],
            [2, 0, 0, 0, 0, 0],
        ]
    )

    arr = defect_detection.mark_islands(arr)
    np.testing.assert_array_equal(arr, expected)


def test_extract_islands():

    arr = np.array(
        [
            [2, 2, 0, 0, 2, 0],
            [0, 2, 0, 0, 2, 0],
            [0, 0, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 3],
            [0, 0, 0, 1, 0, 0],
            [2, 0, 0, 0, 0, 0],
        ]
    )

    exp_1 = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    exp_2 = np.array(
        [
            [2, 2, 0, 0, 2, 0],
            [0, 2, 0, 0, 2, 0],
            [0, 0, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0],
        ]
    )
    exp_3 = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    result = defect_detection.extract_islands(arr)

    assert len(result) == 3
    np.testing.assert_array_equal(result[0], exp_1)
    np.testing.assert_array_equal(result[1], exp_2)
    np.testing.assert_array_equal(result[2], exp_3)
