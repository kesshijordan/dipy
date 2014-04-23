import numpy as np
import numpy.testing as npt

from dipy.tracking import kfg

def test_clipMask():
    mask = np.ones((10, 10, 10))
    newmask = kfg.clipMask(mask)
    npt.assert_array_equal(mask, 1)
    npt.assert_array_equal(newmask[0], 0)
    npt.assert_array_equal(newmask[:, -1], 0)

    expected = np.zeros((3, 3, 3))
    expected[1, 1, 1] = 1
    mask = np.ones((3, 3, 3))
    newmask = kfg.clipMask(mask)
    npt.assert_array_equal(newmask, expected)

