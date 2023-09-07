import numpy as np
import pytest
import transformations as tr

from dd_pose.dataset_item import StampedTransforms
from dd_pose.evaluation_helpers import angle_from_matrix, ZipFilePredictor


def test_angle_from_matrix_zero():
    zero_rotation_matrix = np.eye(4)
    assert angle_from_matrix(zero_rotation_matrix) == 0.0


def test_angle_from_matrix_randoms():
    for i in range(100):
        m = tr.random_rotation_matrix()
        angle_expected, _, _ = tr.rotation_from_matrix(m)
        angle = angle_from_matrix(m)
        np.testing.assert_allclose(angle, np.abs(angle_expected))
