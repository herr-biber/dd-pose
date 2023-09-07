import numpy as np
import pytest
import transformations as tr
import zipfile

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


def test_zip_file_predictor_empty_zip(tmp_path):
    # empty zip
    zip_file_name = tmp_path / "a.zip"
    di_dict = {
        "subject": 0,
        "scenario": 1,
        "humanhash": "a-b-c-d"
    }

    with zipfile.ZipFile(zip_file_name.as_posix(), 'w') as zip_file:
        pass

    zfp = ZipFilePredictor(zip_file_name.as_posix(), di_dict)
    assert isinstance(zfp.predictions, StampedTransforms)
    assert isinstance(zfp.metadata, dict)


def test_zip_file_predictor_empty_json(tmp_path):
    # empty json
    zip_file_name = tmp_path / "a.zip"
    di_dict = {
        "subject": 0,
        "scenario": 1,
        "humanhash": "a-b-c-d"
    }

    with zipfile.ZipFile(zip_file_name.as_posix(), 'w') as zip_file:
        zip_file.writestr('subject-00/scenario-01/a-b-c-d/t-camdriver-head-predictions.json', "")

    with pytest.raises(ValueError) as excinfo:
        zfp = ZipFilePredictor(zip_file_name.as_posix(), di_dict)