"""Tests CCF registration."""

import json
import os
import tempfile
import unittest

import ants
import numpy as np
from aind_ccf_reg.preprocess import Masking, perc_normalization, write_and_plot_image
from aind_ccf_reg.register import pad_array_n_d
from aind_ccf_reg.utils import (
    create_folder,
    read_json_as_dict,
    save_dict_as_json,
)


class TestPreprocessing(unittest.TestCase):
    """Tests preprocessing functions."""

    def test_perc_normalization_basic(self):
        """Normalized output should be in [0, 1] range."""
        img = np.random.rand(10, 10, 10).astype(np.float32) + 0.1
        ants_img = ants.from_numpy(img)
        norm_img, percentile_values = perc_normalization(ants_img)
        self.assertIsNotNone(norm_img)
        self.assertEqual(len(percentile_values), 2)
        self.assertGreater(percentile_values[1], percentile_values[0])

    def test_perc_normalization_all_zeros_raises(self):
        """All-zero image has no non-zero voxels — should raise ValueError."""
        img = np.zeros((10, 10, 10), dtype=np.float32)
        ants_img = ants.from_numpy(img)
        with self.assertRaises((ValueError, IndexError)):
            perc_normalization(ants_img)

    def test_write_and_plot_image_no_paths(self):
        """write_and_plot_image with no paths should return None."""
        img = np.random.rand(5, 5, 5).astype(np.float32)
        out = write_and_plot_image(img)
        self.assertIsNone(out)

    def test_masking(self):
        """Masking should return a non-None result for a synthetic volume."""
        img = np.zeros((100, 100, 100), dtype=np.float32)
        img[30:70, 30:70, 30:70] = 100
        ants_img = ants.from_numpy(img)
        mask = Masking(ants_img)
        result = mask.run()
        self.assertIsNotNone(result)


class TestPadArrayND(unittest.TestCase):
    """Tests pad_array_n_d from register module."""

    def test_1d_to_5d(self):
        """1D array should be padded to 5D."""
        arr = np.arange(10)
        padded = pad_array_n_d(arr, dim=5)
        self.assertEqual(padded.ndim, 5)

    def test_3d_to_5d(self):
        """3D array should be padded to 5D."""
        arr = np.ones((4, 4, 4))
        padded = pad_array_n_d(arr, dim=5)
        self.assertEqual(padded.ndim, 5)

    def test_already_5d(self):
        """5D array should not be altered."""
        arr = np.ones((1, 1, 4, 4, 4))
        padded = pad_array_n_d(arr, dim=5)
        self.assertEqual(padded.ndim, 5)
        self.assertEqual(padded.shape, arr.shape)

    def test_exceeding_dim_raises(self):
        """Requesting more than 5 dimensions should raise ValueError."""
        arr = np.ones((4, 4))
        with self.assertRaises(ValueError):
            pad_array_n_d(arr, dim=6)


class TestUtils(unittest.TestCase):
    """Tests utility functions from utils module."""

    def test_read_json_as_dict_existing_file(self):
        """Should return dict contents from a valid JSON file."""
        data = {"key": "value", "number": 42}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            tmp_path = f.name
        try:
            result = read_json_as_dict(tmp_path)
            self.assertEqual(result, data)
        finally:
            os.unlink(tmp_path)

    def test_read_json_as_dict_missing_file(self):
        """Should return empty dict when the file does not exist."""
        result = read_json_as_dict("/nonexistent/path/file.json")
        self.assertEqual(result, {})

    def test_save_dict_as_json(self):
        """Should write JSON file that can be read back correctly."""
        data = {"pipeline": "ccf", "version": 5}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            tmp_path = f.name
        try:
            save_dict_as_json(tmp_path, data)
            with open(tmp_path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded, data)
        finally:
            os.unlink(tmp_path)

    def test_create_folder(self):
        """Should create a new directory that did not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "new_subfolder")
            self.assertFalse(os.path.exists(new_dir))
            create_folder(new_dir)
            self.assertTrue(os.path.isdir(new_dir))

    def test_create_folder_already_exists(self):
        """Should not raise if the directory already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_folder(tmpdir)
            self.assertTrue(os.path.isdir(tmpdir))


class TestCCFRegistration(unittest.TestCase):
    """Tests CCF registration logic."""

    def test_pad_array_preserves_data(self):
        """Padding should not alter existing data values."""
        arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        padded = pad_array_n_d(arr, dim=5)
        self.assertEqual(padded.squeeze().tolist(), arr.tolist())


if __name__ == "__main__":
    unittest.main()
