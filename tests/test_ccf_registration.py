"""Tests for the aind-smartspim-ccf-registration package."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

import dask.array as da
import numpy as np

# ── Pre-mock aind_data_schema before any project imports ─────────────────────
# aind_data_schema 1.0.0 uses Pydantic v2 discriminated-union schemas that
# can fail to build at import time on certain pydantic/Python 3.8 builds.
# sys.modules.setdefault only applies the mock when the module has NOT yet
# been imported, so a working installation is never replaced.
for _mod in [
    "aind_data_schema",
    "aind_data_schema.core",
    "aind_data_schema.core.processing",
]:
    sys.modules.setdefault(_mod, MagicMock())
# ─────────────────────────────────────────────────────────────────────────────

import ants
from aind_ccf_reg.preprocess import (Masking, invert_perc_normalization,
                                     perc_normalization, write_and_plot_image)
from aind_ccf_reg.register import (compute_pyramid, get_pyramid_metadata,
                                   pad_array_n_d)
from aind_ccf_reg.utils import (check_orientation, create_folder,
                                get_channel_translations, get_size,
                                read_json_as_dict, rotate_image,
                                save_dict_as_json)
from skimage.measure import label

# ── Optional: functions from main.py (not part of the installed package) ─────
_CODE_DIR = os.path.join(os.path.dirname(__file__), "..", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_CODE_DIR))
try:
    from main import get_estimated_downsample

    _HAS_MAIN = True
except Exception:
    _HAS_MAIN = False
# ─────────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ants(arr: np.ndarray) -> ants.ANTsImage:
    return ants.from_numpy(arr.astype(np.float32))


def _box_image(size: int = 80, lo: int = 25, hi: int = 55, val: float = 100.0):
    arr = np.zeros((size, size, size), dtype=np.float32)
    arr[lo:hi, lo:hi, lo:hi] = val
    return _ants(arr)


# ---------------------------------------------------------------------------
# Percentile normalisation
# ---------------------------------------------------------------------------


class TestPercentileNormalization(unittest.TestCase):
    """Tests for perc_normalization and invert_perc_normalization."""

    def test_returns_image_and_two_values(self):
        img = _ants(np.random.rand(10, 10, 10) + 0.1)
        norm, pvals = perc_normalization(img)
        self.assertIsNotNone(norm)
        self.assertEqual(len(pvals), 2)

    def test_upper_percentile_greater_than_lower(self):
        img = _ants(np.random.uniform(0.1, 1.0, (10, 10, 10)))
        _, pvals = perc_normalization(img)
        self.assertGreater(pvals[1], pvals[0])

    def test_all_zeros_raises(self):
        """All-zero image has no non-zero voxels → ValueError or IndexError."""
        img = _ants(np.zeros((10, 10, 10)))
        with self.assertRaises((ValueError, IndexError)):
            perc_normalization(img)

    def test_only_nonzero_voxels_used(self):
        """Percentiles computed from non-zero voxels only."""
        arr = np.zeros((10, 10, 10), dtype=np.float32)
        arr[5:, 5:, 5:] = np.random.uniform(0.5, 1.0, (5, 5, 5)).astype(
            np.float32
        )
        _, pvals = perc_normalization(_ants(arr))
        self.assertGreaterEqual(
            pvals[0], 0.4
        )  # must be in the non-zero region

    def test_custom_percentile_range(self):
        """Narrower percentile window produces a smaller [p0, p1] range."""
        img = _ants(np.random.uniform(0.1, 1.0, (20, 20, 20)))
        _, pvals_wide = perc_normalization(img, lower_perc=5, upper_perc=95)
        _, pvals_narrow = perc_normalization(img, lower_perc=40, upper_perc=60)
        self.assertLess(
            pvals_narrow[1] - pvals_narrow[0],
            pvals_wide[1] - pvals_wide[0],
        )

    def test_invert_applies_correct_formula(self):
        """invert_perc_normalization: result = img*(p1-p0) + p0."""
        pvals = [100.0, 500.0]
        img = _ants(np.ones((3, 3, 3)) * 0.5)
        recovered = invert_perc_normalization(img, pvals)
        expected = 0.5 * (500.0 - 100.0) + 100.0  # = 300.0
        np.testing.assert_allclose(np.array(recovered), expected, rtol=1e-5)

    def test_invert_roundtrip(self):
        """Normalise then invert should approximately recover the original."""
        arr = np.random.uniform(0.3, 0.7, (10, 10, 10)).astype(np.float32)
        norm, pvals = perc_normalization(_ants(arr))
        recovered = invert_perc_normalization(norm, pvals)
        # Values inside [p0,p1] should round-trip exactly (no clipping)
        np.testing.assert_allclose(
            np.array(recovered).ravel(),
            (np.array(norm) * (pvals[1] - pvals[0]) + pvals[0]).ravel(),
            rtol=1e-5,
        )


# ---------------------------------------------------------------------------
# write_and_plot_image
# ---------------------------------------------------------------------------


class TestWriteAndPlotImage(unittest.TestCase):
    """Tests for write_and_plot_image."""

    def test_no_paths_returns_none(self):
        result = write_and_plot_image(
            np.random.rand(5, 5, 5).astype(np.float32)
        )
        self.assertIsNone(result)

    def test_writes_nifti_to_disk(self):
        img = _ants(np.random.rand(5, 5, 5))
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            path = f.name
        try:
            write_and_plot_image(img, data_path=path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


class TestMasking(unittest.TestCase):
    """Tests for the Masking class."""

    def test_run_returns_ants_image(self):
        result = Masking(_box_image()).run()
        self.assertIsInstance(result, ants.ANTsImage)

    def test_mask_is_binary(self):
        arr = Masking(_box_image()).run().numpy()
        self.assertTrue(set(np.unique(arr)).issubset({0.0, 1.0}))

    def test_mask_spacing_preserved(self):
        spacing = (0.5, 0.5, 0.5)
        arr = np.zeros((60, 60, 60), dtype=np.float32)
        arr[20:40, 20:40, 20:40] = 100
        img = ants.from_numpy(arr, spacing=spacing)
        mask = Masking(img).run()
        np.testing.assert_allclose(mask.spacing, spacing)

    def test_get_largest_cc_all_zeros_returns_input(self):
        arr = np.zeros((10, 10, 10), dtype=int)
        masking = Masking(_ants(np.zeros((10, 10, 10))))
        result = masking._getLargestCC(arr)
        np.testing.assert_array_equal(result, arr)

    def test_get_largest_cc_selects_largest(self):
        arr = np.zeros((20, 20, 20), dtype=int)
        arr[1:3, 1:3, 1:3] = 1  # small: 2³ = 8 voxels
        arr[10:16, 10:16, 10:16] = 1  # large: 6³ = 216 voxels
        masking = Masking(_ants(np.zeros((20, 20, 20))))
        result = masking._getLargestCC(arr)
        self.assertEqual(
            result[1, 1, 1], 0, "small component should be removed"
        )
        self.assertEqual(
            result[13, 13, 13], 1, "large component should remain"
        )

    def test_get_largest_cc_single_component(self):
        arr = np.zeros((10, 10, 10), dtype=int)
        arr[3:7, 3:7, 3:7] = 1
        masking = Masking(_ants(np.zeros((10, 10, 10))))
        result = masking._getLargestCC(arr)
        n_components = label(result).max()
        self.assertLessEqual(n_components, 1)

    def test_cleanup_mask_preserves_shape(self):
        arr = np.zeros((30, 30, 30), dtype=int)
        arr[10:20, 10:20, 10:20] = 1
        masking = Masking(_ants(np.zeros((30, 30, 30))))
        result = masking._cleanup_mask(arr)
        self.assertEqual(result.shape, arr.shape)

    def test_threshold_li_returns_nonnegative(self):
        arr = np.random.randint(0, 256, (20, 20, 20)).astype(np.float32)
        masking = Masking(_ants(arr))
        thresh = masking._get_threshold_li(arr)
        self.assertGreaterEqual(thresh, 0)

    def test_threshold_li_less_than_max(self):
        arr = np.random.uniform(10, 200, (20, 20, 20)).astype(np.float32)
        masking = Masking(_ants(arr))
        thresh = masking._get_threshold_li(arr)
        self.assertLess(thresh, arr.max())


# ---------------------------------------------------------------------------
# pad_array_n_d
# ---------------------------------------------------------------------------


class TestPadArrayND(unittest.TestCase):
    """Tests for pad_array_n_d."""

    def test_1d_to_5d(self):
        self.assertEqual(pad_array_n_d(np.arange(10), dim=5).ndim, 5)

    def test_3d_to_5d(self):
        self.assertEqual(pad_array_n_d(np.ones((4, 4, 4)), dim=5).ndim, 5)

    def test_2d_to_4d(self):
        self.assertEqual(pad_array_n_d(np.eye(3), dim=4).ndim, 4)

    def test_already_5d_unchanged(self):
        arr = np.ones((1, 1, 4, 4, 4))
        padded = pad_array_n_d(arr, dim=5)
        self.assertEqual(padded.shape, arr.shape)

    def test_data_values_preserved(self):
        arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        np.testing.assert_array_equal(pad_array_n_d(arr, dim=5).squeeze(), arr)

    def test_exceeds_5_raises(self):
        with self.assertRaises(ValueError):
            pad_array_n_d(np.ones((4, 4)), dim=6)

    def test_dask_array(self):
        arr = da.from_array(np.ones((4, 4, 4)), chunks=2)
        self.assertEqual(pad_array_n_d(arr, dim=5).ndim, 5)


# ---------------------------------------------------------------------------
# Pyramid
# ---------------------------------------------------------------------------


class TestPyramid(unittest.TestCase):
    """Tests for compute_pyramid and get_pyramid_metadata."""

    def test_metadata_has_required_keys(self):
        meta = get_pyramid_metadata()
        self.assertIn("metadata", meta)
        for key in ("description", "method", "version", "args", "kwargs"):
            self.assertIn(key, meta["metadata"])

    def test_metadata_version_is_string(self):
        self.assertIsInstance(
            get_pyramid_metadata()["metadata"]["version"], str
        )

    def test_pyramid_length_matches_n_lvls(self):
        data = da.from_array(np.ones((16, 16, 16), dtype=np.float32), chunks=8)
        result = compute_pyramid(data, n_lvls=3, scale_axis=(2, 2, 2))
        self.assertEqual(len(result), 3)

    def test_pyramid_shapes_decrease(self):
        data = da.from_array(
            np.ones((32, 32, 32), dtype=np.float32), chunks=16
        )
        result = compute_pyramid(data, n_lvls=3, scale_axis=(2, 2, 2))
        for i in range(len(result) - 1):
            self.assertGreater(
                sum(result[i].shape),
                sum(result[i + 1].shape),
                msg=f"Level {i} not larger than level {i+1}",
            )

    def test_pyramid_returns_dask_arrays(self):
        data = da.from_array(np.ones((16, 16, 16), dtype=np.float32), chunks=8)
        for arr in compute_pyramid(data, n_lvls=2, scale_axis=(2, 2, 2)):
            self.assertIsInstance(arr, da.core.Array)

    def test_single_level_full_resolution(self):
        src = np.ones((8, 8, 8), dtype=np.float32)
        data = da.from_array(src, chunks=4)
        result = compute_pyramid(data, n_lvls=1, scale_axis=(2, 2, 2))
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0].compute(), src)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestUtils(unittest.TestCase):
    """Tests for standalone functions in utils.py."""

    # ── JSON I/O ─────────────────────────────────────────────────────────────

    def test_read_json_returns_dict(self):
        data = {"a": 1, "b": [2, 3], "c": {"nested": True}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name
        try:
            self.assertEqual(read_json_as_dict(path), data)
        finally:
            os.unlink(path)

    def test_read_json_missing_file_returns_empty_dict(self):
        self.assertEqual(read_json_as_dict("/nonexistent/path/file.json"), {})

    def test_save_and_reload_roundtrip(self):
        data = {"pipeline": "ccf", "version": 5, "nested": {"x": [1, 2]}}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_dict_as_json(path, data)
            with open(path) as f:
                self.assertEqual(json.load(f), data)
        finally:
            os.unlink(path)

    def test_save_json_is_pretty_printed(self):
        """Output should be indented (not a single line)."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_dict_as_json(path, {"k": "v"})
            with open(path) as f:
                self.assertIn("\n", f.read())
        finally:
            os.unlink(path)

    # ── create_folder ─────────────────────────────────────────────────────────

    def test_create_folder_new_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            new = os.path.join(tmp, "sub")
            self.assertFalse(os.path.exists(new))
            create_folder(new)
            self.assertTrue(os.path.isdir(new))

    def test_create_folder_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            create_folder(tmp)  # already exists – must not raise
            self.assertTrue(os.path.isdir(tmp))

    def test_create_folder_nested_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "a", "b", "c")
            create_folder(nested)
            self.assertTrue(os.path.isdir(nested))

    # ── get_size ──────────────────────────────────────────────────────────────

    def test_get_size_bytes(self):
        self.assertEqual(get_size(512), "512.00B")

    def test_get_size_kilobytes(self):
        self.assertEqual(get_size(1024), "1.00KB")

    def test_get_size_megabytes(self):
        self.assertEqual(get_size(1024**2), "1.00MB")

    def test_get_size_gigabytes(self):
        self.assertEqual(get_size(1024**3), "1.00GB")

    def test_get_size_custom_suffix(self):
        self.assertIn("iB", get_size(1024, suffix="iB"))

    # ── get_channel_translations ──────────────────────────────────────────────

    def test_channels_excitation_emission_format(self):
        params = {"excitation": [488, 647], "emmission": [561, 690]}
        result = get_channel_translations(params, "Ex_488_Em_561")
        self.assertEqual(result, ["Ex_647_Em_690"])

    def test_channels_name_dict_format(self):
        params = {"ch1": "Ex_488_Em_561", "ch2": "Ex_647_Em_690"}
        result = get_channel_translations(params, "Ex_488_Em_561")
        self.assertEqual(result, ["Ex_647_Em_690"])

    def test_channels_excludes_registration_channel(self):
        params = {"excitation": [488, 561, 647], "emmission": [561, 630, 690]}
        result = get_channel_translations(params, "Ex_488_Em_561")
        self.assertNotIn("Ex_488_Em_561", result)

    def test_channels_single_entry_returns_empty(self):
        params = {"ch1": "Ex_488_Em_561"}
        self.assertEqual(get_channel_translations(params, "Ex_488_Em_561"), [])

    def test_channels_multiple_additional(self):
        params = {"excitation": [488, 561, 647], "emmission": [561, 630, 690]}
        result = get_channel_translations(params, "Ex_488_Em_561")
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# Orientation utilities
# ---------------------------------------------------------------------------


class TestRotateImage(unittest.TestCase):
    """Tests for rotate_image."""

    def test_identity_leaves_image_unchanged(self):
        img = np.arange(24, dtype=float).reshape(2, 3, 4)
        img_out, _ = rotate_image(img, np.eye(3, dtype=int), reverse=False)
        np.testing.assert_array_equal(img_out, img)

    def test_swap_axes_changes_shape(self):
        """Permutation that swaps axes 0↔1 must change the shape."""
        img = np.zeros((2, 3, 4))
        perm = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
        img_out, _ = rotate_image(img, perm, reverse=False)
        self.assertEqual(img_out.shape, (3, 2, 4))

    def test_returns_image_and_matrix(self):
        img = np.ones((4, 5, 6))
        result = rotate_image(img, np.eye(3, dtype=int), reverse=False)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)

    def test_forward_reverse_roundtrip(self):
        """forward → reverse should recover the original image."""
        img = np.arange(60, dtype=float).reshape(3, 4, 5)
        perm = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
        fwd, _ = rotate_image(img, perm, reverse=False)
        rev, _ = rotate_image(fwd, perm, reverse=True)
        np.testing.assert_array_equal(rev, img)

    def test_third_axis_unchanged_after_swap(self):
        """Swapping axes 0↔1 must not change axis 2 size."""
        img = np.zeros((2, 3, 7))
        perm = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
        img_out, _ = rotate_image(img, perm, reverse=False)
        self.assertEqual(img_out.shape[2], 7)


class TestCheckOrientation(unittest.TestCase):
    """Tests for check_orientation."""

    # Standard orientation used in CCF pipeline
    _ORIENTATIONS = {
        "anterior_to_posterior": 0,
        "superior_to_inferior": 1,
        "right_to_left": 2,
    }

    def _aligned_params(self):
        """Acquisition already aligned to the CCF orientation."""
        return [
            {"direction": "anterior_to_posterior", "dimension": 0},
            {"direction": "superior_to_inferior", "dimension": 1},
            {"direction": "right_to_left", "dimension": 2},
        ]

    def test_returns_three_values(self):
        img = np.zeros((10, 20, 30))
        result = check_orientation(
            img, self._aligned_params(), self._ORIENTATIONS
        )
        self.assertEqual(len(result), 3)

    def test_orient_mat_is_3x3(self):
        img = np.zeros((10, 20, 30))
        _, orient_mat, _ = check_orientation(
            img, self._aligned_params(), self._ORIENTATIONS
        )
        self.assertEqual(orient_mat.shape, (3, 3))

    def test_output_image_is_3d(self):
        img = np.zeros((10, 20, 30))
        img_out, _, _ = check_orientation(
            img, self._aligned_params(), self._ORIENTATIONS
        )
        self.assertEqual(img_out.ndim, 3)

    def test_already_aligned_preserves_shape(self):
        """When acquisition matches the target orientation the shape is unchanged."""
        img = np.zeros((10, 20, 30))
        img_out, _, _ = check_orientation(
            img, self._aligned_params(), self._ORIENTATIONS
        )
        self.assertEqual(img_out.shape, img.shape)

    def test_flipped_direction_handled(self):
        """A direction not in orientations (posterior_to_anterior) should use its flip."""
        img = np.zeros((10, 20, 30))
        params = [
            {"direction": "posterior_to_anterior", "dimension": 0},  # flipped
            {"direction": "superior_to_inferior", "dimension": 1},
            {"direction": "right_to_left", "dimension": 2},
        ]
        orientations = {
            "anterior_to_posterior": 0,
            "superior_to_inferior": 1,
            "right_to_left": 2,
        }
        img_out, orient_mat, _ = check_orientation(img, params, orientations)
        self.assertEqual(img_out.ndim, 3)
        # The flipped direction should produce a -1 in the matrix
        self.assertIn(-1, orient_mat)


# ---------------------------------------------------------------------------
# get_estimated_downsample  (main.py)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_MAIN, "main.py not importable from this environment")
class TestGetEstimatedDownsample(unittest.TestCase):
    """Tests for get_estimated_downsample (main.py)."""

    def test_docstring_example(self):
        """voxel (1.8,1.8,2.0) vs registration (3.6,3.6,4.0) → level 1."""
        self.assertEqual(
            get_estimated_downsample([1.8, 1.8, 2.0], (3.6, 3.6, 4.0)), 1
        )

    def test_double_downsample(self):
        """voxel (0.9,0.9,1.0) vs registration (3.6,3.6,4.0) → level 2."""
        self.assertEqual(
            get_estimated_downsample([0.9, 0.9, 1.0], (3.6, 3.6, 4.0)), 2
        )

    def test_equal_resolutions_gives_zero(self):
        self.assertEqual(
            get_estimated_downsample([16.0, 14.4, 14.4], (16.0, 14.4, 14.4)), 0
        )

    def test_returns_int(self):
        self.assertIsInstance(get_estimated_downsample([1.8, 1.8, 2.0]), int)

    def test_uses_minimum_axis_ratio(self):
        """Should pick the smallest ratio across axes (most conservative level)."""
        # axis 0: 4x, axes 1&2: 2x → min=2 → level 1
        self.assertEqual(
            get_estimated_downsample([1.0, 2.0, 2.0], (4.0, 4.0, 4.0)), 1
        )

    def test_default_registration_res(self):
        """Function should work with the default registration_res."""
        result = get_estimated_downsample([1.8, 1.8, 2.0])
        self.assertIsInstance(result, int)


if __name__ == "__main__":
    unittest.main()
