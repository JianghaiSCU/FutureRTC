"""Tests for ALOHA policy conversion helpers."""

# ruff: noqa: SLF001

import numpy as np

from openpi.policies import aloha_policy


def test_songling_gripper_input_normalizes_to_unit_range():
    values = np.array([0.0, 0.05, 0.1], dtype=np.float32)

    normalized = aloha_policy._gripper_to_angular(values)

    np.testing.assert_allclose(normalized, np.array([0.0, 0.5, 1.0], dtype=np.float32))


def test_songling_gripper_output_unnormalizes_to_physical_range():
    values = np.array([0.0, 0.5, 1.0], dtype=np.float32)

    physical = aloha_policy._gripper_from_angular(values)

    np.testing.assert_allclose(physical, np.array([0.0, 0.05, 0.1], dtype=np.float32))


def test_songling_gripper_action_labels_normalize_to_training_range():
    physical = np.array([0.0, 0.05, 0.1], dtype=np.float32)

    normalized = aloha_policy._gripper_from_angular_inv(physical)

    np.testing.assert_allclose(normalized, np.array([0.0, 0.5, 1.0], dtype=np.float32))
