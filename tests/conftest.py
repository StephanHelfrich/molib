"""Fixtures that are available in all tests without import."""
import pytest

from helpers import random_images


@pytest.fixture
def images_3d():
    """60 random images with three objectives (fixed seed)."""
    return random_images(60, 3, seed=1)