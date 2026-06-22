"""Shared test fixtures."""

import pytest

from procedural_warmup.config import load_config


@pytest.fixture
def base_cfg():
    """A default warm-up config (callers mutate fields as needed)."""
    return load_config(None)
