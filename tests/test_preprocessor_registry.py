"""Tests for preprocessor and middleware registries."""

import pytest

from sam_ml.preprocessing.base import get_preprocessor, list_preprocessors
from sam_ml.preprocessing.middleware import get_middleware, list_middlewares
from sam_ml.preprocessing.preprocess_ddr2019 import Ddr2019Preprocessor


class TestPreprocessorRegistry:
    """Tests for preprocessor registry."""

    def test_list_preprocessors_includes_ddr2019(self) -> None:
        """Registry includes ddr2019 after preprocess_ddr2019 is loaded."""
        preprocessors = list_preprocessors()
        assert "ddr2019" in preprocessors

    def test_get_preprocessor_ddr2019(self) -> None:
        """get_preprocessor('ddr2019') returns Ddr2019Preprocessor class."""
        cls = get_preprocessor("ddr2019")
        assert cls is Ddr2019Preprocessor

    def test_get_preprocessor_invalid_raises(self) -> None:
        """get_preprocessor with unknown key raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            get_preprocessor("nonexistent")


class TestMiddlewareRegistry:
    """Tests for middleware registry."""

    def test_list_middlewares_includes_builtins(self) -> None:
        """Registry includes default, paper_dual, resize_norm."""
        middlewares = list_middlewares()
        assert "default" in middlewares
        assert "paper_dual" in middlewares
        assert "resize_norm" in middlewares

    def test_get_middleware_default(self) -> None:
        """get_middleware('default') returns an instance."""
        mw = get_middleware("default", min_size=512, target_size=(512, 512))
        assert mw.min_size == 512
        assert mw.target_size == (512, 512)

    def test_get_middleware_invalid_raises(self) -> None:
        """get_middleware with unknown key raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            get_middleware("nonexistent")
