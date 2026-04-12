"""Tests for socialscikit.ui.main_app — unified launcher."""

from __future__ import annotations

import pytest


class TestMainApp:
    def test_creates_blocks(self):
        from socialscikit.ui.main_app import create_app
        app = create_app()
        assert app is not None

    def test_cli_default_launches_main(self):
        """CLI with no args should default to main_app."""
        from socialscikit.cli import main
        # Just verify it doesn't crash on import
        assert callable(main)
