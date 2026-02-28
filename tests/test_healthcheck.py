from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from watchdog.cli import app

runner = CliRunner()


@pytest.fixture
def mock_settings():
    with patch("watchdog.cli.get_settings") as mock_get:
        settings = mock_get.return_value
        settings.enable_live_trading = False
        settings.anthropic_api_key = "sk-ant"
        settings.openai_api_key = None
        settings.telegram_bot_token = None
        settings.manifold_api_key = "mani-key"
        settings.manifold_user_id = "mani-user"
        yield settings


@pytest.fixture
def mock_db():
    with patch("watchdog.cli.build_engine") as mock_build, patch("watchdog.cli.text"):
        # The mock connection should not raise
        mock_conn = mock_build.return_value.connect.return_value.__enter__.return_value
        yield mock_conn


@pytest.fixture
def mock_polymarket_cli():
    with patch("watchdog.cli.PolymarketCli") as mock_cli:
        yield mock_cli


def test_healthcheck_paper_success(mock_settings, mock_db, mock_polymarket_cli):
    result = runner.invoke(app, ["healthcheck", "--mode", "paper"])
    assert result.exit_code == 0
    assert "[OK]  MANIFOLD_API_KEY" in result.stdout
    assert "[OK]  MANIFOLD_USER_ID" in result.stdout
    mock_polymarket_cli.assert_not_called()


def test_healthcheck_paper_missing_manifold_key(mock_settings, mock_db, mock_polymarket_cli):
    mock_settings.manifold_api_key = None
    result = runner.invoke(app, ["healthcheck", "--mode", "paper"])
    assert result.exit_code == 1
    assert "[FAIL] missing MANIFOLD_API_KEY" in result.stdout
    mock_polymarket_cli.assert_not_called()


def test_healthcheck_paper_does_not_call_polymarket_cli(mock_settings, mock_db, mock_polymarket_cli):
    result = runner.invoke(app, ["healthcheck"])  # paper is default
    assert result.exit_code == 0
    mock_polymarket_cli.assert_not_called()
