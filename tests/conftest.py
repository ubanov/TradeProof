"""Shared pytest fixtures that prevent CI hangs when API keys are absent."""

import os
import socket
from pathlib import Path

import pytest


def _blank_settings_overlay():
    """Blank every TRADINGAGENTS_* setting before the package is imported.

    The package loads .env on import and folds these variables into
    DEFAULT_CONFIG, so a contributor's own settings would become the defaults
    the suite asserts on. A blank value is still present, so load_dotenv leaves
    it alone, and the overlay reads it as unset. Tests of the overlay set their own.
    """
    from dotenv import dotenv_values, find_dotenv

    names = set(os.environ)
    for filename in (".env", ".env.enterprise"):
        names |= set(dotenv_values(find_dotenv(filename, usecwd=True)))
    for name in names:
        if name.startswith("TRADINGAGENTS_"):
            os.environ[name] = ""


_blank_settings_overlay()


def pytest_configure(config):
    for marker in ("unit", "integration", "smoke"):
        config.addinivalue_line("markers", f"{marker}: {marker}-level tests")


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
    """Tests do not reach the network; one that must is marked integration."""
    if request.node.get_closest_marker("integration"):
        return

    def refuse(self, address):
        raise OSError(f"test tried to reach the network: {address}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)


_API_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_CN_API_KEY",
    "ZHIPU_API_KEY",
    "ZHIPU_CN_API_KEY",
    "MINIMAX_API_KEY",
    "MINIMAX_CN_API_KEY",
    "OPENROUTER_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
)


@pytest.fixture(autouse=True)
def _dummy_api_keys(monkeypatch):
    for env_var in _API_KEY_ENV_VARS:
        # `or` not a .get default: an env var present but empty (e.g. a key left
        # blank in a .env copied from .env.example) must still get the placeholder.
        monkeypatch.setenv(env_var, os.environ.get(env_var) or "placeholder")


@pytest.fixture(autouse=True)
def _isolate_config():
    """Reset the global dataflows config before and after each test.

    ``set_config`` merges (it never clears keys absent from the override), so a
    test that sets e.g. ``tool_vendors`` would otherwise leak into later tests
    and make routing behavior order-dependent. Replace the global outright so
    every test starts from a clean DEFAULT_CONFIG.
    """
    import copy

    import tradingagents.dataflows.config as config_module
    import tradingagents.default_config as default_config

    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    yield
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


@pytest.fixture(autouse=True)
def _isolate_ledger_root(tmp_path, monkeypatch):
    """Redirect the RecommendationRecord ledger fallback into ``tmp_path``.

    ``resolve_ledger_root`` falls back to ``DEFAULT_CONFIG["ledger_root"]``
    (``~/.tradingagents/ledger`` by default) for any config dict that omits
    its own ``ledger_root`` -- which a test building a bare config dict
    easily does. Without this, such a test writes real JSON files into the
    contributor's actual home directory ledger instead of staying inside
    pytest's own tmp_path sandbox.

    Patches ``resolve_ledger_root`` itself rather than
    ``DEFAULT_CONFIG["ledger_root"]``: several other test files reload
    ``tradingagents.default_config`` (e.g. tests/test_env_overrides.py),
    which rebinds its module-level ``DEFAULT_CONFIG`` to a brand new dict
    object -- any earlier ``monkeypatch.setitem`` on the OLD dict would then
    silently stop affecting what ``resolve_ledger_root`` actually reads,
    and a later test in the same session would write to the real ledger
    again without any assertion catching it.

    Also setitem's ``cli.run``'s OWN ``DEFAULT_CONFIG`` (captured by direct
    import, same reload-staleness risk) in place: ``_build_run_config`` --
    shared by the interactive AND headless entry points -- does
    ``DEFAULT_CONFIG.copy()``, so a real run's ``config`` dict already
    carries a (possibly stale) ``ledger_root`` key, which would otherwise
    win over the patched fallback above (``config.get("ledger_root")`` is
    then truthy, not absent).
    """
    import cli.run as cli_run
    import tradingagents.recommendation_record as recommendation_record

    fallback = Path(tmp_path) / "ledger"

    def _isolated_resolve_ledger_root(config=None):
        config = config or {}
        return Path(config.get("ledger_root") or fallback)

    monkeypatch.setattr(recommendation_record, "resolve_ledger_root", _isolated_resolve_ledger_root)
    monkeypatch.setitem(cli_run.DEFAULT_CONFIG, "ledger_root", str(fallback))


@pytest.fixture(autouse=True)
def _reset_run_scoped_contextvars():
    """Reset the tracer/seed/source-coverage contextvars before and after
    every test.

    ``execute_analysis`` itself always resets these per run (see
    ``_apply_seed_context``/``_start_llm_trace``'s own docstrings: "must not
    leak across sequential calls in the same thread"). But a test that
    drives a REAL CLI invocation end-to-end (e.g. through typer's
    ``CliRunner``) resolves a real, non-zero auto-generated base seed and
    activates ``_active_seed_context`` for real -- if no later call in that
    same test resets it, it stays active for whichever unrelated test runs
    next in the same process, e.g. one that invokes a researcher node
    directly against a bare ``MagicMock`` LLM and does not expect a
    ``seed=`` kwarg to show up on ``invoke()``.
    """
    from tradingagents.dataflows.source_coverage import set_active_source_coverage_collector
    from tradingagents.llm_trace import set_active_seed_context, set_active_tracer

    set_active_tracer(None)
    set_active_seed_context(None)
    set_active_source_coverage_collector(None)
    yield
    set_active_tracer(None)
    set_active_seed_context(None)
    set_active_source_coverage_collector(None)
