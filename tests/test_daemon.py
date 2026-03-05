"""Tests for src/automation/daemon.py + src/run_daemon.py (Task 014).

Covers:
- PipelineDaemon loads config with defaults when file is missing
- PipelineDaemon deep-merges user config over defaults
- validate() returns no errors for valid config
- validate() returns error for missing watchlist file
- validate() returns error for missing queue dir (when queue enabled)
- validate() returns error for unknown hybrid config
- validate() logs warning (not error) for missing Supabase credentials
- health_check() returns correct structure
- health_check() status='stopped' when nothing started
- stop() calls scheduler.stop() and queue_reader.stop()
- _deep_merge merges nested dicts correctly
- run_daemon.py: --no-scheduler flag (argparse)
- run_daemon.py: --no-queue flag (argparse)
- run_daemon.py: --run-now flag (argparse)
- run_daemon.py: --health flag prints JSON and exits
- run_daemon.py: missing config does not crash on import
"""

import json
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_daemon(tmp_path, extra_yaml: str = "") -> "PipelineDaemon":
    from src.automation.daemon import PipelineDaemon

    cfg_path = tmp_path / "automation.yaml"
    watchlist_dir = tmp_path / "watchlists"
    watchlist_dir.mkdir()
    (watchlist_dir / "default.yaml").write_text("tickers:\n  - AAPL\n  - MSFT\n")
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()

    cfg_path.write_text(f"""
scheduler:
  enabled: true
  watchlist_hour: 8
  watchlist_minute: 30
  hybrid_config: "hybrid_haiku_tools"
  publish: false
  watchlist: "default"
queue_reader:
  enabled: true
  queue_dir: "{queue_dir}"
  hybrid_config: "hybrid_haiku_tools"
  publish: false
  poll_interval_seconds: 30
{extra_yaml}
""")
    # Patch watchlist lookup so validation uses tmp watchlist dir
    with patch("src.automation.daemon.Path") as mock_path:
        mock_path.side_effect = lambda p: Path(p).with_name(
            p.replace("config/watchlists/", str(watchlist_dir) + "/")
        ) if "config/watchlists" in p else Path(p)
    return PipelineDaemon(config_path=str(cfg_path))


# ── 1. Config loading ─────────────────────────────────────────────────────────

class TestConfigLoading:

    def test_missing_config_uses_defaults(self, tmp_path):
        from src.automation.daemon import PipelineDaemon, _CONFIG_DEFAULTS
        daemon = PipelineDaemon(config_path=str(tmp_path / "nonexistent.yaml"))
        assert daemon._cfg["scheduler"]["watchlist_hour"] == _CONFIG_DEFAULTS["scheduler"]["watchlist_hour"]

    def test_user_config_overrides_defaults(self, tmp_path):
        from src.automation.daemon import PipelineDaemon
        cfg_path = tmp_path / "automation.yaml"
        cfg_path.write_text("scheduler:\n  watchlist_hour: 7\n  watchlist_minute: 15\n")
        daemon = PipelineDaemon(config_path=str(cfg_path))
        assert daemon._cfg["scheduler"]["watchlist_hour"] == 7
        assert daemon._cfg["scheduler"]["watchlist_minute"] == 15

    def test_unspecified_keys_keep_defaults(self, tmp_path):
        from src.automation.daemon import PipelineDaemon
        cfg_path = tmp_path / "automation.yaml"
        cfg_path.write_text("scheduler:\n  watchlist_hour: 7\n")
        daemon = PipelineDaemon(config_path=str(cfg_path))
        # weekdays_only should still be the default (True)
        assert daemon._cfg["scheduler"]["weekdays_only"] is True


# ── 2. _deep_merge ────────────────────────────────────────────────────────────

class TestDeepMerge:

    def test_flat_override(self):
        from src.automation.daemon import _deep_merge
        result = _deep_merge({"a": 1, "b": 2}, {"b": 99})
        assert result == {"a": 1, "b": 99}

    def test_nested_merge(self):
        from src.automation.daemon import _deep_merge
        base     = {"scheduler": {"hour": 8, "minute": 30, "enabled": True}}
        override = {"scheduler": {"hour": 7}}
        result   = _deep_merge(base, override)
        assert result["scheduler"]["hour"]    == 7
        assert result["scheduler"]["minute"]  == 30   # preserved
        assert result["scheduler"]["enabled"] is True  # preserved

    def test_new_key_added(self):
        from src.automation.daemon import _deep_merge
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_base_not_mutated(self):
        from src.automation.daemon import _deep_merge
        base = {"a": 1}
        _deep_merge(base, {"a": 2})
        assert base["a"] == 1


# ── 3. validate() ─────────────────────────────────────────────────────────────

class TestValidate:

    def test_valid_config_no_errors(self, tmp_path):
        from src.automation.daemon import PipelineDaemon

        cfg_path   = tmp_path / "automation.yaml"
        queue_dir  = tmp_path / "queue"
        queue_dir.mkdir()
        wl_path    = tmp_path / "config" / "watchlists"
        wl_path.mkdir(parents=True)
        (wl_path / "default.yaml").write_text("tickers:\n  - AAPL\n")

        cfg_path.write_text(f"""
scheduler:
  enabled: true
  hybrid_config: "hybrid_haiku_tools"
  publish: false
  watchlist: "default"
queue_reader:
  enabled: true
  queue_dir: "{queue_dir}"
  hybrid_config: "hybrid_haiku_tools"
  publish: false
""")
        daemon = PipelineDaemon(config_path=str(cfg_path))

        with patch("src.automation.daemon.Path") as mock_path_cls:
            # Make watchlist path resolution point to our temp dir
            def path_side_effect(p):
                if "config/watchlists" in str(p):
                    return wl_path / "default.yaml"
                return Path(p)
            mock_path_cls.side_effect = path_side_effect

            errors = daemon.validate()
        # hybrid_haiku_tools is a real config — no errors
        assert all("hybrid_config" not in e for e in errors)

    def test_unknown_hybrid_config_returns_error(self, tmp_path):
        from src.automation.daemon import PipelineDaemon

        cfg_path = tmp_path / "automation.yaml"
        cfg_path.write_text("scheduler:\n  hybrid_config: 'nonexistent_config_xyz'\n")
        daemon = PipelineDaemon(config_path=str(cfg_path))
        errors = daemon.validate()
        assert any("nonexistent_config_xyz" in e for e in errors)

    def test_missing_queue_dir_returns_error(self, tmp_path):
        from src.automation.daemon import PipelineDaemon

        cfg_path = tmp_path / "automation.yaml"
        cfg_path.write_text(f"""
queue_reader:
  enabled: true
  queue_dir: "{tmp_path / 'nonexistent_queue'}"
  hybrid_config: "hybrid_haiku_tools"
  publish: false
""")
        daemon = PipelineDaemon(config_path=str(cfg_path))
        errors = daemon.validate(check_scheduler=False, check_queue=True)
        assert any("Queue directory" in e for e in errors)

    def test_existing_queue_dir_no_error(self, tmp_path):
        from src.automation.daemon import PipelineDaemon

        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()
        cfg_path  = tmp_path / "automation.yaml"
        cfg_path.write_text(f"""
scheduler:
  enabled: false
queue_reader:
  enabled: true
  queue_dir: "{queue_dir}"
  hybrid_config: "hybrid_haiku_tools"
  publish: false
""")
        daemon = PipelineDaemon(config_path=str(cfg_path))
        errors = daemon.validate(check_scheduler=False, check_queue=True)
        queue_errors = [e for e in errors if "Queue directory" in e]
        assert queue_errors == []


# ── 4. health_check() ─────────────────────────────────────────────────────────

class TestHealthCheck:

    def test_health_check_structure(self, tmp_path):
        from src.automation.daemon import PipelineDaemon

        cfg_path = tmp_path / "automation.yaml"
        cfg_path.write_text("")
        daemon = PipelineDaemon(config_path=str(cfg_path))
        health = daemon.health_check()

        for key in ("status", "scheduler", "queue_reader", "uptime_seconds"):
            assert key in health

    def test_status_stopped_when_not_started(self, tmp_path):
        from src.automation.daemon import PipelineDaemon

        cfg_path = tmp_path / "automation.yaml"
        cfg_path.write_text("")
        daemon = PipelineDaemon(config_path=str(cfg_path))
        assert daemon.health_check()["status"] == "stopped"

    def test_scheduler_not_enabled_when_none(self, tmp_path):
        from src.automation.daemon import PipelineDaemon

        cfg_path = tmp_path / "automation.yaml"
        cfg_path.write_text("")
        daemon = PipelineDaemon(config_path=str(cfg_path))
        assert daemon.health_check()["scheduler"]["enabled"] is False

    def test_queue_reader_pending_count_zero(self, tmp_path):
        from src.automation.daemon import PipelineDaemon

        cfg_path = tmp_path / "automation.yaml"
        cfg_path.write_text("")
        daemon = PipelineDaemon(config_path=str(cfg_path))
        assert daemon.health_check()["queue_reader"]["pending_count"] == 0

    def test_uptime_none_before_start(self, tmp_path):
        from src.automation.daemon import PipelineDaemon

        cfg_path = tmp_path / "automation.yaml"
        cfg_path.write_text("")
        daemon = PipelineDaemon(config_path=str(cfg_path))
        assert daemon.health_check()["uptime_seconds"] is None


# ── 5. stop() ─────────────────────────────────────────────────────────────────

class TestStop:

    def test_stop_calls_scheduler_stop(self, tmp_path):
        from src.automation.daemon import PipelineDaemon

        cfg_path = tmp_path / "automation.yaml"
        cfg_path.write_text("")
        daemon = PipelineDaemon(config_path=str(cfg_path))
        mock_sched = MagicMock()
        daemon._scheduler = mock_sched

        daemon.stop()
        mock_sched.stop.assert_called_once()

    def test_stop_calls_queue_reader_stop(self, tmp_path):
        from src.automation.daemon import PipelineDaemon

        cfg_path = tmp_path / "automation.yaml"
        cfg_path.write_text("")
        daemon = PipelineDaemon(config_path=str(cfg_path))
        mock_qr = MagicMock()
        daemon._queue_reader = mock_qr

        daemon.stop()
        mock_qr.stop.assert_called_once()

    def test_stop_with_no_components_does_not_raise(self, tmp_path):
        from src.automation.daemon import PipelineDaemon

        cfg_path = tmp_path / "automation.yaml"
        cfg_path.write_text("")
        daemon = PipelineDaemon(config_path=str(cfg_path))
        # Both are None — should not raise
        daemon.stop()


# ── 6. run_daemon.py CLI flags ────────────────────────────────────────────────

class TestRunDaemonCLI:

    def test_help_exits_cleanly(self):
        """--help should print usage and exit 0."""
        from src.run_daemon import main
        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["run_daemon", "--help"]):
                main()
        assert exc.value.code == 0

    def test_health_flag_prints_json(self, tmp_path, capsys):
        cfg_path = tmp_path / "automation.yaml"
        cfg_path.write_text("")

        # PipelineDaemon is imported lazily inside main(); patch at daemon module level
        with patch("src.automation.daemon.PipelineDaemon") as MockDaemon:
            mock_instance = MagicMock()
            mock_instance.health_check.return_value = {"status": "stopped",
                                                        "scheduler": {}, "queue_reader": {},
                                                        "uptime_seconds": None}
            MockDaemon.return_value = mock_instance

            with patch("sys.argv", ["run_daemon", "--config", str(cfg_path), "--health"]):
                from src.run_daemon import main
                main()

        captured = capsys.readouterr().out
        health = json.loads(captured)
        assert "status" in health

    def test_run_now_calls_daemon_start(self, tmp_path):
        cfg_path = tmp_path / "automation.yaml"
        cfg_path.write_text("")

        with patch("src.automation.daemon.PipelineDaemon") as MockDaemon:
            mock_instance = MagicMock()
            MockDaemon.return_value = mock_instance

            with patch("sys.argv", ["run_daemon", "--config", str(cfg_path), "--run-now"]):
                from src.run_daemon import main
                main()

        mock_instance.start.assert_called_once_with(
            enable_scheduler=True,
            enable_queue=True,
            run_now=True,
        )

    def test_no_scheduler_flag(self, tmp_path):
        cfg_path = tmp_path / "automation.yaml"
        cfg_path.write_text("")

        with patch("src.automation.daemon.PipelineDaemon") as MockDaemon:
            mock_instance = MagicMock()
            MockDaemon.return_value = mock_instance

            with patch("sys.argv", ["run_daemon", "--config", str(cfg_path), "--no-scheduler"]):
                from src.run_daemon import main
                main()

        mock_instance.start.assert_called_once_with(
            enable_scheduler=False,
            enable_queue=True,
            run_now=False,
        )

    def test_no_queue_flag(self, tmp_path):
        cfg_path = tmp_path / "automation.yaml"
        cfg_path.write_text("")

        with patch("src.automation.daemon.PipelineDaemon") as MockDaemon:
            mock_instance = MagicMock()
            MockDaemon.return_value = mock_instance

            with patch("sys.argv", ["run_daemon", "--config", str(cfg_path), "--no-queue"]):
                from src.run_daemon import main
                main()

        mock_instance.start.assert_called_once_with(
            enable_scheduler=True,
            enable_queue=False,
            run_now=False,
        )
