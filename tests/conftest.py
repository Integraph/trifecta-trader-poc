"""Shared pytest fixtures and configuration."""


def pytest_addoption(parser):
    parser.addoption(
        "--run-comparison",
        action="store_true",
        default=False,
        help="Run reasoning comparison tests (makes real API calls)",
    )
