"""Ensure repo root is the process cwd so lexicons/ and config paths resolve."""

import os

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session", autouse=True)
def _cwd_repo_root():
    prev = os.getcwd()
    os.chdir(_REPO_ROOT)
    yield
    os.chdir(prev)
