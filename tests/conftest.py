import pytest

from distillrl import report


@pytest.fixture(scope="session")
def full():
    return report.full()


@pytest.fixture(scope="session")
def models():
    return report.models()
