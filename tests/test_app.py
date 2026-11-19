from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_runs_without_errors():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    assert len(at.tabs) == 4


def test_app_euler_without_gravity():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.radio(key="gravity").set_value("Sin gravedad").run()
    at.radio(key="method").set_value("euler").run()
    assert not at.exception
