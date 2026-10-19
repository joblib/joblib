from pathlib import Path

import pytest

from joblib import PrintTime


@pytest.mark.parametrize("logfile", ["run.log", "nested/run.log"])
def test_relative_logfile_and_rotation(tmp_path, monkeypatch, logfile):
    monkeypatch.chdir(tmp_path)
    timer = PrintTime(logfile=logfile)
    timer("first")
    original = Path(logfile).read_text()
    assert "first" in original

    timer = PrintTime(logfile=logfile)
    assert Path(logfile + ".1").read_text() == original
    timer("second")
    assert "second" in Path(logfile).read_text()
    assert "first" not in Path(logfile).read_text()


def test_logdir_still_works(tmp_path):
    timer = PrintTime(logdir=str(tmp_path / "logs"))
    timer("finished")
    assert "finished" in (tmp_path / "logs/joblib.log").read_text()
