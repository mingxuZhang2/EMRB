from pathlib import Path

import numpy as np
import pytest

from evaluation.executor import (
    execute_python,
    extract_signal_filename,
    isolated_signal_workspace,
)


def test_extract_signal_filename_requires_one_plain_filename():
    assert extract_signal_filename("Signal file: EMRB_L5_2000.npy") == "EMRB_L5_2000.npy"
    with pytest.raises(ValueError):
        extract_signal_filename("No signal is named here")
    with pytest.raises(ValueError):
        extract_signal_filename("Use a.npy and b.npy")


def test_workspace_contains_only_requested_signal(tmp_path):
    np.save(tmp_path / "EMRB_L5_2000.npy", np.arange(8))
    (tmp_path / "EMRB_L5_2000.json").write_text('{"ground_truth": 42}')
    np.save(tmp_path / "EMRB_L5_2001.npy", np.arange(4))

    question = "Signal file: EMRB_L5_2000.npy"
    with isolated_signal_workspace(question, tmp_path) as workspace:
        visible = {p.name for p in Path(workspace).iterdir() if not p.name.startswith(".")}
        assert visible == {"EMRB_L5_2000.npy"}

        output = execute_python(
            "import os\n"
            "print(sorted(name for name in os.listdir('.') if not name.startswith('.')))\n"
            "print(os.path.exists('EMRB_L5_2000.json'))\n",
            workspace,
        )
        assert "['EMRB_L5_2000.npy']" in output
        assert "False" in output


def test_bubblewrap_hides_repository_path(tmp_path):
    np.save(tmp_path / "EMRB_L5_2000.npy", np.arange(8))
    secret = tmp_path / "EMRB_L5_2000.json"
    secret.write_text('{"ground_truth": 42}')

    with isolated_signal_workspace("EMRB_L5_2000.npy", tmp_path) as workspace:
        output = execute_python(
            f"import os\nprint(os.path.exists({str(secret)!r}))\n",
            workspace,
        )
        assert "False" in output
