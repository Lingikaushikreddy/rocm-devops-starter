import json
import subprocess
import sys
from pathlib import Path

import pytest

from rocm_portscan.__main__ import main

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_exit_1_when_blockers_found(tmp_path, capsys):
    (tmp_path / "requirements.txt").write_text("cupy-cuda12x\n")
    assert main([str(tmp_path)]) == 1
    assert "ROCM008" in capsys.readouterr().out


def test_exit_0_when_only_review_findings(tmp_path, capsys):
    (tmp_path / "requirements.txt").write_text("apex\n")
    assert main([str(tmp_path)]) == 0
    assert "ROCM001" in capsys.readouterr().out


def test_exit_code_ignores_min_tier_filter(tmp_path):
    (tmp_path / "requirements.txt").write_text("cupy-cuda12x\n")
    assert main([str(tmp_path), "--min-tier", "blocker"]) == 1


def test_exit_2_on_missing_path(tmp_path, capsys):
    assert main([str(tmp_path / "nope")]) == 2
    assert "no such file or directory" in capsys.readouterr().err


def test_exit_2_on_bad_format(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path), "--format", "xml"])
    assert exc.value.code == 2


def test_single_file_target(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("cupy-cuda12x\n")
    assert main([str(req)]) == 1


def test_exclude_can_be_repeated(tmp_path):
    for d in ("vendor", "third_party"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "requirements.txt").write_text("cupy-cuda12x\n")
    assert main([str(tmp_path), "--exclude", "vendor", "--exclude", "third_party"]) == 0


def test_json_format(tmp_path, capsys):
    (tmp_path / "requirements.txt").write_text("apex\n")
    main([str(tmp_path), "--format", "json"])
    data = json.loads(capsys.readouterr().out)
    assert data["findings"][0]["rule"] == "ROCM001"


def test_markdown_format_names_the_target(tmp_path, capsys):
    target = tmp_path / "myrepo"
    target.mkdir()
    main([str(target), "--format", "markdown"])
    assert capsys.readouterr().out.startswith("# ROCm portability report: myrepo\n")


def test_runs_as_module(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "rocm_portscan", str(tmp_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "0 blocker(s)" in result.stdout


def test_markdown_report_lists_cli_excludes(tmp_path, capsys):
    main([str(tmp_path), "--format", "markdown", "--exclude", "examples/"])
    assert "Excluded by request: `examples`." in capsys.readouterr().out
