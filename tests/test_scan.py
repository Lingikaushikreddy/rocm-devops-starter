import os

import pytest

from rocm_portscan import scan as scan_mod
from rocm_portscan.rules import Finding


def visited(monkeypatch, root, exclude=()):
    """Paths scan() handed to the text collector, with rules taken out of the picture."""
    seen = []

    def fake_collect_text(path, text):
        seen.append(path)
        return []

    monkeypatch.setattr(scan_mod, "collect_text", fake_collect_text)
    scan_mod.scan(root, exclude)
    return sorted(seen)


def write(path, content="x\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_empty_tree(tmp_path):
    assert scan_mod.scan(tmp_path) == []


def test_visits_nested_files_with_relative_posix_paths(monkeypatch, tmp_path):
    write(tmp_path / "a" / "b" / "requirements.txt")
    write(tmp_path / "setup.py")
    assert visited(monkeypatch, tmp_path) == ["a/b/requirements.txt", "setup.py"]


def test_prunes_vendor_and_vcs_dirs(monkeypatch, tmp_path):
    for d in (".git", "node_modules", "__pycache__", ".venv", "site-packages"):
        write(tmp_path / d / "requirements.txt")
    write(tmp_path / "keep.txt")
    assert visited(monkeypatch, tmp_path) == ["keep.txt"]


def test_exclude_glob_prunes_directories_and_files(monkeypatch, tmp_path):
    write(tmp_path / "vendor" / "deep" / "requirements.txt")
    write(tmp_path / "docs" / "notes.sh")
    write(tmp_path / "keep.sh")
    got = visited(monkeypatch, tmp_path, exclude=["vendor", "*.sh"])
    assert got == []
    got = visited(monkeypatch, tmp_path, exclude=["vendor"])
    assert got == ["docs/notes.sh", "keep.sh"]


def test_binary_and_oversized_files_are_skipped(monkeypatch, tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"\x00\x01\x02" * 10)
    (tmp_path / "huge.txt").write_text("a" * (scan_mod.MAX_BYTES + 1))
    write(tmp_path / "ok.txt")
    assert visited(monkeypatch, tmp_path) == ["ok.txt"]


def test_invalid_utf8_does_not_crash(monkeypatch, tmp_path):
    (tmp_path / "latin1.txt").write_bytes("caf\xe9\n".encode("latin-1"))
    assert visited(monkeypatch, tmp_path) == ["latin1.txt"]


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_symlinks_are_not_followed(monkeypatch, tmp_path):
    outside = tmp_path / "outside"
    write(outside / "secret.txt")
    root = tmp_path / "root"
    write(root / "real.txt")
    (root / "link_to_outside").symlink_to(outside, target_is_directory=True)
    (root / "loop").symlink_to(root, target_is_directory=True)
    (root / "file_link.txt").symlink_to(outside / "secret.txt")
    assert visited(monkeypatch, root) == ["real.txt"]


def test_single_file_root(monkeypatch, tmp_path):
    target = tmp_path / "requirements.txt"
    write(target)
    assert visited(monkeypatch, target) == ["requirements.txt"]


def test_findings_are_sorted_and_deduplicated(monkeypatch, tmp_path):
    write(tmp_path / "b.txt")
    write(tmp_path / "a.txt")

    def fake(path, text):
        f = Finding(path, 1, "ROCM001", "x")
        return [f, f]

    monkeypatch.setattr(scan_mod, "collect_text", fake)
    assert scan_mod.scan(tmp_path) == [
        Finding("a.txt", 1, "ROCM001", "x"),
        Finding("b.txt", 1, "ROCM001", "x"),
    ]
