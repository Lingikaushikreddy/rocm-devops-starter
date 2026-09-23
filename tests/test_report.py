import json

from rocm_portscan.report import (
    count_by_tier,
    render_json,
    render_markdown,
    render_terminal,
)
from rocm_portscan.rules import Finding, Tier

FINDINGS = [
    Finding("requirements.txt", 3, "ROCM008", "cupy-cuda12x==13.3.0"),
    Finding("train.py", 1, "ROCM001", "import apex"),
    Finding("train.py", 9, "ROCM201", "torch.cuda.synchronize()"),
]


def test_count_by_tier():
    assert count_by_tier(FINDINGS) == {Tier.BLOCKER: 1, Tier.REVIEW: 1, Tier.INFO: 1}


def test_terminal_hides_info_by_default_but_counts_it():
    out = render_terminal(FINDINGS)
    assert "requirements.txt:3: ROCM008 [BLOCKER]" in out
    assert "train.py:1: ROCM001 [REVIEW]" in out
    assert "ROCM201" not in out
    assert out.rstrip().endswith("1 blocker(s), 1 to review, 1 info")


def test_terminal_min_tier_info_shows_everything():
    assert "train.py:9: ROCM201 [INFO]" in render_terminal(FINDINGS, Tier.INFO)


def test_json_is_valid_and_carries_proofs():
    data = json.loads(render_json(FINDINGS, Tier.INFO))
    assert data["summary"] == {"blocker": 1, "review": 1, "info": 1}
    assert [f["rule"] for f in data["findings"]] == ["ROCM008", "ROCM001", "ROCM201"]
    assert data["findings"][0]["proof"] == "doc:https://docs.cupy.dev/en/stable/install.html"
    assert data["findings"][0]["tier"] == "BLOCKER"
    assert data["findings"][1]["proof"] is None


def test_markdown_never_files_a_review_finding_under_breaks():
    md = render_markdown(FINDINGS, Tier.REVIEW, target="demo")
    breaks, review = md.split("## Worth checking")
    assert "ROCM008" in breaks and "ROCM001" not in breaks
    assert "ROCM001" in review
    assert "https://docs.cupy.dev/en/stable/install.html" in md
    assert "ROCM201" not in md
    assert md.startswith("# ROCm portability report: demo\n")
    assert "Nothing here was run on AMD hardware." in md


def test_markdown_says_none_found_when_no_blockers():
    md = render_markdown(FINDINGS[1:], Tier.REVIEW, target="demo")
    assert "## Breaks on ROCm\n\nNone found." in md


def test_markdown_truncates_long_location_lists():
    many = [Finding(f"f{i:02}.py", 1, "ROCM001", "import apex") for i in range(25)]
    md = render_markdown(many, Tier.REVIEW, target="demo")
    assert "f19.py" in md and "f20.py" not in md
    assert "... and 5 more" in md


def test_markdown_neutralises_backticks_in_snippets():
    md = render_markdown([Finding("a.sh", 1, "ROCM106", "docker run `--gpus all`")], target="demo")
    assert "docker run '--gpus all'" in md


def test_selftest_proof_is_described_as_runnable():
    md = render_markdown([Finding("q.py", 2, "ROCM005", "w.to(torch.float8_e4m3fn)")], target="demo")
    assert "rocm_portscan.proofs.fn_max_overflows_fnuz()" in md
