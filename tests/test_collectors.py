from rocm_portscan.collectors import collect_text
from rocm_portscan.rules import Finding


def test_line_number_and_snippet_keep_the_raw_line():
    text = "torch\ncupy-cuda12x==13.3.0  # gpu\n"
    assert collect_text("requirements.txt", text) == [
        Finding("requirements.txt", 2, "ROCM008", "cupy-cuda12x==13.3.0  # gpu")
    ]


def test_crlf_line_endings():
    found = collect_text("requirements.txt", "torch\r\ncupy-cuda12x\r\n")
    assert [(f.line, f.rule_id) for f in found] == [(2, "ROCM008")]


def test_unknown_file_types_are_ignored():
    assert collect_text("README.md", "pip install cupy-cuda12x apex\n") == []


def test_one_finding_per_rule_per_line():
    found = collect_text("requirements.txt", "nvidia-ml-py3 pynvml\n")
    assert [f.rule_id for f in found] == ["ROCM007"]


def test_nested_requirements_files_are_dependency_files():
    found = collect_text("requirements/requirements_nlp.txt", "apex\n")
    assert [f.rule_id for f in found] == ["ROCM001"]


def test_cuda_source_reports_once_at_line_one():
    found = collect_text("csrc/ops.cuh", "#pragma once\n__device__ int f();\n")
    assert found == [Finding("csrc/ops.cuh", 1, "ROCM003", "CUDA source file")]
