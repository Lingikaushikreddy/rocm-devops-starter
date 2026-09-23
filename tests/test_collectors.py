import warnings

from rocm_portscan.collectors import collect_python, collect_text
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


def test_unparseable_python_is_skipped():
    assert collect_python("py2.py", "import apex\nprint 'hello'\n") == []
    assert collect_python("tmpl.py", "import {{ cookiecutter.pkg }}\n") == []


def test_python_line_number_and_snippet():
    source = "import os\n\nimport pynvml  # telemetry\n"
    assert collect_python("m.py", source) == [
        Finding("m.py", 3, "ROCM007", "import pynvml  # telemetry")
    ]


def test_attribute_named_like_a_package_does_not_fire():
    assert collect_python("m.py", "cfg.apex = True\nx = obj.flash_attn\n") == []


def rule_lines(findings, rule_id):
    return [f.line for f in findings if f.rule_id == rule_id]


def test_once_per_file_rules_report_first_line_only():
    source = "import torch\ntorch.cuda.synchronize()\nx = torch.cuda.current_device()\n"
    assert rule_lines(collect_python("m.py", source), "ROCM201") == [2]


def test_neg_inf_fires_at_each_fill_when_file_has_fp8():
    source = (
        "import torch\n"
        "a = x.masked_fill(m, float('-inf'))\n"
        "b = y.masked_fill(m, -float('inf'))\n"
        "c = z.to(torch.float8_e4m3fn)\n"
    )
    found = collect_python("m.py", source)
    assert rule_lines(found, "ROCM006") == [2, 3]
    assert rule_lines(found, "ROCM005") == [4]


def test_fnuz_dtypes_are_not_flagged():
    source = "import torch\nz = x.to(torch.float8_e4m3fnuz)\nw = y.to(torch.float8_e5m2fnuz)\n"
    assert collect_python("m.py", source) == []


def test_nccl_is_case_insensitive_and_other_backends_are_silent():
    assert rule_lines(collect_python("d.py", "init_process_group('NCCL')\n"), "ROCM200") == [1]
    assert collect_python("d.py", "init_process_group(backend='gloo')\n") == []


def test_warnings_from_scanned_code_are_not_printed():
    # Real repositories are full of invalid escapes like "\d" in plain
    # strings; ast.parse warns about each one, and a scan must not echo them.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        collect_python("m.py", 'import re\nPAT = re.compile("\\d+")\n')
    assert [str(w.message) for w in caught] == []


def test_pathologically_deep_python_is_skipped_not_fatal():
    # A long enough operator chain makes ast.parse raise RecursionError.
    source = "x = " + "+".join(["1"] * 200_000) + "\nimport apex\n"
    assert collect_python("deep.py", source) == []
