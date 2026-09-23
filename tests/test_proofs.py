import os

import pytest

from rocm_portscan import proofs
from rocm_portscan.rules import RULES

if os.environ.get("CI"):
    # CI must run the proofs, never skip them: a missing torch fails collection.
    import torch  # noqa: F401
else:
    pytest.importorskip("torch")

SELFTESTS = sorted(
    rule.proof.removeprefix("selftest:")
    for rule in RULES.values()
    if rule.proof.startswith("selftest:")
)


def test_selftest_list_is_not_empty():
    # Guards the parametrize below from silently collecting nothing.
    assert SELFTESTS


@pytest.mark.parametrize("name", SELFTESTS)
def test_selftest_proof_still_reproduces(name):
    fn = getattr(proofs, name, None)
    assert callable(fn), f"a rule cites selftest:{name} but proofs.py has no such function"
    assert fn() is True, f"selftest:{name} no longer reproduces; demote or delete the rule citing it"
