"""Every rule is pinned by a directory that must trip it and one that must not.

Directory names map to rules: breaks/rocm008-underscore/ belongs to ROCM008.
"""

from pathlib import Path

import pytest

from rocm_portscan.rules import RULES, Tier
from rocm_portscan.scan import scan

FIXTURES = Path(__file__).parent / "fixtures"
BREAKS = FIXTURES / "breaks"
PORTABLE = FIXTURES / "portable"


def cases(base):
    return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []


def rule_of(case):
    return case.split("-")[0].upper()


@pytest.mark.parametrize("case", cases(BREAKS))
def test_breaks_fixture_trips_its_rule(case):
    fired = {f.rule_id for f in scan(BREAKS / case)}
    assert rule_of(case) in fired, f"{case}: fired {sorted(fired)}"


@pytest.mark.parametrize("case", cases(PORTABLE))
def test_portable_fixture_does_not_trip_its_rule(case):
    found = [f for f in scan(PORTABLE / case) if f.rule_id == rule_of(case)]
    assert found == []


def test_portable_tree_has_zero_blockers():
    blockers = [f for f in scan(PORTABLE) if RULES[f.rule_id].tier is Tier.BLOCKER]
    assert blockers == []
