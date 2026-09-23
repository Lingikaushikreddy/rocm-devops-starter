import re

from rocm_portscan.rules import _RULES, RULES, Tier


def test_rule_ids_are_unique_and_well_formed():
    assert len(RULES) == len(_RULES), "duplicate rule id"
    for rule_id, rule in RULES.items():
        assert rule.id == rule_id
        assert re.fullmatch(r"ROCM\d{3}", rule_id), rule_id


def test_every_blocker_names_a_proof():
    for rule in RULES.values():
        if rule.tier is Tier.BLOCKER:
            assert rule.proof, f"{rule.id} is a BLOCKER with no proof"


def test_proofs_are_a_known_kind():
    for rule in RULES.values():
        if rule.proof:
            kind = rule.proof.split(":", 1)[0]
            assert kind in {"doc", "selftest"}, rule.id


def test_doc_proofs_are_urls_not_labels():
    # The first draft of the spec cited docs from memory and four of five
    # were wrong. A doc proof has to be a link someone opened.
    for rule in RULES.values():
        if rule.proof.startswith("doc:"):
            assert rule.proof.startswith("doc:https://"), rule.id


def test_only_rocm008_is_a_blocker():
    # Promotion needs a verified proof (spec Revision log). Change this test
    # deliberately, in the same commit as the evidence.
    blockers = sorted(r.id for r in RULES.values() if r.tier is Tier.BLOCKER)
    assert blockers == ["ROCM008"]
