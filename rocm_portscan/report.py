"""Render findings for a terminal, a forum post, or another program.

Renderers hold no opinions about severity: tier comes from the rule table.
The markdown report is what gets published, so its wording follows the spec -
only the BLOCKER section says anything breaks.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence

from .rules import RULES, Finding, Tier
from .scan import MAX_BYTES, PRUNED_DIRS, normalise_excludes

HEADINGS = {
    Tier.BLOCKER: "Breaks on ROCm",
    Tier.REVIEW: "Worth checking",
    Tier.INFO: "Portable, noted for reference",
}
MAX_LOCATIONS = 20


def count_by_tier(findings: Sequence[Finding]) -> dict[Tier, int]:
    counts = {tier: 0 for tier in Tier}
    for f in findings:
        counts[RULES[f.rule_id].tier] += 1
    return counts


def render_terminal(findings: Sequence[Finding], min_tier: Tier = Tier.REVIEW) -> str:
    out = []
    for f in _visible(findings, min_tier):
        rule = RULES[f.rule_id]
        out.append(f"{f.path}:{f.line}: {rule.id} [{rule.tier.name}] {rule.title}")
    out.append(_summary(findings))
    return "\n".join(out) + "\n"


def render_json(findings: Sequence[Finding], min_tier: Tier = Tier.REVIEW) -> str:
    payload = {
        "summary": {tier.name.lower(): n for tier, n in count_by_tier(findings).items()},
        "findings": [
            {
                "rule": f.rule_id,
                "tier": RULES[f.rule_id].tier.name,
                "title": RULES[f.rule_id].title,
                "path": f.path,
                "line": f.line,
                "snippet": f.snippet,
                "proof": RULES[f.rule_id].proof or None,
            }
            for f in _visible(findings, min_tier)
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


def render_markdown(
    findings: Sequence[Finding],
    min_tier: Tier = Tier.REVIEW,
    target: str = "",
    excluded: Sequence[str] = (),
) -> str:
    out = [
        f"# ROCm portability report: {target}",
        "",
        "Static analysis by rocm_portscan. Nothing here was run on AMD hardware.",
        'Only "Breaks on ROCm" asserts a failure, and each rule there cites its proof.',
        "Everything else is a pointer for a person to check.",
        "",
        _scope(excluded),
        "",
        f"**Summary:** {_summary(findings)}",
    ]
    by_rule: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by_rule[f.rule_id].append(f)
    for tier in sorted(Tier, reverse=True):
        if tier < min_tier:
            continue
        out += ["", f"## {HEADINGS[tier]}"]
        rule_ids = sorted(r for r in by_rule if RULES[r].tier is tier)
        if not rule_ids:
            out += ["", "None found."]
            continue
        for rule_id in rule_ids:
            rule = RULES[rule_id]
            out += ["", f"### {rule.id}: {rule.title}", "", rule.message]
            if rule.proof:
                out += ["", f"Proof: {_proof_text(rule.proof)}"]
            out.append("")
            hits = by_rule[rule_id]
            for f in hits[:MAX_LOCATIONS]:
                out.append(f"- `{f.path}:{f.line}`: `{_code(f.snippet)}`")
            if len(hits) > MAX_LOCATIONS:
                out.append(f"- ... and {len(hits) - MAX_LOCATIONS} more")
    return "\n".join(out) + "\n"


def _scope(excluded: Sequence[str]) -> str:
    # A report about a subset of a repository has to say which subset.
    line = (
        f"Not scanned: directories named {', '.join(f'`{d}`' for d in sorted(PRUNED_DIRS))}; "
        f"symlinks; binary files; files over {MAX_BYTES // 1_000_000} MB."
    )
    patterns = normalise_excludes(excluded)
    if patterns:
        line += " Excluded by request: " + ", ".join(f"`{p}`" for p in patterns) + "."
    return line


def _visible(findings: Sequence[Finding], min_tier: Tier) -> list[Finding]:
    return [f for f in findings if RULES[f.rule_id].tier >= min_tier]


def _summary(findings: Sequence[Finding]) -> str:
    c = count_by_tier(findings)
    return f"{c[Tier.BLOCKER]} blocker(s), {c[Tier.REVIEW]} to review, {c[Tier.INFO]} info"


def _proof_text(proof: str) -> str:
    kind, _, value = proof.partition(":")
    if kind == "doc":
        return value
    return f"runnable check `rocm_portscan.proofs.{value}()`, executed in CI"


def _code(text: str) -> str:
    # A backtick in a snippet would close the inline code span early.
    return text.replace("`", "'")
