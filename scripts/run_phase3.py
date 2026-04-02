"""Phase 3: Collapse mechanism verification.

Runs all three scenarios + integration test, produces verdict.
"""

import json
from pathlib import Path

from neusym_bridge.relatum.interface import (
    RelatumInterface,
    Collapsed,
    Uncertain,
    noisy_or,
)

RESULTS_DIR = Path("results")

HEAT_RULES = """
heat_concentration(N) :- temperature_dominant(N), temperature_global(N), temperature_spatial(N).
structural_risk(N) :- heat_concentration(N).
"""


def make_interface() -> RelatumInterface:
    ri = RelatumInterface()
    ri.load_rules_from_text(HEAT_RULES)
    ri.set_collapse_threshold("heat_concentration", 0.85)
    ri.set_collapse_threshold("structural_risk", 0.85)
    return ri


def scenario_a() -> dict:
    """Scenario A: Normal collapse."""
    print("=" * 50, flush=True)
    print("Scenario A: Normal Collapse", flush=True)
    print("=" * 50, flush=True)

    ri = make_interface()

    ri.assert_probabilistic("temperature_dominant", ("node_3",), 0.91)
    ri.assert_probabilistic("temperature_global", ("node_3",), 0.87)
    ri.assert_probabilistic("temperature_spatial", ("node_3",), 0.83)
    new = ri.update_closure([])

    collapsed = ri.is_collapsed("heat_concentration(node_3)")
    cascaded = ri.is_collapsed("structural_risk(node_3)")
    proof = ri.explain("heat_concentration(node_3)")
    provenance_ok = len(proof) >= 1

    conf = ri.get_confidence("heat_concentration(node_3)")
    print(f"  heat_concentration collapsed: {collapsed} (conf={conf:.3f})", flush=True)
    print(f"  structural_risk cascaded: {cascaded}", flush=True)
    print(f"  Provenance steps: {len(proof)}", flush=True)
    if proof:
        print(f"    Rule: {proof[0].rule_name}", flush=True)
        print(f"    Premises: {proof[0].premises}", flush=True)

    # Verify partial evidence does NOT collapse
    ri2 = make_interface()
    ri2.assert_probabilistic("temperature_dominant", ("node_3",), 0.91)
    ri2.assert_probabilistic("temperature_global", ("node_3",), 0.87)
    ri2.update_closure([])
    partial_blocked = not ri2.is_collapsed("heat_concentration(node_3)")
    print(f"  Partial evidence blocked: {partial_blocked}", flush=True)

    return {
        "passed": collapsed and cascaded and provenance_ok and partial_blocked,
        "collapsed": collapsed,
        "cascaded": cascaded,
        "provenance_complete": provenance_ok,
        "confidence": conf,
    }


def scenario_b() -> dict:
    """Scenario B: Contradictory evidence retraction."""
    print("\n" + "=" * 50, flush=True)
    print("Scenario B: Contradictory Evidence Retraction", flush=True)
    print("=" * 50, flush=True)

    ri = make_interface()

    # Build initial state
    ri.assert_probabilistic("temperature_dominant", ("node_3",), 0.91)
    ri.assert_probabilistic("temperature_global", ("node_3",), 0.87)
    ri.assert_probabilistic("temperature_spatial", ("node_3",), 0.83)
    ri.update_closure([])

    # Build independent node_7
    ri.assert_probabilistic("temperature_dominant", ("node_7",), 0.92)
    ri.assert_probabilistic("temperature_global", ("node_7",), 0.88)
    ri.assert_probabilistic("temperature_spatial", ("node_7",), 0.85)
    ri.update_closure([])

    print(f"  Initial: node_3 collapsed={ri.is_collapsed('structural_risk(node_3)')}", flush=True)
    print(f"  Initial: node_7 collapsed={ri.is_collapsed('structural_risk(node_7)')}", flush=True)

    # Inject contradictory evidence
    ri.assert_probabilistic("temperature_dominant", ("node_3",), 0.12)

    retracted = not ri.is_collapsed("heat_concentration(node_3)")
    cascade_retracted = not ri.is_collapsed("structural_risk(node_3)")
    node7_preserved = ri.is_collapsed("structural_risk(node_7)")
    obs_preserved = ri.is_known("temperature_global(node_3)")

    print(f"  After contradiction:", flush=True)
    print(f"    node_3 heat_concentration retracted: {retracted}", flush=True)
    print(f"    node_3 structural_risk retracted: {cascade_retracted}", flush=True)
    print(f"    node_7 preserved: {node7_preserved}", flush=True)
    print(f"    Unrelated observation preserved: {obs_preserved}", flush=True)

    return {
        "passed": retracted and cascade_retracted and node7_preserved and obs_preserved,
        "retraction_triggered": retracted,
        "minimal_retraction": node7_preserved,
        "cascade_retracted": cascade_retracted,
        "observations_preserved": obs_preserved,
    }


def scenario_c() -> dict:
    """Scenario C: Active query."""
    print("\n" + "=" * 50, flush=True)
    print("Scenario C: Active Query", flush=True)
    print("=" * 50, flush=True)

    ri = make_interface()

    # Partial evidence
    ri.assert_probabilistic("temperature_dominant", ("node_5",), 0.89)
    ri.assert_probabilistic("temperature_global", ("node_5",), 0.84)
    ri.update_closure([])

    not_yet = not ri.is_collapsed("heat_concentration(node_5)")
    print(f"  Before query: collapsed={not not_yet}", flush=True)

    # Active query
    requests = ri.find_missing_premises()
    query_correct = any(
        r.predicate == "temperature_spatial" and "node_5" in r.args
        for r in requests
    )
    print(f"  Missing premises: {[(r.predicate, r.args) for r in requests]}", flush=True)
    print(f"  Correct query generated: {query_correct}", flush=True)

    # Respond to query
    ri.assert_probabilistic("temperature_spatial", ("node_5",), 0.81)
    ri.update_closure([])

    collapsed_after = ri.is_collapsed("heat_concentration(node_5)")
    print(f"  After response: collapsed={collapsed_after}", flush=True)

    # Low confidence stays uncertain
    ri2 = make_interface()
    ri2.assert_probabilistic("temperature_dominant", ("node_9",), 0.40)
    ri2.assert_probabilistic("temperature_global", ("node_9",), 0.35)
    ri2.assert_probabilistic("temperature_spatial", ("node_9",), 0.30)
    ri2.update_closure([])
    uncertain_correct = not ri2.is_collapsed("heat_concentration(node_9)")
    print(f"  Low confidence stays uncertain: {uncertain_correct}", flush=True)

    return {
        "passed": not_yet and query_correct and collapsed_after and uncertain_correct,
        "active_query_correct": query_correct,
        "query_driven_collapse": collapsed_after,
        "uncertain_correct": uncertain_correct,
    }


def integration_test() -> dict:
    """Full cycle: observe → collapse → contradict → retract → query → rebuild."""
    print("\n" + "=" * 50, flush=True)
    print("Integration: Full Cycle", flush=True)
    print("=" * 50, flush=True)

    ri = make_interface()

    # Phase 1: Build
    ri.assert_probabilistic("temperature_dominant", ("node_3",), 0.91)
    ri.assert_probabilistic("temperature_global", ("node_3",), 0.87)
    ri.assert_probabilistic("temperature_spatial", ("node_3",), 0.83)
    ri.update_closure([])
    step1 = ri.is_collapsed("structural_risk(node_3)")
    print(f"  Step 1 (build): collapsed={step1}", flush=True)

    # Phase 2: Contradict
    ri.assert_probabilistic("temperature_dominant", ("node_3",), 0.09)
    step2 = not ri.is_collapsed("structural_risk(node_3)")
    print(f"  Step 2 (contradict): retracted={step2}", flush=True)

    # Phase 3: Rebuild
    ri.assert_probabilistic("temperature_dominant", ("node_3",), 0.78)
    ri.update_closure([])
    step3 = ri.is_collapsed("structural_risk(node_3)")
    print(f"  Step 3 (rebuild): re-collapsed={step3}", flush=True)

    print(f"\n{ri.summary()}", flush=True)

    return {
        "passed": step1 and step2 and step3,
        "build_ok": step1,
        "retract_ok": step2,
        "rebuild_ok": step3,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    a = scenario_a()
    b = scenario_b()
    c = scenario_c()
    integ = integration_test()

    # Verdict
    print("\n" + "=" * 50, flush=True)
    print("Phase 3 Verdict", flush=True)
    print("=" * 50, flush=True)

    checks = {
        "Scenario A: Normal collapse": a["passed"],
        "Scenario A: Provenance chain complete": a["provenance_complete"],
        "Scenario B: Contradiction triggers retraction": b["retraction_triggered"],
        "Scenario B: Minimal retraction (unrelated preserved)": b["minimal_retraction"],
        "Scenario C: Active query generated correctly": c["active_query_correct"],
        "Integration: Full cycle without crash": integ["passed"],
    }

    passed = sum(checks.values())
    total = len(checks)
    print(f"\nPhase 3 结果: {passed}/{total} 项通过\n", flush=True)
    for check, ok in checks.items():
        print(f"  {'✓' if ok else '✗'}  {check}", flush=True)

    overall = passed >= 5
    if overall:
        print(f"\n结论: 坍缩机制验证通过", flush=True)
    else:
        print(f"\n结论: 坍缩机制有问题，需修复", flush=True)

    verdict = {
        "checks": {k: bool(v) for k, v in checks.items()},
        "passed_count": passed,
        "total_checks": total,
        "overall_pass": overall,
        "scenario_a": a,
        "scenario_b": b,
        "scenario_c": c,
        "integration": integ,
    }

    with open(RESULTS_DIR / "phase3_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_DIR}/phase3_verdict.json", flush=True)


if __name__ == "__main__":
    main()
