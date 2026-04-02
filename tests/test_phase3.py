"""Phase 3 tests: collapse mechanism verification.

Scenario A: Normal collapse (3 corroborating facts → deterministic)
Scenario B: Contradictory evidence retraction (minimal retraction)
Scenario C: Active query (detect missing premises, query-driven collapse)
Integration: Full observe → collapse → contradict → retract → query → rebuild cycle
"""

import pytest

from neusym_bridge.relatum.interface import (
    RelatumInterface,
    ProbFact,
    Collapsed,
    Uncertain,
    noisy_or,
)


HEAT_RULES = """
% Three modes simultaneously active → heat concentration state
heat_concentration(N) :- temperature_dominant(N), temperature_global(N), temperature_spatial(N).

% Heat concentration → structural risk
structural_risk(N) :- heat_concentration(N).
"""


def make_interface() -> RelatumInterface:
    """Create a configured RelatumInterface with heat rules."""
    ri = RelatumInterface()
    ri.load_rules_from_text(HEAT_RULES)
    ri.set_collapse_threshold("heat_concentration", 0.85)
    ri.set_collapse_threshold("structural_risk", 0.85)
    return ri


def inject_node3(ri: RelatumInterface) -> None:
    """Inject the standard 3-observation pattern for node_3."""
    ri.assert_probabilistic("temperature_dominant", ("node_3",), 0.91)
    ri.assert_probabilistic("temperature_global", ("node_3",), 0.87)
    ri.assert_probabilistic("temperature_spatial", ("node_3",), 0.83)
    ri.update_closure([])


# ---------------------------------------------------------------------------
# Unit tests: Noisy-OR
# ---------------------------------------------------------------------------

class TestNoisyOR:
    def test_empty(self):
        assert noisy_or([]) == 0.0

    def test_single(self):
        assert abs(noisy_or([0.7]) - 0.7) < 1e-8

    def test_two_sources(self):
        # 1 - (1-0.7)(1-0.7) = 1 - 0.09 = 0.91
        assert abs(noisy_or([0.7, 0.7]) - 0.91) < 1e-8

    def test_three_high(self):
        # 1 - (0.09)(0.13)(0.17) = 1 - 0.001989 ≈ 0.998
        result = noisy_or([0.91, 0.87, 0.83])
        assert result > 0.99

    def test_all_zero(self):
        assert noisy_or([0.0, 0.0]) == 0.0

    def test_one_certain(self):
        assert noisy_or([1.0, 0.5]) == 1.0


# ---------------------------------------------------------------------------
# Scenario A: Normal Collapse
# ---------------------------------------------------------------------------

class TestScenarioA:
    def test_three_facts_trigger_collapse(self):
        """Three corroborating observations should trigger collapse."""
        ri = make_interface()
        inject_node3(ri)

        assert ri.is_collapsed("heat_concentration(node_3)"), \
            "heat_concentration should be collapsed"

    def test_structural_risk_cascades(self):
        """Collapse should cascade through rules."""
        ri = make_interface()
        inject_node3(ri)

        assert ri.is_collapsed("structural_risk(node_3)"), \
            "structural_risk should cascade from heat_concentration"

    def test_collapse_confidence_high(self):
        """Combined confidence should be very high (Noisy-OR of 0.91, 0.87, 0.83)."""
        ri = make_interface()
        inject_node3(ri)

        conf = ri.get_confidence("heat_concentration(node_3)")
        assert conf > 0.95

    def test_provenance_chain_complete(self):
        """Provenance should trace back through rule to original observations."""
        ri = make_interface()
        inject_node3(ri)

        proof = ri.explain("heat_concentration(node_3)")
        assert len(proof) >= 1, "Should have at least one derivation step"
        step = proof[0]
        assert step.rule_name == "heat_concentration_rule"
        assert "temperature_dominant(node_3)" in step.premises

    def test_partial_evidence_no_collapse(self):
        """Two out of three observations should NOT trigger collapse."""
        ri = make_interface()
        ri.assert_probabilistic("temperature_dominant", ("node_3",), 0.91)
        ri.assert_probabilistic("temperature_global", ("node_3",), 0.87)
        # Deliberately omit temperature_spatial
        ri.update_closure([])

        assert not ri.is_collapsed("heat_concentration(node_3)")


# ---------------------------------------------------------------------------
# Scenario B: Contradictory Evidence Retraction
# ---------------------------------------------------------------------------

class TestScenarioB:
    def test_contradiction_triggers_retraction(self):
        """Low-confidence update on collapsed fact should trigger retraction."""
        ri = make_interface()
        inject_node3(ri)
        assert ri.is_collapsed("structural_risk(node_3)")

        # Inject contradictory evidence (confidence well below retraction threshold)
        ri.assert_probabilistic("temperature_dominant", ("node_3",), 0.12)

        assert not ri.is_collapsed("heat_concentration(node_3)"), \
            "heat_concentration should be retracted"
        assert not ri.is_collapsed("structural_risk(node_3)"), \
            "structural_risk should be retracted (depends on heat_concentration)"

    def test_unrelated_facts_preserved(self):
        """Retraction should not affect independent collapsed facts."""
        ri = make_interface()

        # Set up node_3
        inject_node3(ri)

        # Set up independent node_7
        ri.assert_probabilistic("temperature_dominant", ("node_7",), 0.92)
        ri.assert_probabilistic("temperature_global", ("node_7",), 0.88)
        ri.assert_probabilistic("temperature_spatial", ("node_7",), 0.85)
        ri.update_closure([])
        assert ri.is_collapsed("structural_risk(node_7)")

        # Retract node_3
        ri.assert_probabilistic("temperature_dominant", ("node_3",), 0.08)

        # node_3 retracted
        assert not ri.is_collapsed("structural_risk(node_3)")
        # node_7 preserved
        assert ri.is_collapsed("structural_risk(node_7)"), \
            "Independent node_7 should NOT be affected by node_3 retraction"

    def test_observation_facts_preserved(self):
        """Non-dependent observation facts should survive retraction."""
        ri = make_interface()
        inject_node3(ri)

        # Retract via temperature_dominant
        ri.assert_probabilistic("temperature_dominant", ("node_3",), 0.10)

        # temperature_global and temperature_spatial were observations,
        # not derived from temperature_dominant — should still be known
        assert ri.is_known("temperature_global(node_3)")
        assert ri.is_known("temperature_spatial(node_3)")


# ---------------------------------------------------------------------------
# Scenario C: Active Query
# ---------------------------------------------------------------------------

class TestScenarioC:
    def test_missing_premise_detected(self):
        """System should detect missing premises needed for derivation."""
        ri = make_interface()
        ri.assert_probabilistic("temperature_dominant", ("node_5",), 0.89)
        ri.assert_probabilistic("temperature_global", ("node_5",), 0.84)
        # Deliberately omit temperature_spatial
        ri.update_closure([])

        requests = ri.find_missing_premises()
        missing_preds = [(r.predicate, r.args) for r in requests]

        assert any(
            pred == "temperature_spatial" and "node_5" in args
            for pred, args in missing_preds
        ), "Should request temperature_spatial(node_5)"

    def test_query_response_triggers_collapse(self):
        """Responding to a query should complete the derivation and collapse."""
        ri = make_interface()
        ri.assert_probabilistic("temperature_dominant", ("node_5",), 0.89)
        ri.assert_probabilistic("temperature_global", ("node_5",), 0.84)
        ri.update_closure([])

        assert not ri.is_collapsed("heat_concentration(node_5)")

        # Respond to the query
        ri.assert_probabilistic("temperature_spatial", ("node_5",), 0.81)
        ri.update_closure([])

        assert ri.is_collapsed("heat_concentration(node_5)")

    def test_low_confidence_stays_uncertain(self):
        """Low-confidence evidence should not collapse."""
        ri = make_interface()
        ri.assert_probabilistic("temperature_dominant", ("node_9",), 0.40)
        ri.assert_probabilistic("temperature_global", ("node_9",), 0.35)
        ri.assert_probabilistic("temperature_spatial", ("node_9",), 0.30)
        ri.update_closure([])

        # Noisy-OR of 0.40, 0.35, 0.30 = 1 - 0.60*0.65*0.70 = 1 - 0.273 = 0.727
        # Below 0.85 threshold
        assert not ri.is_collapsed("heat_concentration(node_9)")

    def test_no_false_queries(self):
        """Fully satisfied rules should not generate queries."""
        ri = make_interface()
        inject_node3(ri)  # All premises present

        requests = ri.find_missing_premises()
        # node_3 is fully satisfied, should not appear
        node3_requests = [r for r in requests if "node_3" in r.args]
        assert len(node3_requests) == 0


# ---------------------------------------------------------------------------
# Integration: Full Cycle
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_cycle(self):
        """Complete: observe → collapse → contradict → retract → query → rebuild."""
        ri = make_interface()

        # Phase 1: Build initial knowledge
        inject_node3(ri)
        assert ri.is_collapsed("structural_risk(node_3)")
        initial_count = ri.collapsed_count()
        assert initial_count >= 2  # heat_concentration + structural_risk

        # Phase 2: Contradictory evidence
        ri.assert_probabilistic("temperature_dominant", ("node_3",), 0.09)
        assert not ri.is_collapsed("structural_risk(node_3)")

        # Phase 3: Active query
        requests = ri.find_missing_premises()
        # System should want temperature_dominant for node_3 (was retracted)

        # Phase 4: Respond with new observation (weaker but sufficient)
        ri.assert_probabilistic("temperature_dominant", ("node_3",), 0.78)
        ri.update_closure([])

        # With 0.78, 0.87, 0.83 → Noisy-OR = 1 - 0.22*0.13*0.17 ≈ 0.995 > 0.85
        assert ri.is_collapsed("heat_concentration(node_3)"), \
            "Should re-collapse after new evidence"
        assert ri.is_collapsed("structural_risk(node_3)"), \
            "structural_risk should cascade again"

    def test_multiple_nodes_independent(self):
        """Multiple independent nodes should not interfere."""
        ri = make_interface()

        for node in ["n1", "n2", "n3"]:
            ri.assert_probabilistic("temperature_dominant", (node,), 0.90)
            ri.assert_probabilistic("temperature_global", (node,), 0.88)
            ri.assert_probabilistic("temperature_spatial", (node,), 0.85)
        ri.update_closure([])

        for node in ["n1", "n2", "n3"]:
            assert ri.is_collapsed(f"structural_risk({node})")

        # Retract just n2
        ri.assert_probabilistic("temperature_dominant", ("n2",), 0.05)
        assert not ri.is_collapsed("structural_risk(n2)")
        assert ri.is_collapsed("structural_risk(n1)")
        assert ri.is_collapsed("structural_risk(n3)")
