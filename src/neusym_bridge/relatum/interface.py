"""Relatum collapse mechanism: probabilistic facts → deterministic knowledge.

Implements the core Phase 3 logic:
- Probabilistic fact injection with confidence scores
- Noisy-OR confidence combination across corroborating evidence
- Collapse: probabilistic → deterministic when threshold exceeded
- Provenance tracking for every derivation step
- Minimal retraction along provenance chains on contradictory evidence
- Active query: detect missing premises needed for derivation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class FactSource(Enum):
    OBSERVATION = auto()
    DERIVED = auto()
    COLLAPSED = auto()


@dataclass
class ProbFact:
    """A probabilistic fact with confidence and provenance metadata."""
    predicate: str
    args: tuple[str, ...]
    confidence: float
    source: FactSource = FactSource.OBSERVATION
    fact_id: str = ""

    def __post_init__(self):
        if not self.fact_id:
            self.fact_id = f"{self.predicate}({', '.join(self.args)})"


@dataclass
class DerivationStep:
    """One step in a provenance chain."""
    rule_name: str
    premises: list[str]    # fact_ids of premises
    conclusion: str        # fact_id of conclusion
    confidence: float


@dataclass
class Rule:
    """An inference rule: if all premises hold, derive conclusion."""
    name: str
    premises: list[tuple[str, int]]  # (predicate, n_args) — patterns
    conclusion: tuple[str, int]      # (predicate, n_args)
    # For grounding: which premise arg positions map to conclusion arg positions
    # Simplified: all premises and conclusion share the same argument (node N)
    shared_arg_index: int = 0


@dataclass
class QueryRequest:
    """A request for an external observation to complete a derivation."""
    predicate: str
    args: tuple[str, ...]
    reason: str
    urgency: float = 1.0


class CollapseResult:
    """Result of a collapse attempt."""
    pass

@dataclass
class Collapsed(CollapseResult):
    fact_id: str
    confidence: float

@dataclass
class Uncertain(CollapseResult):
    fact_id: str
    confidence: float

@dataclass
class Contradicted(CollapseResult):
    fact_id: str
    conflicting_fact_id: str


# ---------------------------------------------------------------------------
# Confidence combination
# ---------------------------------------------------------------------------

def noisy_or(confidences: list[float]) -> float:
    """Noisy-OR: P(true) = 1 - Π(1 - p_i).

    Multiple independent evidence sources reinforce each other.
    """
    if not confidences:
        return 0.0
    failure = 1.0
    for p in confidences:
        failure *= (1.0 - p)
    return 1.0 - failure


# ---------------------------------------------------------------------------
# RelatumInterface
# ---------------------------------------------------------------------------

class RelatumInterface:
    """Python implementation of the Relatum collapse mechanism.

    Manages probabilistic facts, deterministic (collapsed) facts,
    inference rules, provenance chains, and active queries.
    """

    def __init__(self):
        self.prob_facts: dict[str, ProbFact] = {}
        self.collapsed_facts: dict[str, ProbFact] = {}
        self.rules: list[Rule] = []
        self.provenance: dict[str, list[DerivationStep]] = {}
        self.collapse_thresholds: dict[str, float] = {}
        self.default_threshold: float = 0.85
        self.retraction_threshold: float = 0.3
        self._log: list[str] = []

    # --- Rule management ---

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)

    def load_rules_from_text(self, text: str) -> None:
        """Parse simple Prolog-like rules.

        Format:
            conclusion(N) :- premise1(N), premise2(N), premise3(N).
        """
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if ":-" not in line:
                continue
            head, body = line.split(":-", 1)
            head = head.strip().rstrip(".")
            body = body.strip().rstrip(".")

            # Parse head: pred(args)
            h_pred, h_rest = head.split("(", 1)
            h_args = [a.strip() for a in h_rest.rstrip(")").split(",")]

            # Parse body premises
            premises = []
            for prem in body.split("),"):
                prem = prem.strip().rstrip(")")
                if "(" in prem:
                    p_pred, p_rest = prem.split("(", 1)
                    p_args = [a.strip() for a in p_rest.split(",")]
                    premises.append((p_pred.strip(), len(p_args)))

            self.rules.append(Rule(
                name=f"{h_pred.strip()}_rule",
                premises=premises,
                conclusion=(h_pred.strip(), len(h_args)),
            ))

    def set_collapse_threshold(self, predicate: str, threshold: float) -> None:
        self.collapse_thresholds[predicate] = threshold

    # --- Fact management ---

    def assert_probabilistic(
        self,
        predicate: str,
        args: tuple[str, ...] | list[str],
        confidence: float,
    ) -> str:
        """Inject a probabilistic fact.

        If the fact was previously collapsed and new confidence is below
        retraction_threshold, trigger retraction along provenance chain.
        """
        args = tuple(args)
        fact = ProbFact(
            predicate=predicate, args=args,
            confidence=confidence, source=FactSource.OBSERVATION,
        )
        fact_id = fact.fact_id

        # Check contradiction: if fact_id itself is collapsed, or if any
        # collapsed fact depends on fact_id through provenance
        if confidence < self.retraction_threshold:
            # Find all collapsed facts that depend on this observation
            dependents = self._find_dependents(fact_id)
            affected_collapsed = [d for d in dependents if d in self.collapsed_facts]

            if fact_id in self.collapsed_facts:
                affected_collapsed.append(fact_id)

            if affected_collapsed:
                self._log.append(
                    f"↩ Contradiction detected: {fact_id} conf={confidence:.3f}"
                )
                self._retract_with_provenance(fact_id)
                self.prob_facts[fact_id] = fact
                return fact_id

        # Check direct collapsed state (high confidence update: keep collapsed)
        if fact_id in self.collapsed_facts:
            return fact_id

        self.prob_facts[fact_id] = fact
        return fact_id

    def is_collapsed(self, fact_id: str) -> bool:
        return fact_id in self.collapsed_facts

    def is_known(self, fact_id: str) -> bool:
        return fact_id in self.prob_facts or fact_id in self.collapsed_facts

    def get_confidence(self, fact_id: str) -> float:
        if fact_id in self.collapsed_facts:
            return self.collapsed_facts[fact_id].confidence
        if fact_id in self.prob_facts:
            return self.prob_facts[fact_id].confidence
        return 0.0

    def collapsed_count(self) -> int:
        return len(self.collapsed_facts)

    # --- Closure and collapse ---

    def update_closure(self, delta_fact_ids: list[str]) -> list[str]:
        """Incrementally update deductive closure after new facts.

        Fires rules whose premises are now satisfied, then attempts
        collapse on any new conclusions.

        Returns list of newly derived fact_ids.
        """
        new_conclusions = []

        for rule in self.rules:
            # Find all groundings that satisfy the rule
            groundings = self._find_groundings(rule)
            for grounding in groundings:
                conclusion_id = self._instantiate_conclusion(rule, grounding)

                # Skip if already known
                if conclusion_id in self.collapsed_facts or conclusion_id in self.prob_facts:
                    continue

                # Compute conclusion confidence from premises
                premise_confs = []
                premise_ids = []
                for pred, _ in rule.premises:
                    prem_id = f"{pred}({', '.join(grounding)})"
                    conf = self.get_confidence(prem_id)
                    premise_confs.append(conf)
                    premise_ids.append(prem_id)

                if all(c > 0 for c in premise_confs):
                    # Conclusion confidence = min of premise confidences
                    conclusion_conf = min(premise_confs)
                    derived = ProbFact(
                        predicate=rule.conclusion[0],
                        args=tuple(grounding),
                        confidence=conclusion_conf,
                        source=FactSource.DERIVED,
                    )
                    self.prob_facts[conclusion_id] = derived

                    # Record provenance
                    step = DerivationStep(
                        rule_name=rule.name,
                        premises=premise_ids,
                        conclusion=conclusion_id,
                        confidence=conclusion_conf,
                    )
                    self.provenance.setdefault(conclusion_id, []).append(step)

                    new_conclusions.append(conclusion_id)

        # Attempt collapse on all new conclusions
        for cid in new_conclusions:
            self.try_collapse(cid)

        # Cascade: new collapsed facts might enable further rules
        if new_conclusions:
            further = self.update_closure(new_conclusions)
            new_conclusions.extend(further)

        return new_conclusions

    def try_collapse(self, fact_id: str) -> CollapseResult:
        """Attempt to collapse a probabilistic fact to deterministic.

        Gathers all supporting evidence (direct + derived), combines
        via Noisy-OR, and collapses if above threshold.
        """
        support = self._gather_support(fact_id)

        if not support:
            return Uncertain(fact_id=fact_id, confidence=0.0)

        combined = noisy_or(support)

        pred = fact_id.split("(")[0] if "(" in fact_id else fact_id
        threshold = self.collapse_thresholds.get(pred, self.default_threshold)

        if combined >= threshold:
            # Collapse!
            fact = self.prob_facts.pop(fact_id, None)
            if fact is None:
                fact = ProbFact(
                    predicate=pred,
                    args=tuple(fact_id.split("(")[1].rstrip(")").split(", ")),
                    confidence=combined,
                )
            fact.confidence = combined
            fact.source = FactSource.COLLAPSED
            self.collapsed_facts[fact_id] = fact
            self._log.append(f"✓ Collapsed: {fact_id} (confidence={combined:.3f})")
            return Collapsed(fact_id=fact_id, confidence=combined)

        return Uncertain(fact_id=fact_id, confidence=combined)

    # --- Provenance and explanation ---

    def explain(self, fact_id: str) -> list[DerivationStep]:
        """Return the full provenance chain for a conclusion."""
        return self.provenance.get(fact_id, [])

    # --- Retraction ---

    def _retract_with_provenance(self, fact_id: str) -> list[str]:
        """Retract a fact and all conclusions that depend on it.

        Minimal retraction: only remove facts in the provenance subtree
        rooted at fact_id. Unrelated collapsed facts are preserved.
        """
        retracted = []
        dependents = self._find_dependents(fact_id)

        # Retract dependents first (deepest first)
        for dep in reversed(dependents):
            if dep in self.collapsed_facts:
                del self.collapsed_facts[dep]
                retracted.append(dep)
                self._log.append(f"↩ Retracted: {dep}")
            if dep in self.prob_facts:
                del self.prob_facts[dep]

        # Retract the fact itself
        if fact_id in self.collapsed_facts:
            del self.collapsed_facts[fact_id]
            retracted.append(fact_id)
            self._log.append(f"↩ Retracted: {fact_id}")
        if fact_id in self.prob_facts:
            del self.prob_facts[fact_id]

        return retracted

    def _find_dependents(self, fact_id: str) -> list[str]:
        """Find all facts whose provenance depends on fact_id (BFS)."""
        dependents = []
        queue = [fact_id]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            # Find conclusions that have current as a premise
            for cid, steps in self.provenance.items():
                for step in steps:
                    if current in step.premises and cid not in visited:
                        dependents.append(cid)
                        queue.append(cid)

        return dependents

    # --- Active query ---

    def find_missing_premises(self) -> list[QueryRequest]:
        """Find premises needed to fire rules but not yet observed.

        Returns requests sorted by urgency (highest first).
        """
        requests = []
        seen = set()

        for rule in self.rules:
            # Collect all known argument groundings
            known_args = set()
            for pred, _ in rule.premises:
                for fid in list(self.prob_facts) + list(self.collapsed_facts):
                    if fid.startswith(f"{pred}("):
                        arg_str = fid.split("(", 1)[1].rstrip(")")
                        known_args.add(arg_str)

            for arg_str in known_args:
                args = tuple(a.strip() for a in arg_str.split(","))
                # Check which premises are missing for this grounding
                for pred, _ in rule.premises:
                    prem_id = f"{pred}({arg_str})"
                    if prem_id not in self.prob_facts and prem_id not in self.collapsed_facts:
                        key = (pred, args)
                        if key not in seen:
                            seen.add(key)
                            requests.append(QueryRequest(
                                predicate=pred,
                                args=args,
                                reason=f"Needed to derive {rule.conclusion[0]}({arg_str})",
                                urgency=1.0,
                            ))

        requests.sort(key=lambda r: -r.urgency)
        return requests

    # --- Internal helpers ---

    def _find_groundings(self, rule: Rule) -> list[list[str]]:
        """Find all argument groundings that could satisfy a rule's premises.

        Returns list of argument lists (one per grounding).
        Simplified: assumes all premises share the same argument tuple.
        """
        # Collect all argument tuples from known facts matching any premise
        candidate_args: set[tuple[str, ...]] = set()
        for pred, _ in rule.premises:
            for fid in list(self.prob_facts) + list(self.collapsed_facts):
                if fid.startswith(f"{pred}("):
                    arg_str = fid.split("(", 1)[1].rstrip(")")
                    args = tuple(a.strip() for a in arg_str.split(","))
                    candidate_args.add(args)

        # Filter: keep only groundings where ALL premises are satisfied
        valid = []
        for args in candidate_args:
            all_satisfied = True
            for pred, _ in rule.premises:
                prem_id = f"{pred}({', '.join(args)})"
                if prem_id not in self.prob_facts and prem_id not in self.collapsed_facts:
                    all_satisfied = False
                    break
            if all_satisfied:
                valid.append(list(args))

        return valid

    def _instantiate_conclusion(self, rule: Rule, grounding: list[str]) -> str:
        pred = rule.conclusion[0]
        return f"{pred}({', '.join(grounding)})"

    def _gather_support(self, fact_id: str) -> list[float]:
        """Gather all confidence values supporting a fact."""
        support = []

        # Direct observation confidence
        if fact_id in self.prob_facts:
            support.append(self.prob_facts[fact_id].confidence)

        # Derived confidence from provenance
        for step in self.provenance.get(fact_id, []):
            support.append(step.confidence)

        return support

    def summary(self) -> str:
        """Return a summary of the current knowledge base state."""
        lines = [
            f"Collapsed facts ({len(self.collapsed_facts)}):",
            *[f"  ✓ {fid} (conf={f.confidence:.3f})"
              for fid, f in self.collapsed_facts.items()],
            f"Probabilistic facts ({len(self.prob_facts)}):",
            *[f"  ? {fid} (conf={f.confidence:.3f})"
              for fid, f in self.prob_facts.items()],
            f"Rules ({len(self.rules)}):",
            *[f"  {r.name}" for r in self.rules],
        ]
        return "\n".join(lines)
