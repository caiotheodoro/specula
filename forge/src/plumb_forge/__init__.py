"""Plumb forge package.

Seeded synthetic label generator + CFR verifier-as-oracle + contamination
monitor + golden benchmark. Contracts in ../CONTRACTS.md.
"""

from .schema import Label, NutritionFacts, Violation, Verdict  # noqa: F401

__all__ = ["Label", "NutritionFacts", "Violation", "Verdict"]