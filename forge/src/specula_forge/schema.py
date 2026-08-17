"""Canonical Specula data model. Mirrors CONTRACTS.md §1–§2.

Design: a `Label` carries both what is *printed* on the label (as rendered,
what the model sees) and the *true* product facts (what the oracle compares
against). The generator draws true facts, decides which violations to inject,
derives printed fields that deviate accordingly, then runs the oracle's
self-check gate: oracle(printed, true) must equal the injected set exactly.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


SEVERITY_WEIGHTS: dict[Severity, float] = {
    Severity.CRITICAL: 1.0,
    Severity.HIGH: 0.8,
    Severity.MEDIUM: 0.5,
    Severity.LOW: 0.2,
}


class ViolationType(str, Enum):
    ALLERGEN = "ALLERGEN"
    HEALTH_CLAIM = "HEALTH_CLAIM"
    CLAIM_THRESHOLD = "CLAIM_THRESHOLD"
    SERVING_SIZE = "SERVING_SIZE"
    MISSING_NUTRIENT = "MISSING_NUTRIENT"
    DV_ERROR = "DV_ERROR"
    HEALTHY_RULE = "HEALTHY_RULE"
    FOP_RULE = "FOP_RULE"
    ALLERGEN_STATEMENT = "ALLERGEN_STATEMENT"
    FORMATTING = "FORMATTING"
    IDENTITY = "IDENTITY"
    NET_QUANTITY = "NET_QUANTITY"
    DV_ROUNDING = "DV_ROUNDING"
    LEGIBILITY = "LEGIBILITY"


VIOLATION_WEIGHTS: dict[ViolationType, float] = {
    ViolationType.ALLERGEN: 1.0,
    ViolationType.HEALTH_CLAIM: 1.0,
    ViolationType.CLAIM_THRESHOLD: 1.0,
    ViolationType.SERVING_SIZE: 1.0,
    ViolationType.MISSING_NUTRIENT: 0.8,
    ViolationType.DV_ERROR: 0.8,
    ViolationType.HEALTHY_RULE: 0.8,
    ViolationType.FOP_RULE: 0.8,
    ViolationType.ALLERGEN_STATEMENT: 0.8,
    ViolationType.FORMATTING: 0.5,
    ViolationType.IDENTITY: 0.5,
    ViolationType.NET_QUANTITY: 0.5,
    ViolationType.DV_ROUNDING: 0.2,
    ViolationType.LEGIBILITY: 0.2,
}


class NutritionFacts(BaseModel):
    """Per-serving nutrient values. When this is the *printed* panel, the
    %DV fields are the declared ones (which may be wrong)."""

    serving_size_household: str
    serving_size_grams: float
    servings_per_container: float
    calories: float
    total_fat_g: float = 0.0
    sat_fat_g: float = 0.0
    trans_fat_g: float = 0.0
    cholesterol_mg: float = 0.0
    sodium_mg: float = 0.0
    total_carb_g: float = 0.0
    dietary_fiber_g: float = 0.0
    total_sugars_g: float = 0.0
    added_sugars_g: float = 0.0
    protein_g: float = 0.0
    vitamin_d_mcg: float = 0.0
    calcium_mg: float = 0.0
    iron_mg: float = 0.0
    potassium_mg: float = 0.0
    # declared %DV values (printed). Empty list means the panel omits the
    # corresponding %DV (itself a MISSING_NUTRIENT-adjacent formatting issue).
    dv_declared: dict[str, float] = Field(default_factory=dict)
    omitted_nutrients: list[str] = Field(default_factory=list)
    formatting_flags: list[str] = Field(default_factory=list)
    legibility_flags: list[str] = Field(default_factory=list)


class ProductFacts(BaseModel):
    """True underlying product facts (what the label *should* declare)."""

    category: str
    racc_household: str
    racc_grams: float
    serving_grams: float
    ingredients: list[str] = Field(default_factory=list)
    allergens_present: list[str] = Field(default_factory=list)  # top-9
    nutrients: NutritionFacts  # true per-serving values at the true serving


class Label(BaseModel):
    """One label as rendered (printed) + true product facts."""

    id: str
    product_name: str
    product_category: str
    printed: NutritionFacts
    true: ProductFacts
    printed_ingredients: list[str] = Field(default_factory=list)
    printed_contains: list[str] = Field(default_factory=list)
    printed_claims: list[str] = Field(default_factory=list)
    printed_health_claims: list[str] = Field(default_factory=list)
    printed_healthy_flag: bool = False
    printed_fop_symbol: str | None = None
    statement_of_identity: str | None = None
    net_quantity: str | None = None
    difficulty: float = 0.5


class Violation(BaseModel):
    type: ViolationType
    severity: Severity
    cfr: str
    observed: str
    expected: str
    correction: str


class Verdict(BaseModel):
    verdict: str  # "PASS" | "FLAG"
    violations: list[Violation] = Field(default_factory=list)


class Task(BaseModel):
    """One benchmark task: a rendered label + oracle ground truth."""

    task_id: str
    seed: int
    label: Label
    image_bytes_sha256: str
    expected: Verdict
    signature: str