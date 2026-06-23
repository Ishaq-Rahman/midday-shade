"""
Deterministic product scorer.

Design rule: the LLM never owns the numeric rank. It adds qualitative tags
elsewhere in the pipeline, but the ordering that drives real decisions is plain,
inspectable arithmetic — hard disqualifiers first, then a weighted composite of
margin / velocity / competition. That keeps ranking reproducible and debuggable,
and means a hallucination can't quietly reorder the shortlist.

Excerpt from the market-intelligence pipeline. See docs/dropship-intel.md.
"""
import math
from dataclasses import dataclass


@dataclass
class ProductScore:
    product: "NormalizedProduct"
    margin_score: float       # 0-10
    velocity_score: float     # 0-10
    competition_score: float  # 0-10 (higher = less competition = better)
    total_score: float        # weighted composite
    disqualified: bool = False
    disqualify_reason: str = ""


def score_products(products: list, config: dict) -> list[ProductScore]:
    cfg = config.get("scoring", {})
    min_margin_pct = cfg.get("min_margin_pct", 60)
    max_shipping_days = cfg.get("max_shipping_days", 14)
    target_aov = config.get("business", {}).get("target_aov", 39.99)

    velocity_w = cfg.get("velocity_weight", 0.4)
    margin_w = cfg.get("margin_weight", 0.3)
    competition_w = cfg.get("competition_weight", 0.3)

    scores: list[ProductScore] = []
    for p in products:
        # --- Hard disqualifiers: fail fast, and record *why* ---
        if p.shipping_days > max_shipping_days and p.source != "shopify_competitor":
            scores.append(ProductScore(
                p, 0, 0, 0, 0, True,
                f"shipping_days={p.shipping_days} > max {max_shipping_days}",
            ))
            continue

        # --- Margin: assume a 3x markup, reward hitting the target price band ---
        if p.price_usd > 0:
            estimated_sell = p.price_usd * 3.0
            margin_pct = (estimated_sell - p.price_usd) / estimated_sell * 100
            margin_score = min(10.0, (margin_pct / min_margin_pct) * 7.0)
            if abs(estimated_sell - target_aov) < 10:
                margin_score = min(10.0, margin_score + 1.5)
        else:
            margin_score = 0.0

        # --- Velocity: log-normalized so a few viral outliers don't dominate ---
        velocity_score = min(10.0, math.log1p(p.sales_velocity) / math.log1p(1000) * 10)

        # --- Competition: few reviews + good rating = emerging = opportunity ---
        if p.review_count == 0:
            competition_score = 5.0                      # unknown
        elif p.review_count < 200:
            competition_score = 9.0                      # early signal
        else:
            competition_score = max(1.0, 9.0 - math.log10(p.review_count))

        total = (
            margin_w * margin_score
            + velocity_w * velocity_score
            + competition_w * competition_score
        )
        scores.append(ProductScore(
            p, margin_score, velocity_score, competition_score, round(total, 2),
        ))

    scores.sort(key=lambda s: s.total_score, reverse=True)
    return scores
