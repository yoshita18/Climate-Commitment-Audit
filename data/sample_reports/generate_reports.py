#!/usr/bin/env python3
"""
Synthetic climate report generator.

Generates 200+ realistic corporate climate commitment reports spanning the
full credibility spectrum (A–F) across 10 industry sectors and 6 regions.
All reports are procedurally generated — no LLM required.

Usage:
    python data/sample_reports/generate_reports.py               # generate 200 reports
    python data/sample_reports/generate_reports.py --count 50    # generate 50
    python data/sample_reports/generate_reports.py --count 5 --grades A B C
"""
from __future__ import annotations

import argparse
import random
import string
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

OUTPUT_DIR = Path(__file__).parent

# ──────────────────────────────────── company templates ───────────────────

SECTORS = [
    ("technology",   "Technology"),
    ("energy",       "Energy & Utilities"),
    ("oil_gas",      "Oil & Gas"),
    ("manufacturing","Manufacturing"),
    ("finance",      "Financial Services"),
    ("retail",       "Retail & Consumer Goods"),
    ("food_agri",    "Food & Agriculture"),
    ("transport",    "Transport & Logistics"),
    ("real_estate",  "Real Estate & Construction"),
    ("mining",       "Mining & Materials"),
]

REGIONS = ["North America", "European Union", "Asia-Pacific", "United Kingdom",
           "Latin America", "Middle East & Africa"]

PREFIXES = [
    "Global", "Pacific", "Atlantic", "Northern", "Southern", "Eastern", "Western",
    "Continental", "Trans", "Apex", "Vertex", "Nexus", "Summit", "Peak", "Crown",
    "Core", "Prime", "Alpha", "Sigma", "Vega", "Orion", "Helios", "Titan",
    "Nordic", "Alpine", "Arctic", "Meridian", "Horizon", "Pinnacle",
]

SUFFIXES_BY_SECTOR = {
    "technology":    ["Tech", "Systems", "Digital", "Solutions", "AI", "Data", "Cloud"],
    "energy":        ["Energy", "Power", "Utilities", "Grid", "Electric", "Renewables"],
    "oil_gas":       ["Petroleum", "Resources", "Energy", "Upstream", "Fuels", "Hydrocarbons"],
    "manufacturing": ["Industries", "Manufacturing", "Works", "Group", "Fabrication"],
    "finance":       ["Capital", "Financial", "Bank", "Asset Management", "Investments"],
    "retail":        ["Retail", "Commerce", "Consumer", "Brands", "Markets"],
    "food_agri":     ["Foods", "Agriculture", "Agri", "Harvest", "Natural Foods"],
    "transport":     ["Logistics", "Transport", "Freight", "Aviation", "Shipping"],
    "real_estate":   ["Properties", "Realty", "Development", "Infrastructure", "Build"],
    "mining":        ["Mining", "Minerals", "Resources", "Metals", "Extraction"],
}

GRADES = ["A", "B", "C", "D", "F"]
GRADE_WEIGHTS = [0.10, 0.20, 0.30, 0.25, 0.15]   # realistic distribution


# ─────────────────────────── template blocks by grade ─────────────────────

def _target_block_A(rng: random.Random, sector: str) -> str:
    reduction_30 = rng.randint(46, 55)
    reduction_50 = rng.randint(70, 85)
    s3_30 = rng.randint(40, 48)
    baseline = rng.choice([2019, 2018, 2020])
    net_zero = rng.randint(2043, 2050)

    return f"""OUR SCIENCE-BASED CLIMATE TARGETS (SBTi VALIDATED)
Near-term (2030): Reduce absolute Scope 1 and 2 emissions by {reduction_30}% versus {baseline} baseline.
Near-term (2030): Reduce absolute Scope 3 emissions by {s3_30}% versus {baseline} baseline.
Long-term (2050): Net-zero across Scopes 1, 2, and 3 by {net_zero}.
Interim (2035):   {reduction_50}% absolute reduction in Scope 1 and 2.

All targets independently validated by SBTi in 2023 and aligned with the 1.5°C pathway
as defined in IPCC AR6. Targets cover 100% of our operational footprint and all material
Scope 3 categories (purchased goods, use of sold products, end-of-life treatment).

BASELINE EMISSIONS ({baseline})
Scope 1: {rng.randint(80_000, 800_000):,} tCO2e
Scope 2 (market-based): {rng.randint(20_000, 200_000):,} tCO2e
Scope 3 (15 categories): {rng.randint(500_000, 8_000_000):,} tCO2e
Scope 3 share of total: {rng.randint(75, 92)}%

CURRENT PROGRESS
Scope 1+2 reduction vs baseline: {rng.randint(22, 38)}%
Scope 3 reduction vs baseline: {rng.randint(12, 22)}%
Renewable electricity share: {rng.randint(72, 99)}%
Third-party assurance: Bureau Veritas (reasonable assurance, Scope 1+2; limited, Scope 3)"""


def _target_block_B(rng: random.Random, sector: str) -> str:
    reduction_30 = rng.randint(35, 45)
    s3_30 = rng.randint(25, 35)
    baseline = rng.choice([2019, 2018])
    net_zero = rng.randint(2045, 2050)

    return f"""OUR CLIMATE TARGETS
We have set the following targets, pending SBTi validation:
- Scope 1 and 2: {reduction_30}% absolute reduction by 2030 vs {baseline}
- Scope 3: {s3_30}% absolute reduction by 2030 vs {baseline}
- Net-zero emissions by {net_zero}

BASELINE ({baseline})
Scope 1: {rng.randint(100_000, 1_000_000):,} tCO2e
Scope 2 (market-based): {rng.randint(30_000, 300_000):,} tCO2e
Scope 3: {rng.randint(600_000, 6_000_000):,} tCO2e

PROGRESS TO DATE
{rng.randint(14, 28)}% Scope 1+2 reduction achieved. On track for 2030 milestones.
Renewable electricity: {rng.randint(45, 70)}% of consumption.
Offsetting: <{rng.randint(8, 15)}% of total reductions from offsets; direct cuts prioritised."""


def _target_block_C(rng: random.Random, sector: str) -> str:
    net_zero = rng.randint(2045, 2055)
    intensity_reduction = rng.randint(25, 40)

    return f"""OUR SUSTAINABILITY GOALS
We are committed to achieving net-zero emissions by {net_zero} across our direct operations.

Our near-term goal is to reduce our carbon intensity by {intensity_reduction}% per unit of revenue
by 2030, compared to our 2019 baseline.

Regarding Scope 3: We are developing a methodology to quantify and set targets for our
value chain emissions. We plan to publish Scope 3 targets by 2026.

OFFSET STRATEGY
We will use a combination of operational improvements and high-quality carbon credits
to achieve carbon neutrality in our headquarters operations by 2028.
We currently offset {rng.randint(30, 50)}% of our Scope 1+2 emissions through verified projects."""


def _target_block_D(rng: random.Random, sector: str) -> str:
    net_zero = rng.randint(2060, 2070)
    offset_pct = rng.randint(65, 85)

    return f"""OUR CLIMATE AMBITION
We aspire to achieve carbon neutrality in our operations by {net_zero}.

We have reduced our emissions intensity by {rng.randint(15, 25)}% since 2010, demonstrating
our long-term commitment to environmental stewardship.

We do not currently report Scope 3 emissions as industry methodologies continue to evolve.

Our carbon neutrality commitment will be achieved primarily through:
- Operational efficiency improvements ({100 - offset_pct}% of expected reductions)
- High-quality carbon credits ({offset_pct}% of expected reductions)"""


def _target_block_F(rng: random.Random, sector: str) -> str:
    net_zero = rng.randint(2075, 2090)
    return f"""OUR SUSTAINABILITY VISION
We are committed to a sustainable future and take our environmental responsibilities seriously.
We aspire to become carbon neutral as quickly as possible, with a long-term vision of
achieving net-zero by {net_zero} as technology and market conditions allow.

Our absolute emissions have grown by {rng.randint(5, 25)}% since 2015, reflecting our
business expansion. On an intensity basis, we have improved efficiency.

We offset our executive team's travel emissions, demonstrating our commitment."""


ABATEMENT_BLOCKS = {
    "A": lambda rng, s: f"""KEY DECARBONISATION ACTIONS
1. Renewable Energy: {rng.randint(70,99)}% clean electricity via long-term PPAs; 100% by 2026.
2. Fleet Electrification: {rng.randint(55,90)}% of owned vehicles electric by 2028.
3. Supply Chain: Requiring all Tier 1 suppliers (80% of spend) to set SBTi targets by 2026.
4. Carbon Removal: ${rng.randint(20,80)}M invested in high-quality CDR (DAC + nature-based) by 2030.
5. Internal Carbon Price: ${rng.randint(75,120)}/tCO2e applied to all capital decisions >$1M.
CDP Score: A | SBTi Status: Targets Validated | TCFD: Aligned""",

    "B": lambda rng, s: f"""KEY ACTIONS
- Procured {rng.randint(50,75)}% renewable electricity in 2023 (PPAs + RECs)
- Launched {rng.randint(2,5)} energy efficiency programmes across manufacturing
- Internal carbon price: ${rng.randint(40,75)}/tCO2e
- Supplier engagement programme: {rng.randint(30,50)}% of Tier 1 suppliers surveyed
CDP Score: B | SBTi Status: Committed""",

    "C": lambda rng, s: f"""SUSTAINABILITY INITIATIVES
- Energy efficiency: ${rng.randint(10,50)}M invested in equipment upgrades since 2020
- Renewable energy: {rng.randint(20,45)}% of electricity from renewables
- Tree planting: {rng.randint(500_000, 2_000_000):,} trees planted in partnership with NGOs
CDP Score: C""",

    "D": lambda rng, s: f"""ENVIRONMENTAL PROGRAMMES
- LED lighting retrofit completed at {rng.randint(30,60)}% of facilities
- Renewable energy PPA signed for one major facility
- Employee engagement: Green Team launched in {rng.randint(3,8)} offices""",

    "F": lambda rng, s: f"""ENVIRONMENTAL HIGHLIGHTS
- Recycled {rng.randint(40,70)}% of office waste
- Switched to eco-friendly cleaning products
- Organised a company tree-planting day""",
}

GOVERNANCE_BLOCKS = {
    "A": lambda rng: f"Board Sustainability Committee meets monthly. CEO pay tied {rng.randint(15,25)}% to climate metrics. Annual third-party assurance. CDP: A.",
    "B": lambda rng: f"Board-level oversight. Executive bonus {rng.randint(8,15)}% linked to sustainability KPIs. CDP: B.",
    "C": lambda rng: f"Chief Sustainability Officer reports to CFO. Sustainability included in executive scorecard.",
    "D": lambda rng: f"Sustainability working group established. Annual internal reporting.",
    "F": lambda rng: f"We are developing our governance framework for sustainability.",
}


def generate_report(
    company_name: str,
    sector_key: str,
    sector_name: str,
    region: str,
    grade: str,
    report_year: int,
    rng: random.Random,
) -> str:
    grade_labels = {
        "A": "Science-Aligned Net-Zero Transition Plan",
        "B": "Climate Progress Report",
        "C": "Sustainability Report",
        "D": "Environmental Commitment Statement",
        "F": "Sustainability Vision Document",
    }
    title = f"{company_name} — {grade_labels[grade]} {report_year}"
    sector_label = f"Sector: {sector_name} | Region: {region}"

    target_funcs = {
        "A": _target_block_A, "B": _target_block_B, "C": _target_block_C,
        "D": _target_block_D, "F": _target_block_F,
    }
    target_block = target_funcs[grade](rng, sector_key)
    abatement_block = ABATEMENT_BLOCKS[grade](rng, sector_key)
    governance_block = GOVERNANCE_BLOCKS[grade](rng)

    disclaimer_map = {
        "A": "Forward-looking statements are based on IPCC AR6 1.5°C scenarios.",
        "B": "Targets subject to annual review and aligned with Paris Agreement.",
        "C": "Targets reflect our current best estimates and are subject to revision.",
        "D": "All figures are unaudited management estimates.",
        "F": "Sustainability commitments are aspirational and subject to business conditions.",
    }
    disclaimer = disclaimer_map[grade]

    return f"""{title}
{sector_label}

{target_block}

{abatement_block}

GOVERNANCE
{governance_block}

DISCLAIMER
{disclaimer}
"""


def make_company_name(sector_key: str, rng: random.Random) -> str:
    prefix = rng.choice(PREFIXES)
    suffix = rng.choice(SUFFIXES_BY_SECTOR[sector_key])
    return f"{prefix} {suffix}"


def generate_all(
    count: int = 200,
    output_dir: Path = OUTPUT_DIR,
    grades_filter: Optional[list[str]] = None,
    seed: int = 42,
) -> list[Path]:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine grade distribution
    grades_pool = grades_filter or GRADES
    weights = [GRADE_WEIGHTS[GRADES.index(g)] for g in grades_pool]
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    written = []
    seen_names: set[str] = set()

    for i in range(count):
        sector_key, sector_name = rng.choice(SECTORS)
        region = rng.choice(REGIONS)
        grade = rng.choices(grades_pool, weights=weights)[0]
        year = rng.choice([2022, 2023, 2024])

        # Unique company name
        company = make_company_name(sector_key, rng)
        attempt = 0
        while company in seen_names and attempt < 10:
            company = make_company_name(sector_key, rng)
            attempt += 1
        seen_names.add(company)

        slug = company.lower().replace(" ", "_")
        filename = f"report_{i+6:03d}_{grade.lower()}_{slug}.txt"
        content = generate_report(company, sector_key, sector_name, region, grade, year, rng)

        path = output_dir / filename
        path.write_text(content, encoding="utf-8")
        written.append(path)

    print(f"Generated {len(written)} synthetic reports in {output_dir}")
    grade_counts = {}
    for p in written:
        # extract grade from filename: report_NNN_<grade>_...
        g = p.stem.split("_")[2].upper()
        grade_counts[g] = grade_counts.get(g, 0) + 1
    for g in GRADES:
        if g in grade_counts:
            print(f"  Grade {g}: {grade_counts[g]} reports")

    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic climate reports")
    parser.add_argument("--count", type=int, default=200, help="Number of reports to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--grades", nargs="+", choices=GRADES, default=None,
                        help="Restrict to specific grades")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory")
    args = parser.parse_args()

    generate_all(
        count=args.count,
        output_dir=Path(args.output_dir),
        grades_filter=args.grades,
        seed=args.seed,
    )
