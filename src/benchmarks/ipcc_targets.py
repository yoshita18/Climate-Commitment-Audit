from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

from ..rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

BENCHMARKS_PATH = Path(__file__).resolve().parents[2] / "data" / "ipcc_benchmarks" / "ar6_targets.json"


class IPCCBenchmarkLoader:
    """Loads IPCC AR6 benchmarks and indexes them into the vector store."""

    def __init__(self, benchmarks_path: Path = BENCHMARKS_PATH):
        self.benchmarks_path = benchmarks_path
        self._data: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        if not self._data:
            with open(self.benchmarks_path, "r") as f:
                self._data = json.load(f)
        return self._data

    def to_passages(self) -> list[tuple[str, str, dict]]:
        """Convert structured benchmark data into (id, text, metadata) tuples."""
        data = self.load()
        passages = []

        # Global 1.5°C targets
        g = data["global_targets"]["1.5C_pathway"]
        passages.append((
            "global_1.5c_co2_2030",
            (
                "IPCC AR6: For a 1.5°C pathway, global CO2 emissions must fall by 43-48% by 2030 "
                "relative to 2019 levels. This requires immediate and deep cuts across all sectors."
            ),
            {"category": "global_1.5C", "metric": "co2_reduction_2030", "value": "-43%"},
        ))
        passages.append((
            "global_1.5c_netzero",
            (
                "IPCC AR6: Net-zero CO2 emissions must be achieved globally by around 2050 to limit "
                "warming to 1.5°C. This requires a complete transformation of energy systems."
            ),
            {"category": "global_1.5C", "metric": "net_zero_year", "value": "2050"},
        ))
        passages.append((
            "global_1.5c_methane",
            (
                "IPCC AR6: Methane emissions must be reduced by approximately 34% by 2030 compared "
                "to 2019 levels on the 1.5°C pathway."
            ),
            {"category": "global_1.5C", "metric": "methane_reduction_2030", "value": "-34%"},
        ))

        # Global 2°C targets
        passages.append((
            "global_2c_co2_2030",
            (
                "IPCC AR6: For a 2°C pathway, global CO2 emissions must fall by 27-32% by 2030 "
                "relative to 2019 levels. Net-zero CO2 is reached around 2070."
            ),
            {"category": "global_2C", "metric": "co2_reduction_2030", "value": "-27%"},
        ))

        # Sectoral targets
        sectoral = data["sectoral_targets"]
        passages.append((
            "energy_1.5c",
            (
                "IPCC AR6 Energy Sector: Low-carbon sources (renewables, nuclear, CCS) must supply "
                "90-99% of electricity by 2050 for 1.5°C alignment. Renewables must reach ~60% of "
                "electricity by 2030. Coal must be phased out from electricity by 2040."
            ),
            {"category": "energy", "pathway": "1.5C"},
        ))
        passages.append((
            "transport_1.5c",
            (
                "IPCC AR6 Transport Sector: Transport emissions must fall by 50-90% by 2050 vs 2019. "
                "Electric vehicles should represent ~90% of new passenger car sales by 2035 for 1.5°C."
            ),
            {"category": "transport", "pathway": "1.5C"},
        ))
        passages.append((
            "buildings_1.5c",
            (
                "IPCC AR6 Buildings Sector: Building emissions must decline by 50-80% by 2050 relative "
                "to 2019 on the 1.5°C pathway. All new buildings should be net-zero by 2030."
            ),
            {"category": "buildings", "pathway": "1.5C"},
        ))
        passages.append((
            "industry_1.5c",
            (
                "IPCC AR6 Industry Sector: Industrial emissions (steel, cement, chemicals) must fall "
                "63-90% by 2050 vs 2019. Green hydrogen is expected to provide ~15% of industrial "
                "energy by 2050 on the 1.5°C pathway."
            ),
            {"category": "industry", "pathway": "1.5C"},
        ))
        passages.append((
            "afolu_1.5c",
            (
                "IPCC AR6 AFOLU: Agriculture, Forestry, and Land Use emissions must fall ~30% by 2050. "
                "Net deforestation should reach zero by 2030. Land sinks are critical for CDR."
            ),
            {"category": "afolu", "pathway": "1.5C"},
        ))

        # Carbon budgets
        passages.append((
            "carbon_budget_1.5c",
            (
                "IPCC AR6 Carbon Budget: The remaining carbon budget for a 67% probability of limiting "
                "warming to 1.5°C was approximately 400 GtCO2 from 2020. At current emission rates of "
                "~40 GtCO2/year, this budget is exhausted within approximately 10 years."
            ),
            {"category": "carbon_budget", "pathway": "1.5C", "probability": "67%"},
        ))
        passages.append((
            "carbon_budget_2c",
            (
                "IPCC AR6 Carbon Budget: The remaining carbon budget for 2°C warming (67% probability) "
                "was approximately 1150 GtCO2 from 2020."
            ),
            {"category": "carbon_budget", "pathway": "2C", "probability": "67%"},
        ))

        # Greenwashing red flags
        passages.append((
            "greenwashing_offsets",
            (
                "Greenwashing Red Flag: Relying primarily on carbon offsets (>50% of claimed reductions) "
                "rather than direct emission cuts is a major credibility concern. Offsets are not "
                "equivalent to actual emission reductions and carry significant reversal risks."
            ),
            {"category": "greenwashing", "flag": "offset_heavy"},
        ))
        passages.append((
            "greenwashing_scope3",
            (
                "Greenwashing Red Flag: Corporate climate targets that exclude Scope 3 (value chain) "
                "emissions are incomplete. For most companies, Scope 3 represents 70-90% of total "
                "emissions. Science-based targets require covering all material emission sources."
            ),
            {"category": "greenwashing", "flag": "scope3_exclusion"},
        ))
        passages.append((
            "greenwashing_vague",
            (
                "Greenwashing Red Flag: Climate commitments using vague language like 'carbon neutral', "
                "'net-zero', or 'climate positive' without specifying baseline year, scope, methodology, "
                "and verification are considered insufficient and potentially misleading."
            ),
            {"category": "greenwashing", "flag": "vague_language"},
        ))
        passages.append((
            "sbti_criteria",
            (
                "Science Based Targets initiative (SBTi): Corporate targets must cover all material "
                "Scope 1, 2, and 3 emissions. Near-term targets (5-10 years) must align with 1.5°C. "
                "Long-term net-zero targets must be set by 2050 at the latest."
            ),
            {"category": "policy_framework", "framework": "SBTi"},
        ))
        passages.append((
            "paris_agreement",
            (
                "Paris Agreement: Countries commit to holding global warming well below 2°C and "
                "pursuing efforts to limit it to 1.5°C. NDCs must be updated every 5 years with "
                "increasing ambition. The agreement implicitly requires net-zero emissions by 2050."
            ),
            {"category": "policy_framework", "framework": "Paris Agreement"},
        ))

        return passages

    def index_into_vector_store(
        self,
        vector_store: VectorStore,
        collection_name: str,
        force_reindex: bool = False,
    ) -> int:
        existing_count = vector_store.collection_count(collection_name)
        if existing_count > 0 and not force_reindex:
            logger.info(f"Benchmarks already indexed ({existing_count} entries). Skipping.")
            return existing_count

        passages = self.to_passages()
        ids = [p[0] for p in passages]
        texts = [p[1] for p in passages]
        metadatas = [p[2] for p in passages]

        count = vector_store.upsert_texts(ids, texts, metadatas, collection_name)
        logger.info(f"Indexed {count} IPCC benchmark passages.")
        return count
