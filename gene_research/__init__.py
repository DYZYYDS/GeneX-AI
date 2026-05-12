"""
GeneX-AI (formerly XenoGenesis)
Open Source Version 1.0.3
"""

__version__ = "1.0.3"

from .gene_database import GeneDatabase, GeneRecord, ProteinDomain, PathwayRecord
from .gene_tools import GeneToolRegistry, build_default_gene_tools
from .gene_runtime import GeneResearchRuntime, GeneRuntimeConfig
from .gene_chemistry import parse_smiles, search_chemical_by_similarity, infer_chemical_targets

__all__ = [
    "GeneDatabase",
    "GeneRecord",
    "GeneResearchRuntime",
    "GeneRuntimeConfig",
    "GeneToolRegistry",
    "PathwayRecord",
    "ProteinDomain",
    "build_default_gene_tools",
    "infer_chemical_targets",
    "parse_smiles",
    "search_chemical_by_similarity",
]
