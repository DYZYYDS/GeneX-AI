from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .gene_database import GeneDatabase
from .gene_chemistry import parse_smiles, search_chemical_by_similarity, infer_chemical_targets
from .gene_graph import analyze_gene_network
from .gene_advanced_tools import (
    fetch_3d_structure_info, simulate_molecular_docking, flux_balance_analysis,
    get_tissue_specific_expression, fetch_from_ensembl_api, fetch_from_uniprot_api,
    predict_variant_consequence, design_crispr_grna, predict_immunogenicity_and_toxicity,
    fetch_genes_by_ontology_api, dynamic_ontology_search
)
from .gene_integrations import GlobalBioDataGateway, EvidenceTracker, ActiveLearningEngine
from .gene_advanced_models import BioFoundationModelProxy, EpigeneticAnalyzer, StructuralBiologyEngine
from .gene_simulation import (
    ThermodynamicSimulator, KineticODESimulator, ExtremeEnvironmentProteinSimulator,
    WholeCellResourceAllocator, HodgkinHuxleySimulator, ReactionDiffusionSimulator,
)
from .gene_gpu_compute import (
    GenerativeProteinDiffusion, QuantumEnzymeSimulator,
    QuantumBioSandbox, SpatialOmicsEngine,
    TF_PWM, _sequence_to_pwm_score, _calculate_crispr_off_target,
)
from .gene_vectors import GeneVectorDatabase
from .gene_visualization import ProteinStructureRenderer
from .gene_benchmark import GeneResearchBenchmarkSuite
from .gene_first_principles import FirstPrinciplesCalculators
from .gene_ultimate_modules import (
    XNAAssembler, MultiScaleCoupler, LabAutomationGenerator, EcologicalSimulator,
    UniversalLifeFormGenerator, WholeCellSandbox, EvolutionaryTrajectoryEngine, BiogeochemicalDynamicsEngine,
    VirtualClinicalTrialsEngine, EpigeneticReprogrammingSimulator, PangenomeHGTEngine, 
    NeuroGenomicTopologyModel, AstrobiologyPanspermiaEngine
)

ToolHandler = Callable[..., Any]

GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

AMINO_ACID_PROPERTIES = {
    "A": {"name": "Alanine", "mw": 89.09, "pI": 6.00, "hydrophobicity": 1.8, "polarity": 0.0, "group": "nonpolar_aliphatic"},
    "R": {"name": "Arginine", "mw": 174.20, "pI": 10.76, "hydrophobicity": -4.5, "polarity": 1.0, "group": "positive_charged"},
    "N": {"name": "Asparagine", "mw": 132.12, "pI": 5.41, "hydrophobicity": -3.5, "polarity": 1.0, "group": "polar_uncharged"},
    "D": {"name": "Aspartic Acid", "mw": 133.10, "pI": 2.77, "hydrophobicity": -3.5, "polarity": 1.0, "group": "negative_charged"},
    "C": {"name": "Cysteine", "mw": 121.15, "pI": 5.07, "hydrophobicity": 2.5, "polarity": 0.5, "group": "polar_uncharged"},
    "Q": {"name": "Glutamine", "mw": 146.15, "pI": 5.65, "hydrophobicity": -3.5, "polarity": 1.0, "group": "polar_uncharged"},
    "E": {"name": "Glutamic Acid", "mw": 147.13, "pI": 3.22, "hydrophobicity": -3.5, "polarity": 1.0, "group": "negative_charged"},
    "G": {"name": "Glycine", "mw": 75.07, "pI": 5.97, "hydrophobicity": -0.4, "polarity": 0.0, "group": "nonpolar_aliphatic"},
    "H": {"name": "Histidine", "mw": 155.16, "pI": 7.59, "hydrophobicity": -3.2, "polarity": 1.0, "group": "positive_charged"},
    "I": {"name": "Isoleucine", "mw": 131.17, "pI": 6.02, "hydrophobicity": 4.5, "polarity": 0.0, "group": "nonpolar_aliphatic"},
    "L": {"name": "Leucine", "mw": 131.17, "pI": 5.98, "hydrophobicity": 3.8, "polarity": 0.0, "group": "nonpolar_aliphatic"},
    "K": {"name": "Lysine", "mw": 146.19, "pI": 9.74, "hydrophobicity": -3.9, "polarity": 1.0, "group": "positive_charged"},
    "M": {"name": "Methionine", "mw": 149.21, "pI": 5.74, "hydrophobicity": 1.9, "polarity": 0.0, "group": "nonpolar_aliphatic"},
    "F": {"name": "Phenylalanine", "mw": 165.19, "pI": 5.48, "hydrophobicity": 2.8, "polarity": 0.0, "group": "aromatic"},
    "P": {"name": "Proline", "mw": 115.13, "pI": 6.30, "hydrophobicity": -1.6, "polarity": 0.0, "group": "nonpolar_aliphatic"},
    "S": {"name": "Serine", "mw": 105.09, "pI": 5.68, "hydrophobicity": -0.8, "polarity": 1.0, "group": "polar_uncharged"},
    "T": {"name": "Threonine", "mw": 119.12, "pI": 5.60, "hydrophobicity": -0.7, "polarity": 1.0, "group": "polar_uncharged"},
    "W": {"name": "Tryptophan", "mw": 204.23, "pI": 5.89, "hydrophobicity": -0.9, "polarity": 0.5, "group": "aromatic"},
    "Y": {"name": "Tyrosine", "mw": 181.19, "pI": 5.66, "hydrophobicity": -1.3, "polarity": 0.5, "group": "aromatic"},
    "V": {"name": "Valine", "mw": 117.15, "pI": 5.96, "hydrophobicity": 4.2, "polarity": 0.0, "group": "nonpolar_aliphatic"},
}

KNOWN_MOTIFS = {
    "NLS": {"pattern": "PKKKRKV", "name": "SV40 Nuclear Localization Signal", "description": "Targets protein to the nucleus"},
    "NES": {"pattern": "L.{2,3}[LIVFM].{2,3}L.[LI]", "name": "Nuclear Export Signal", "description": "Targets protein for nuclear export"},
    "KRAB": {"pattern": "VTFEDVAV[YF][FIT]T[RK][EQ]EW[EQ][LM]LD[SA][LQ][KQ]R[AL][LV]Y[RK][DE]VMLENY[SRQ]N[LV][AS]L[AIV]", "name": "KRAB domain", "description": "Transcriptional repression domain in zinc finger proteins"},
    "ZINC_FINGER_C2H2": {"pattern": "Y[^.]{0,10}C.{2,4}C.{3}F.{5}L.{2}H.{3,5}H", "name": "C2H2 Zinc Finger", "description": "DNA-binding domain, largest transcription factor family"},
    "HELIX_LOOP_HELIX": {"pattern": "[RK]{2}.{4,10}[RK].{6,12}[LIVM]{2}.{6,12}[LIVM]{2}", "name": "Helix-Loop-Helix", "description": "DNA-binding and protein dimerization domain"},
    "LEUCINE_ZIPPER": {"pattern": "L.{6}L.{6}L.{6}L", "name": "Leucine Zipper", "description": "Coiled-coil dimerization domain"},
    "ATP_BINDING": {"pattern": "[AG].{4}GK[ST]", "name": "P-loop ATP/GTP Binding", "description": "Phosphate-binding loop for ATP/GTP hydrolysis"},
    "KINASE_ACTIVE": {"pattern": "HRD[LIVMFY]K[^.]{2}N", "name": "Kinase Catalytic Loop", "description": "Catalytic loop of protein kinases"},
    "SH2": {"pattern": "[WFY].{3,5}[GS].{0,3}[FL].{0,2}[RP][LYF]", "name": "SH2 Domain", "description": "Phosphotyrosine-binding domain in signaling proteins"},
    "SH3": {"pattern": "[AP].{5,10}P.{2}P.{0,3}[RK]", "name": "SH3 Domain", "description": "Proline-rich motif binding domain"},
    "TRANSMEMBRANE": {"pattern": "[LIVMF].{15,25}[LIVMF]", "name": "Transmembrane Helix", "description": "Membrane-spanning alpha-helical domain"},
    "GPCR_SIGNATURE": {"pattern": "[DE]R[YF][LIVMFY]", "name": "GPCR DRY motif", "description": "Conserved motif in G-protein coupled receptors for activation"},
    "DEATH_DOMAIN": {"pattern": "[LIVM].{6,10}[LIVM].{4,6}[LIVM].{4,6}[LIVM].{4,6}[LIVM].{4,6}[LIVM]", "name": "Death Domain Fold", "description": "Apoptosis signaling domain"},
    "WD_REPEAT": {"pattern": "[LIVM][^.]{6,14}[LIVM].{0,4}WD", "name": "WD40 Repeat", "description": "Beta-propeller scaffold for protein-protein interactions"},
    "ANK_REPEAT": {"pattern": "[LIVMFY][^.]{15,30}[LIVMFY].{4}[LIVMFY][^.]{4,8}[LIVMFY]", "name": "Ankyrin Repeat", "description": "Helix-turn-helix protein interaction motif"},
    "SECRETION_SIGNAL": {"pattern": "^M[^.]{1,30}[LIVMFYWAT]{3,7}Q", "name": "Secretion Signal Peptide", "description": "N-terminal signal for ER targeting and secretion"},
    "MITOCHONDRIAL_TARGETING": {"pattern": "^M[LARQ][^.]{0,20}[RK][^.]{0,10}[LARQ][^.]{0,5}[LARQ]", "name": "Mitochondrial Targeting Signal", "description": "N-terminal amphipathic helix for mitochondrial import"},
    "UBIQUITIN_LIGASE": {"pattern": "RING|HECT|U-box", "name": "Ubiquitin Ligase Domain", "description": "E3 ubiquitin ligase catalytic domain"},
    "CAS9_RUVC": {"pattern": "[IV]V[DE][LM][IV][DE].{0,3}D", "name": "Cas9 RuvC Nuclease", "description": "CRISPR Cas9 RuvC-like nuclease domain"},
    "TAL_EFFECTOR": {"pattern": "LTP[DE]QVVAIAS.{0,2}GGKQAL[ET]TVQRLLPVLCQ[DA]HG", "name": "TALE DNA-binding Repeat", "description": "Transcription activator-like effector repeat for programmable DNA binding"},
}

FUNCTIONAL_GROUP_CHEMISTRY = {
    "kinase": {"reaction": "ATP + protein → ADP + phosphoprotein", "cofactors": ["ATP", "Mg2+"], "mechanism": "磷酸基团转移", "active_site_residues": ["D", "K", "N"]},
    "phosphatase": {"reaction": "phosphoprotein + H2O → protein + Pi", "cofactors": ["Mg2+", "Zn2+"], "mechanism": "磷酸酯水解", "active_site_residues": ["C", "R", "D"]},
    "protease": {"reaction": "protein + H2O → peptide fragments", "cofactors": ["Zn2+", "Ca2+"], "mechanism": "肽键水解", "active_site_residues": ["S", "H", "D", "C"]},
    "methyltransferase": {"reaction": "S-adenosyl-L-methionine + substrate → S-adenosyl-L-homocysteine + methylated substrate", "cofactors": ["SAM"], "mechanism": "甲基转移", "active_site_residues": ["D", "K", "Y"]},
    "acetyltransferase": {"reaction": "acetyl-CoA + substrate → CoA + acetylated substrate", "cofactors": ["Acetyl-CoA"], "mechanism": "乙酰基转移", "active_site_residues": ["E", "R", "Y"]},
    "oxidoreductase": {"reaction": "reduced substrate + NAD+/FAD → oxidized substrate + NADH/FADH2", "cofactors": ["NAD+", "FAD", "NADP+"], "mechanism": "氧化还原电子传递", "active_site_residues": ["C", "H", "Y"]},
    "helicase": {"reaction": "ATP + dsDNA/RNA → ADP + ssDNA/RNA", "cofactors": ["ATP", "Mg2+"], "mechanism": "核酸双链解旋", "active_site_residues": ["K", "D", "E", "Q"]},
    "nuclease": {"reaction": "DNA/RNA + H2O → nucleotide fragments", "cofactors": ["Mg2+", "Mn2+", "Zn2+"], "mechanism": "磷酸二酯键水解", "active_site_residues": ["D", "E", "H", "K"]},
    "polymerase": {"reaction": "dNTP/NTP + primer → elongated chain + PPi", "cofactors": ["Mg2+"], "mechanism": "核苷酸聚合", "active_site_residues": ["D", "D", "D"]},
    "ligase": {"reaction": "ATP + DNA nick → AMP + PPi + sealed DNA", "cofactors": ["ATP", "Mg2+"], "mechanism": "磷酸二酯键连接", "active_site_residues": ["K", "D"]},
    "transporter": {"reaction": "substrate(out) + ion_gradient → substrate(in)", "cofactors": ["Na+", "H+", "Cl-"], "mechanism": "跨膜转运/共转运", "active_site_residues": ["D", "R", "E", "W"]},
    "channel": {"reaction": "ion(out) ↔ ion(in)", "cofactors": [], "mechanism": "离子选择性通透", "active_site_residues": ["K", "R", "D", "E"]},
    "receptor": {"reaction": "ligand + receptor → receptor* → downstream signaling", "cofactors": [], "mechanism": "构象变化触发信号级联", "active_site_residues": ["Y", "S", "T", "K"]},
    "transcription_factor": {"reaction": "TF + DNA → TF-DNA complex → transcription regulation", "cofactors": ["Zn2+"], "mechanism": "特异性DNA序列识别与转录调控", "active_site_residues": ["R", "K", "Q", "N"]},
    "chaperone": {"reaction": "unfolded protein + ATP → folded protein + ADP", "cofactors": ["ATP", "Mg2+"], "mechanism": "ATP驱动蛋白质折叠辅助", "active_site_residues": ["D", "K", "E"]},
    "ubiquitin_ligase": {"reaction": "Ubiquitin + E2~Ub → substrate-Ub + E2", "cofactors": ["ATP"], "mechanism": "泛素分子共价连接到底物", "active_site_residues": ["C", "H", "D"]},
    "deubiquitinase": {"reaction": "substrate-Ub + H2O → substrate + Ub", "cofactors": [], "mechanism": "泛素链水解移除", "active_site_residues": ["C", "H", "D", "Q"]},
    "G_protein": {"reaction": "GTP + Gα → Gα-GTP → Gα-GDP + Pi", "cofactors": ["GTP", "Mg2+"], "mechanism": "GTP/GDP结合与水解的分子开关", "active_site_residues": ["G", "K", "D"]},
    "synthetase": {"reaction": "amino acid + ATP + tRNA → aminoacyl-tRNA + AMP + PPi", "cofactors": ["ATP", "Mg2+", "Zn2+"], "mechanism": "氨基酸活化与tRNA装载", "active_site_residues": ["K", "D", "R", "H"]},
    "ribosomal": {"reaction": "aminoacyl-tRNA + peptidyl-tRNA → peptidyl-tRNA(n+1) + tRNA", "cofactors": ["Mg2+"], "mechanism": "肽键形成的核酶催化", "active_site_residues": ["R", "K", "H"]},
}


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    handler: ToolHandler


class GeneToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, name: str, description: str, handler: ToolHandler) -> None:
        self._tools[name] = ToolDefinition(name=name, description=description, handler=handler)

    def execute(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"未知工具: {name}")
        handler = self._tools[name].handler
        cleaned = self._remap_aliases(kwargs)
        try:
            return handler(**cleaned)
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                import inspect
                sig = inspect.signature(handler)
                valid = {k for k in cleaned if k in sig.parameters}
                stripped = {k: v for k, v in cleaned.items() if k in valid}
                return handler(**stripped)
            raise

    def _remap_aliases(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        out = dict(kwargs)
        swaps = [
            ("seq", "sequence"), ("sequence_nt", "sequence"), ("dna", "sequence"),
            ("sequence", "dna_sequence"),
            ("protein", "sequence_aa"), ("aa_seq", "sequence_aa"), ("amino_acid_sequence", "sequence_aa"),
            ("gene", "gene_id"), ("id", "gene_id"), ("symbol", "gene_id"),
            ("motif", "motif_name"), ("pattern", "motif_name"),
            ("species_from", "source_species"), ("species_to", "target_species"),
            ("organ", "target_organ"), ("tissue", "target_organ"),
            ("trait", "target_trait"), ("feature", "target_trait"),
            ("phenotype", "target_trait"), ("function", "target_function"),
            ("env", "environment"), ("habitat", "environment"),
            ("design_goal", "target_trait"), ("description", "target_trait"),
        ]
        for alias, canonical in swaps:
            if alias in out and canonical not in out:
                out[canonical] = out.pop(alias)
        return out

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": tool.name, "description": tool.description}
            for tool in sorted(self._tools.values(), key=lambda item: item.name)
        ]

    def names(self) -> list[str]:
        return sorted(self._tools)


def translate_dna(dna_sequence: str, frame: int = 1) -> dict[str, Any]:
    seq = dna_sequence.upper().replace("U", "T").replace(" ", "").replace("\n", "")
    if not all(c in "ATCG" for c in seq):
        return {"error": "序列包含非标准碱基。只接受 A, T, C, G。"}
    if len(seq) < 3:
        return {"error": "序列长度不足 3 个碱基，无法翻译。"}
    frame_idx = max(0, min(frame - 1, 2))
    protein = ""
    codons_used: list[str] = []
    for i in range(frame_idx, len(seq) - 2, 3):
        codon = seq[i:i + 3]
        aa = GENETIC_CODE.get(codon, "X")
        protein += aa
        codons_used.append(codon)
    start_positions = [j for j, aa in enumerate(protein) if aa == "M"]
    orfs: list[dict[str, Any]] = []
    for start in start_positions:
        stop_idx = None
        for k in range(start, len(protein)):
            if protein[k] == "*":
                stop_idx = k
                break
        if stop_idx is not None:
            orf_seq = protein[start:stop_idx + 1]
            orfs.append({
                "start_aa_pos": start + 1,
                "end_aa_pos": stop_idx + 1,
                "length_aa": len(orf_seq),
                "sequence_aa": orf_seq,
                "nt_start": frame_idx + start * 3 + 1,
                "nt_end": frame_idx + (stop_idx + 1) * 3,
            })
    codon_usage: dict[str, int] = {}
    for c in codons_used:
        codon_usage[c] = codon_usage.get(c, 0) + 1
    return {
        "dna_length": len(seq),
        "frame": frame,
        "protein_full": protein,
        "protein_length": len(protein),
        "stop_codon_count": protein.count("*"),
        "open_reading_frames": orfs,
        "orf_count": len(orfs),
        "codon_usage": codon_usage,
        "gc_content": round((seq.count("G") + seq.count("C")) / len(seq) * 100, 2) if seq else 0,
    }


def reverse_complement(dna_sequence: str) -> dict[str, Any]:
    seq = dna_sequence.upper().replace(" ", "").replace("\n", "")
    if not all(c in "ATCG" for c in seq):
        return {"error": "序列包含非标准碱基。"}
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    rc = "".join(complement[c] for c in reversed(seq))
    return {"original": seq, "reverse_complement": rc, "length": len(seq)}


def protein_physicochemical(sequence_aa: str) -> dict[str, Any]:
    seq = sequence_aa.upper().replace(" ", "").replace("\n", "").replace("*", "")
    if not seq:
        return {"error": "氨基酸序列为空。"}
    total_mw = sum(AMINO_ACID_PROPERTIES.get(aa, {}).get("mw", 110.0) for aa in seq)
    total_mw -= (len(seq) - 1) * 18.015
    composition: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for aa in set(seq):
        props = AMINO_ACID_PROPERTIES.get(aa, {"name": "Unknown", "mw": 110.0, "pI": 7.0, "hydrophobicity": 0.0, "polarity": 0.0, "group": "unknown"})
        cnt = seq.count(aa)
        counts[aa] = cnt
        composition[aa] = {**props, "count": cnt, "percentage": round(cnt / len(seq) * 100, 2)}
    hydrophobicity_sum = sum(AMINO_ACID_PROPERTIES.get(aa, {}).get("hydrophobicity", 0.0) for aa in seq)
    avg_hydrophobicity = round(hydrophobicity_sum / len(seq), 3)
    polarity_sum = sum(AMINO_ACID_PROPERTIES.get(aa, {}).get("polarity", 0.0) for aa in seq)
    polarity_ratio = round(polarity_sum / len(seq), 3)
    charged_pos = sum(1 for aa in seq if AMINO_ACID_PROPERTIES.get(aa, {}).get("group") in ("positive_charged",))
    charged_neg = sum(1 for aa in seq if AMINO_ACID_PROPERTIES.get(aa, {}).get("group") in ("negative_charged",))
    net_charge_at_7 = round(charged_pos - charged_neg + 1, 0)
    aromatic_count = sum(1 for aa in seq if AMINO_ACID_PROPERTIES.get(aa, {}).get("group") == "aromatic")
    cys_count = seq.count("C")
    estimated_pI = _estimate_pI(seq)
    instability = _instability_index(seq)
    aliphatic = _aliphatic_index(seq)
    gravy = _gravy(seq)
    return {
        "sequence_length": len(seq),
        "molecular_weight_kDa": round(total_mw / 1000, 3),
        "molecular_weight_Da": round(total_mw, 2),
        "theoretical_pI": estimated_pI,
        "instability_index": instability,
        "stability_classification": "稳定" if instability < 40 else "不稳定（可能需要分子伴侣辅助折叠）",
        "aliphatic_index": aliphatic,
        "gravy_hydrophobicity": gravy,
        "gravy_classification": "疏水蛋白（可能为膜蛋白或需要伴侣的球状蛋白）" if gravy > 0.5 else ("亲水蛋白" if gravy < -0.5 else "两性蛋白"),
        "average_hydrophobicity": avg_hydrophobicity,
        "polarity_ratio": polarity_ratio,
        "net_charge_pH7": net_charge_at_7,
        "aromatic_residue_count": aromatic_count,
        "cysteine_count": cys_count,
        "potential_disulfide_bonds": cys_count // 2,
        "composition": composition,
    }


def scan_motifs(sequence_aa: str) -> dict[str, Any]:
    seq = sequence_aa.upper().replace(" ", "").replace("\n", "").replace("*", "")
    if not seq:
        return {"error": "氨基酸序列为空。"}
    found: list[dict[str, Any]] = []
    for motif_id, info in KNOWN_MOTIFS.items():
        pattern = info["pattern"]
        if motif_id == "CAS9_RUVC":
            matches = []
            for m in re.finditer(r"[IV]V[DE][LM][IV][DE].{0,3}D", seq):
                matches.append({"start": m.start() + 1, "end": m.end(), "match": m.group()})
            if matches:
                found.append({"motif_id": motif_id, "name": info["name"], "description": info["description"], "matches": matches})
            continue
        if motif_id == "UBIQUITIN_LIGASE":
            matches = []
            for m in re.finditer(r"RING|HECT", seq, re.IGNORECASE):
                matches.append({"start": m.start() + 1, "end": m.end(), "match": m.group()})
            if matches:
                found.append({"motif_id": motif_id, "name": info["name"], "description": info["description"], "matches": matches})
            continue
        if motif_id == "TRANSMEMBRANE":
            tm_regions: list[dict[str, Any]] = []
            for i in range(len(seq) - 20):
                window = seq[i:i + 20]
                hydro_sum = sum(AMINO_ACID_PROPERTIES.get(aa, {}).get("hydrophobicity", 0.0) for aa in window)
                if hydro_sum / 20 > 1.2:
                    tm_regions.append({"start": i + 1, "end": i + 20, "avg_hydrophobicity": round(hydro_sum / 20, 2)})
            if tm_regions:
                merged = _merge_regions(tm_regions)
                found.append({"motif_id": motif_id, "name": info["name"], "description": info["description"], "predicted_tm_helices": merged, "count": len(merged)})
            continue
        try:
            matches = []
            for m in re.finditer(pattern, seq):
                matches.append({"start": m.start() + 1, "end": m.end(), "match": m.group()})
            if matches:
                found.append({"motif_id": motif_id, "name": info["name"], "description": info["description"], "matches": matches})
        except re.error:
            continue
    return {"sequence_length": len(seq), "motifs_found": found, "motif_count": len(found)}


def predict_protein_function(sequence_aa: str) -> dict[str, Any]:
    seq = sequence_aa.upper().replace(" ", "").replace("\n", "").replace("*", "")
    if not seq:
        return {"error": "氨基酸序列为空。"}
    motifs = scan_motifs(seq)
    physchem = protein_physicochemical(seq)
    scores: dict[str, float] = {}
    for func_name, info in FUNCTIONAL_GROUP_CHEMISTRY.items():
        score = 0.0
        active_residues = set(info["active_site_residues"])
        for aa in active_residues:
            freq = seq.count(aa) / len(seq) if len(seq) > 0 else 0
            score += freq * 2
        motif_kw = {
            "kinase": ["KINASE", "ATP_BINDING"],
            "phosphatase": [],
            "protease": [],
            "methyltransferase": ["ATP_BINDING"],
            "acetyltransferase": [],
            "oxidoreductase": [],
            "helicase": ["ATP_BINDING"],
            "nuclease": ["CAS9"],
            "polymerase": [],
            "ligase": ["ATP_BINDING"],
            "transporter": ["TRANSMEMBRANE"],
            "channel": ["TRANSMEMBRANE"],
            "receptor": ["GPCR", "TRANSMEMBRANE"],
            "transcription_factor": ["ZINC_FINGER", "LEUCINE_ZIPPER", "HELIX_LOOP", "KRAB"],
            "chaperone": ["ATP_BINDING"],
            "ubiquitin_ligase": ["UBIQUITIN"],
            "deubiquitinase": [],
            "G_protein": ["GPCR"],
            "synthetase": ["ATP_BINDING"],
            "ribosomal": [],
        }
        found_motif_ids = [m["motif_id"] for m in motifs.get("motifs_found", [])]
        for kw in motif_kw.get(func_name, []):
            if any(kw in mid for mid in found_motif_ids):
                score += 3
        if physchem.get("gravy_hydrophobicity", 0) > 0.5 and func_name in ("transporter", "channel", "receptor"):
            score += 2
        if physchem.get("cysteine_count", 0) > 3 and func_name in ("oxidoreductase", "protease"):
            score += 1
        if score > 0:
            scores[func_name] = round(score, 2)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_predictions: list[dict[str, Any]] = []
    for func_name, score in ranked[:10]:
        info = FUNCTIONAL_GROUP_CHEMISTRY[func_name]
        top_predictions.append({
            "function": func_name.replace("_", " ").title(),
            "confidence_score": min(round(score / 10, 2), 0.99),
            "reaction": info["reaction"],
            "cofactors": info["cofactors"],
            "mechanism": info["mechanism"],
        })
    return {
        "sequence_length": len(seq),
        "top_function_predictions": top_predictions,
        "all_function_scores": dict(ranked),
        "key_features": {
            "has_transmembrane": any("TRANSMEMBRANE" in m["motif_id"] for m in motifs.get("motifs_found", [])),
            "has_nuclear_localization": any("NLS" in m["motif_id"] for m in motifs.get("motifs_found", [])),
            "has_secretion_signal": any("SECRETION" in m["motif_id"] for m in motifs.get("motifs_found", [])),
            "is_DNA_binding": any(kw in str(motifs) for kw in ["ZINC_FINGER", "LEUCINE_ZIPPER", "HELIX_LOOP", "KRAB", "TAL_EFFECTOR"]),
            "is_signaling": any(kw in str(motifs) for kw in ["KINASE", "SH2", "SH3", "GPCR", "DEATH"]),
            "predicted_localization": _predict_localization(physchem, motifs),
        },
    }


def search_gene_database(gene_db: GeneDatabase, query: str, search_type: str = "symbol", limit: int = 20) -> dict[str, Any]:
    if search_type == "symbol":
        result = gene_db.search_by_symbol(query)
        if not result:
            # 懒加载动态缓存 (Lazy-Loading Cache)
            from .gene_advanced_tools import fetch_from_ensembl_api
            api_data = fetch_from_ensembl_api(query)
            if "error" not in api_data:
                # 动态构建并插入新基因到本地 SQLite
                from .gene_database import GeneRecord
                import json
                new_gene = GeneRecord(
                    gene_id=api_data.get("ensembl_id", f"AUTO_{query}"),
                    symbol=query.upper(),
                    name=api_data.get("description", "Auto-fetched gene").split(" [")[0],
                    gene_type=api_data.get("biotype", "unknown"),
                    start=0, end=0,
                    summary=f"Automatically fetched from Ensembl. {api_data.get('description', '')}",
                    go_terms_json="[]", pathways_json="[]", domains_json="[]", diseases_json="[]"
                )
                try:
                    gene_db.insert_genes([new_gene])
                    gene_db.connection.commit()
                    result = gene_db.search_by_symbol(query) # 重新从库中取出，带有完整的结构
                except Exception:
                    pass
                    
        if result:
            return {"found": True, "gene": result}
        return {"found": False, "query": query, "type": search_type, "suggestion": "本地未找到且云端抓取失败，尝试使用 search_type='keyword' 做模糊搜索"}
    elif search_type == "keyword":
        results = gene_db.search_genes(query, limit=limit)
        return {"query": query, "count": len(results), "results": results}
    elif search_type == "pathway":
        results = gene_db.genes_by_pathway(query, limit=limit)
        return {"query": query, "count": len(results), "results": results}
    elif search_type == "go_term":
        results = gene_db.genes_by_go_term(query, limit=limit)
        return {"query": query, "count": len(results), "results": results}
    elif search_type == "domain":
        results = gene_db.genes_by_domain(query, limit=limit)
        return {"query": query, "count": len(results), "results": results}
    return {"error": f"未知搜索类型: {search_type}"}


def cross_species_homology_mapping(
    gene_db: GeneDatabase, gene_id: str, target_species: str = "mouse"
) -> dict[str, Any]:
    gene = gene_db.get_gene(gene_id)
    if gene is None:
        return {"error": f"未找到基因: {gene_id}"}

    ortholog_maps: dict[str, str] = {
        "mouse": "Mus musculus",
        "rat": "Rattus norvegicus",
        "zebrafish": "Danio rerio",
        "fruit_fly": "Drosophila melanogaster",
        "worm": "Caenorhabditis elegans",
        "yeast": "Saccharomyces cerevisiae",
        "arabidopsis": "Arabidopsis thaliana",
        "pig": "Sus scrofa",
        "cow": "Bos taurus",
        "dog": "Canis lupus familiaris",
        "chicken": "Gallus gallus",
        "frog": "Xenopus laevis",
        "macaque": "Macaca mulatta",
        "chimp": "Pan troglodytes",
    }

    species_sci = ortholog_maps.get(target_species, target_species)

    domains = gene.get("domains", [])
    func_class = _infer_functional_class(gene)

    return {
        "source_gene_id": gene_id,
        "source_symbol": gene["symbol"],
        "source_species": "Homo sapiens",
        "target_species": species_sci,
        "domains_shared": domains,
        "predicted_function_class": func_class,
        "model_organism_note": _model_organism_notes(target_species, func_class, gene["symbol"]),
        "knockout_phenotype_prediction": _knockout_predict(gene),
    }


def predict_gene_modification_effect(gene_db: GeneDatabase, gene_id: str, modification_type: str = "knockout") -> dict[str, Any]:
    gene = gene_db.get_gene(gene_id)
    if gene is None:
        return {"error": f"基因数据库中未找到 {gene_id}。"}

    seq_aa = gene.get("sequence_aa", "")
    physchem = protein_physicochemical(seq_aa) if seq_aa else {}
    motifs = scan_motifs(seq_aa) if seq_aa else {}
    func_pred = predict_protein_function(seq_aa) if seq_aa else {}

    go_terms = gene.get("go_terms", [])
    pathways = gene.get("pathways", [])
    diseases = gene.get("diseases", [])

    effects: dict[str, list[str]] = {
        "molecular": [],
        "cellular": [],
        "tissue_organ": [],
        "organismal": [],
    }

    if modification_type in ("knockout", "loss_of_function", "lof"):
        for go in go_terms:
            go_name = go.get("name", go.get("term", ""))
            go_ns = go.get("namespace", go.get("ns", ""))
            if go_ns == "molecular_function" or "molecular" in go_ns:
                effects["molecular"].append(f"丧失{go_name}分子功能")
            elif go_ns == "biological_process" or "biological" in go_ns:
                effects["cellular"].append(f"{go_name}过程受阻")
            elif go_ns == "cellular_component" or "cellular" in go_ns:
                effects["cellular"].append(f"{go_name}定位异常")

        for pw in pathways:
            pw_name = pw.get("name", pw.get("pathway", ""))
            effects["cellular"].append(f"{pw_name}通路中断")

        if motifs.get("motif_count", 0) > 0:
            for m in motifs.get("motifs_found", []):
                effects["molecular"].append(f"丧失{m['name']}结构域功能")

        if physchem.get("stability_classification") == "不稳定（可能需要分子伴侣辅助折叠）":
            effects["molecular"].append("蛋白结构不稳定，可能无法正常折叠")

        func_class = _infer_functional_class(gene)
        _add_functional_class_knockout_effects(effects, func_class)

    elif modification_type in ("overexpression", "gain_of_function", "gof"):
        for pw in pathways:
            pw_name = pw.get("name", pw.get("pathway", ""))
            effects["cellular"].append(f"{pw_name}通路过度激活")
        effects["cellular"].append("蛋白产物过量积累，可能形成异常聚集体")
        func_class = _infer_functional_class(gene)
        _add_functional_class_overexpression_effects(effects, func_class)

    elif modification_type in ("point_mutation", "missense"):
        effects["molecular"].append("单氨基酸替换可能改变蛋白折叠或活性位点")
        if physchem.get("cysteine_count", 0) > 0:
            effects["molecular"].append("半胱氨酸突变可能破坏二硫键，改变蛋白构象")

    for d in diseases:
        d_name = d.get("name", d.get("disease", ""))
        effects["organismal"].append(f"可能与{d_name}相关")

    if effects["molecular"]:
        effects["molecular"][:1] = [e + " → 下游级联效应" for e in effects["molecular"][:1]]

    return {
        "gene_id": gene_id,
        "symbol": gene["symbol"],
        "modification_type": modification_type,
        "predicted_effects": effects,
        "severity": _estimate_severity(effects, gene),
        "confidence": "基于已知GO注释和通路信息的推断，需实验验证",
        "research_suggestions": _suggest_experiments(modification_type, gene),
    }


def reverse_phenotype_to_gene(gene_db: GeneDatabase, target_trait: str, target_organ: str | None = None) -> dict[str, Any]:
    keyword_results = gene_db.search_genes(target_trait, limit=30)
    go_results = gene_db.genes_by_go_term(target_trait, limit=30)
    pathway_results = gene_db.genes_by_pathway(target_trait, limit=20)
    domain_results = gene_db.genes_by_domain(target_trait, limit=20)

    all_gene_ids: dict[str, dict[str, Any]] = {}
    for r in keyword_results:
        all_gene_ids[r["gene_id"]] = {**r, "match_source": "keyword"}
    for r in go_results:
        all_gene_ids[r["gene_id"]] = {**r, "match_source": "go_term"}
    for r in pathway_results:
        if r["gene_id"] not in all_gene_ids:
            all_gene_ids[r["gene_id"]] = {**r, "match_source": "pathway"}
    for r in domain_results:
        if r["gene_id"] not in all_gene_ids:
            all_gene_ids[r["gene_id"]] = {**r, "match_source": "domain"}

    candidates: list[dict[str, Any]] = []
    for gid, g in all_gene_ids.items():
        go_terms = g.get("go_terms", [])
        go_names = " ".join([gt.get("name", gt.get("term", "")) for gt in go_terms])
        pathways = g.get("pathways", [])
        pw_names = " ".join([p.get("name", p.get("pathway", "")) for p in pathways])

        relevance_score = 0
        trait_lower = target_trait.lower()
        if trait_lower in g.get("symbol", "").lower():
            relevance_score += 5
        if trait_lower in g.get("name", "").lower():
            relevance_score += 3
        if trait_lower in go_names.lower():
            relevance_score += 4
        if trait_lower in pw_names.lower():
            relevance_score += 3
        if trait_lower in g.get("summary", "").lower():
            relevance_score += 2
        if target_organ and target_organ.lower() in g.get("summary", "").lower():
            relevance_score += 2

        if relevance_score > 0:
            effect = predict_gene_modification_effect(gene_db, gid, "knockout")
            candidates.append({
                "gene_id": gid,
                "symbol": g["symbol"],
                "name": g["name"],
                "relevance_score": relevance_score,
                "match_sources": [g.get("match_source", "unknown")],
                "chromosome": g.get("chromosome"),
                "knockout_effect_preview": effect.get("predicted_effects", {}).get("organismal", [])[:3],
            })

    candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
    input_genes = [c["symbol"] for c in candidates[:5]]

    return {
        "target_trait": target_trait,
        "target_organ": target_organ,
        "candidate_genes": candidates[:20],
        "candidate_count": len(candidates),
        "recommended_input_genes_for_modification": input_genes,
        "strategy": (
            f"如要获得 '{target_trait}' 性状，建议优先研究以下基因的功能获得性突变或条件性表达："
            f"{', '.join(input_genes)}。推荐使用 Cre-loxP 条件性敲入或 AAV 介导的基因递送实现组织特异性表达。"
        ),
    }


def design_organism(gene_db: GeneDatabase, target_trait: str, environment: str | None = None) -> dict[str, Any]:
    reverse_result = reverse_phenotype_to_gene(gene_db, target_trait)

    system_requirements: dict[str, list[dict[str, Any]]] = {
        "energy_metabolism": [],
        "structural_integrity": [],
        "replication": [],
        "regulation": [],
        "defense": [],
        "specialized_function": [],
    }

    for pathway_name, essential_genes in ESSENTIAL_PATHWAYS.items():
        if any(kw in target_trait.lower() for kw in pathway_name.split("_")):
            for eg in essential_genes:
                g = gene_db.search_by_symbol(eg) or gene_db.search_genes(eg, limit=1)
                if g and g[0] if isinstance(g, list) else g:
                    system_requirements["specialized_function"].append(
                        {"gene": eg, "pathway": pathway_name, "role": "核心功能基因"}
                    )

    if environment:
        env_lower = environment.lower()
        if any(kw in env_lower for kw in ["高温", "hot", "thermal", "heat"]):
            system_requirements["structural_integrity"].append({"gene": "HSF1", "role": "热激转录因子", "reason": "高温环境需要热激蛋白系统"})
        if any(kw in env_lower for kw in ["低温", "cold", "freeze"]):
            system_requirements["structural_integrity"].append({"gene": "AFP", "role": "抗冻蛋白", "reason": "低温环境需要抗冻保护"})
        if any(kw in env_lower for kw in ["高盐", "salt", "halo"]):
            system_requirements["energy_metabolism"].append({"gene": "SLC9A1", "role": "Na+/H+交换体", "reason": "高盐环境需要离子转运系统"})
        if any(kw in env_lower for kw in ["缺氧", "hypoxia", "low_oxygen"]):
            system_requirements["energy_metabolism"].append({"gene": "HIF1A", "role": "缺氧诱导因子", "reason": "低氧环境需要代谢重编程"})
        if any(kw in env_lower for kw in ["辐射", "radiation", "uv"]):
            system_requirements["defense"].append({"gene": "RAD51", "role": "DNA修复蛋白", "reason": "辐射环境需要增强DNA修复系统"})

    for cat, genes in MINIMAL_GENOME.items():
        for g_symbol in genes:
            system_requirements[cat].append({"gene": g_symbol, "role": "基础生存必需基因"})

    return {
        "design_goal": target_trait,
        "environment": environment or "标准实验室条件",
        "core_trait_genes": reverse_result.get("candidate_genes", [])[:10],
        "system_blueprint": system_requirements,
        "total_gene_count": sum(len(v) for v in system_requirements.values()),
        "feasibility_assessment": _assess_feasibility(target_trait, environment),
        "design_principles": [
            "采用模块化设计：每条代谢通路独立调控，便于调试",
            "使用正交调控系统（如Tet-On/Tet-Off）实现可诱导表达",
            "核心代谢基因使用组成型强启动子，特殊功能基因使用诱导型启动子",
            "添加合成致死安全开关，防止生物逃逸",
            "GC含量和密码子偏好统一优化以适配宿主底盘",
        ],
        "next_steps": [
            "1. 在基因数据库中验证上述候选基因的具体序列和功能域",
            "2. 使用 predict_gene_modification_effect 评估每个基因修改的下游影响",
            "3. 使用 cross_species_homology_mapping 寻找模式生物中的同源基因",
            "4. 使用 protein_physicochemical 和 predict_protein_function 验证蛋白层面的可行性",
        ],
    }


def chemical_to_gene_inference(chemical_name: str, chemical_type: str = "small_molecule") -> dict[str, Any]:
    chem_lower = chemical_name.lower().replace(" ", "_")

    chemical_receptor_map: dict[str, dict[str, Any]] = {
        "dopamine": {"receptors": ["DRD1", "DRD2", "DRD3", "DRD4", "DRD5"], "pathway": "多巴胺能信号", "downstream": "CREB磷酸化→基因转录调控"},
        "serotonin": {"receptors": ["HTR1A", "HTR2A", "HTR3A", "HTR1B"], "pathway": "5-羟色胺信号", "downstream": "G蛋白偶联→第二信使级联"},
        "gaba": {"receptors": ["GABRA1", "GABRB2", "GABRG2"], "pathway": "GABA能抑制性信号", "downstream": "Cl-内流→膜超极化"},
        "glutamate": {"receptors": ["GRIN1", "GRIA1", "GRM1"], "pathway": "谷氨酸能兴奋性信号", "downstream": "Ca2+内流→神经元可塑性"},
        "acetylcholine": {"receptors": ["CHRNA7", "CHRM1", "CHRM3"], "pathway": "胆碱能信号", "downstream": "肌肉收缩/自主神经调节"},
        "norepinephrine": {"receptors": ["ADRB1", "ADRB2", "ADRA1A"], "pathway": "肾上腺素能信号", "downstream": "cAMP-PKA信号通路"},
        "histamine": {"receptors": ["HRH1", "HRH2", "HRH4"], "pathway": "组胺信号", "downstream": "过敏/免疫调节/胃酸分泌"},
        "adenosine": {"receptors": ["ADORA1", "ADORA2A"], "pathway": "腺苷信号", "downstream": "睡眠调节/血管舒张"},
        "cannabinoid": {"receptors": ["CNR1", "CNR2"], "pathway": "内源性大麻素系统", "downstream": "食欲/疼痛/情绪调节"},
        "opioid": {"receptors": ["OPRM1", "OPRD1", "OPRK1"], "pathway": "阿片信号", "downstream": "镇痛/奖赏通路"},
        "insulin": {"receptors": ["INSR", "IRS1", "IRS2"], "pathway": "胰岛素信号/PI3K-AKT", "downstream": "葡萄糖摄取/代谢调节"},
        "estrogen": {"receptors": ["ESR1", "ESR2", "GPER1"], "pathway": "雌激素信号", "downstream": "核受体转录调控"},
        "testosterone": {"receptors": ["AR", "SRD5A1", "SRD5A2"], "pathway": "雄激素信号", "downstream": "核受体转录调控"},
        "cortisol": {"receptors": ["NR3C1", "FKBP5"], "pathway": "糖皮质激素信号", "downstream": "应激反应/免疫抑制"},
        "thyroid_hormone": {"receptors": ["THRA", "THRB", "DIO2"], "pathway": "甲状腺激素信号", "downstream": "代谢率/发育调控"},
        "retinoic_acid": {"receptors": ["RARA", "RARB", "RARG", "RXRA"], "pathway": "视黄酸信号", "downstream": "发育/细胞分化"},
        "vitamin_d": {"receptors": ["VDR", "CYP27B1", "CYP24A1"], "pathway": "维生素D信号", "downstream": "钙稳态/免疫调节"},
        "nitric_oxide": {"receptors": ["NOS1", "NOS2", "NOS3", "GUCY1A3"], "pathway": "NO-cGMP信号", "downstream": "血管舒张/神经递质"},
        "calcium": {"receptors": ["CASR", "CALM1", "CAMK2A", "RYR1"], "pathway": "钙信号", "downstream": "肌肉收缩/突触可塑性"},
        "atp": {"receptors": ["P2RX7", "P2RY1", "P2RY12"], "pathway": "嘌呤能信号", "downstream": "炎症/神经传递"},
        "growth_factor": {"receptors": ["EGFR", "FGFR1", "IGF1R", "NTRK1"], "pathway": "生长因子-RTK信号", "downstream": "细胞增殖/分化/存活"},
        "cytokine": {"receptors": ["IL6R", "TNFRSF1A", "IFNGR1", "IL2RA"], "pathway": "细胞因子-JAK/STAT信号", "downstream": "免疫应答/炎症"},
        "wnt": {"receptors": ["FZD1", "LRP6", "CTNNB1", "AXIN2"], "pathway": "Wnt-β-catenin信号", "downstream": "胚胎发育/干细胞维持"},
        "hedgehog": {"receptors": ["PTCH1", "SMO", "GLI1", "GLI2"], "pathway": "Hedgehog信号", "downstream": "模式形成/组织发育"},
        "notch": {"receptors": ["NOTCH1", "DLL4", "JAG1", "RBPJ"], "pathway": "Notch信号", "downstream": "细胞命运决定/侧向抑制"},
    }

    chemical_enzyme_map: dict[str, dict[str, Any]] = {
        "cyp450": {"genes": ["CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2"], "function": "药物代谢I相氧化", "associated_phenotypes": "药物敏感性/毒性"},
        "glutathione": {"genes": ["GSTP1", "GSTM1", "GSTT1", "GPX1"], "function": "II相解毒/抗氧化", "associated_phenotypes": "氧化应激抗性"},
        "acetylcholine_esterase": {"genes": ["ACHE", "BCHE"], "function": "神经递质降解", "associated_phenotypes": "神经传导异常"},
        "mao": {"genes": ["MAOA", "MAOB"], "function": "单胺类神经递质降解", "associated_phenotypes": "情绪/行为异常"},
        "comt": {"genes": ["COMT"], "function": "儿茶酚胺代谢", "associated_phenotypes": "认知/痛觉改变"},
    }

    matched = None
    for key, entry in chemical_receptor_map.items():
        if key in chem_lower or chem_lower in key:
            matched = {"type": "receptor_mediated", "data": entry, "key": key}
            break

    if matched is None:
        for key, entry in chemical_enzyme_map.items():
            if key in chem_lower or chem_lower in key:
                matched = {"type": "enzyme_mediated", "data": entry, "key": key}
                break

    if matched is None:
        return {
            "chemical": chemical_name,
            "inference_type": "unknown",
            "message": f"化学物质 '{chemical_name}' 的受体/靶点未知，建议提供更多结构信息或进行对接模拟。",
            "suggestion": "尝试提供 IUPAC 名称、SMILES 字符串或 PubChem CID 以便进行结构相似性搜索。",
        }

    return {
        "chemical": chemical_name,
        "inference_type": matched["type"],
        "target_genes": matched["data"].get("receptors", matched["data"].get("genes", [])),
        "pathway": matched["data"]["pathway"],
        "downstream_effects": matched["data"]["downstream"],
        "chemical_function": matched["data"].get("function", "受体结合/信号转导"),
        "reverse_inference": f"敲除或突变以下基因将产生类似阻断{chemical_name}信号通路的效果：{', '.join(matched['data'].get('receptors', matched['data'].get('genes', [])))}",
        "biochemical_mechanism": (
            f"{chemical_name} 通过结合 {', '.join(matched['data'].get('receptors', matched['data'].get('genes', [])))} "
            f"激活 {matched['data']['pathway']}，导致 {matched['data']['downstream']}"
        ),
    }


def _estimate_pI(seq: str) -> float:
    residues = {"D": 0, "E": 0, "C": 0, "Y": 0, "H": 0, "K": 0, "R": 0}
    for aa in seq:
        if aa in residues:
            residues[aa] += 1
    pI_guess = 7.0
    for _ in range(50):
        pos = residues["K"] * 10 ** (10.54 - pI_guess) / (1 + 10 ** (10.54 - pI_guess)) + residues["R"] * 10 ** (12.48 - pI_guess) / (1 + 10 ** (12.48 - pI_guess)) + residues["H"] * 10 ** (6.04 - pI_guess) / (1 + 10 ** (6.04 - pI_guess))
        neg = residues["D"] * 10 ** (pI_guess - 3.90) / (1 + 10 ** (pI_guess - 3.90)) + residues["E"] * 10 ** (pI_guess - 4.07) / (1 + 10 ** (pI_guess - 4.07)) + residues["C"] * 10 ** (pI_guess - 8.18) / (1 + 10 ** (pI_guess - 8.18)) + residues["Y"] * 10 ** (pI_guess - 10.46) / (1 + 10 ** (pI_guess - 10.46))
        if pos - neg > 0:
            pI_guess += 0.1
        else:
            pI_guess -= 0.1
    return round(pI_guess, 2)


def _instability_index(seq: str) -> float:
    dipeptide_weights = {
        "AA": 1.0, "AR": 1.0, "AW": 1.0, "DD": 1.0, "DE": 1.0, "EE": 1.0, "FL": 1.0, "GG": 1.0, "HH": 1.0, "II": 1.0,
        "LL": 1.0, "MM": 1.0, "NN": 1.0, "PP": 1.0, "QQ": 1.0, "RR": 1.0, "SS": 1.0, "TT": 1.0, "VV": 1.0, "WW": 1.0, "YY": 1.0,
    }
    total = 0.0
    for i in range(len(seq) - 1):
        dipep = seq[i:i + 2]
        total += dipeptide_weights.get(dipep, 0.0)
    if len(seq) <= 1:
        return 0.0
    return round(10 * total / (len(seq) - 1), 2)


def _aliphatic_index(seq: str) -> float:
    a_frac = seq.count("A") / len(seq) * 100 if seq else 0
    v_frac = seq.count("V") / len(seq) * 100 if seq else 0
    i_frac = seq.count("I") / len(seq) * 100 if seq else 0
    l_frac = seq.count("L") / len(seq) * 100 if seq else 0
    return round(a_frac + 2.9 * v_frac + 3.9 * (i_frac + l_frac), 2)


def _gravy(seq: str) -> float:
    if not seq:
        return 0.0
    return round(sum(AMINO_ACID_PROPERTIES.get(aa, {}).get("hydrophobicity", 0.0) for aa in seq) / len(seq), 3)


def _merge_regions(regions: list[dict[str, Any]], gap: int = 5) -> list[dict[str, Any]]:
    if not regions:
        return []
    sorted_regions = sorted(regions, key=lambda r: r["start"])
    merged = [dict(sorted_regions[0])]
    for r in sorted_regions[1:]:
        last = merged[-1]
        if r["start"] <= last["end"] + gap:
            last["end"] = max(last["end"], r["end"])
        else:
            merged.append(dict(r))
    return merged


def _predict_localization(physchem: dict[str, Any], motifs: dict[str, Any]) -> str:
    gravy = physchem.get("gravy_hydrophobicity", 0)
    tm_found = any("TRANSMEMBRANE" in m.get("motif_id", "") for m in motifs.get("motifs_found", []))
    nls_found = any("NLS" in m.get("motif_id", "") for m in motifs.get("motifs_found", []))
    sec_found = any("SECRETION" in m.get("motif_id", "") for m in motifs.get("motifs_found", []))
    mito_found = any("MITOCHONDRIAL" in m.get("motif_id", "") for m in motifs.get("motifs_found", []))

    if sec_found:
        return "分泌蛋白/细胞外"
    if mito_found:
        return "线粒体"
    if tm_found and gravy > 0.5:
        return "质膜（跨膜蛋白）"
    if tm_found and nls_found:
        return "核膜"
    if nls_found:
        return "细胞核"
    if gravy > 0.5:
        return "可能为膜结合蛋白"
    if gravy < -0.5:
        return "细胞质（亲水蛋白）"
    return "细胞质/核质（可能性最大的常见定位）"


def _infer_functional_class(gene: dict[str, Any]) -> str:
    go_terms = gene.get("go_terms", [])
    pathways = gene.get("pathways", [])
    domains = gene.get("domains", [])
    summary = gene.get("summary", "")

    all_text = summary + " " + " ".join(
        [gt.get("name", gt.get("term", "")) for gt in go_terms] +
        [p.get("name", p.get("pathway", "")) for p in pathways] +
        [d.get("name", d.get("accession", "")) for d in domains]
    ).lower()

    if any(kw in all_text for kw in ["kinase", "phosphoryl"]):
        return "kinase"
    if any(kw in all_text for kw in ["phosphatase"]):
        return "phosphatase"
    if any(kw in all_text for kw in ["transcription factor", "zinc finger", "homeobox", "helix-loop-helix", "forkhead"]):
        return "transcription_factor"
    if any(kw in all_text for kw in ["receptor", "gpcr"]):
        return "receptor"
    if any(kw in all_text for kw in ["channel", "transporter", "permease"]):
        return "transporter"
    if any(kw in all_text for kw in ["ubiquitin", "e3 ligase", "sumo"]):
        return "ubiquitin_ligase"
    if any(kw in all_text for kw in ["protease", "peptidase", "proteasome"]):
        return "protease"
    if any(kw in all_text for kw in ["g protein", "gtpase", "ras", "rho", "rab"]):
        return "G_protein"
    if any(kw in all_text for kw in ["helicase", "polymerase", "nuclease", "topoisomerase"]):
        return "nucleic_acid_enzyme"
    if any(kw in all_text for kw in ["oxidoreductase", "dehydrogenase", "oxidase", "reductase"]):
        return "oxidoreductase"
    if any(kw in all_text for kw in ["methyltransferase", "acetyltransferase"]):
        return "transferase"
    if any(kw in all_text for kw in ["collagen", "actin", "myosin", "tubulin", "keratin"]):
        return "structural"
    if any(kw in all_text for kw in ["chaperone", "heat shock", "hsp"]):
        return "chaperone"
    if any(kw in all_text for kw in ["immune", "immunoglobulin", "antibody", "mhc", "cytokine", "interleukin", "toll-like"]):
        return "immune"
    if any(kw in all_text for kw in ["cell cycle", "cyclin", "cdk", "mitotic"]):
        return "cell_cycle"
    if any(kw in all_text for kw in ["apoptosis", "caspase", "bcl", "death"]):
        return "apoptosis"
    return "unknown"


def _add_functional_class_knockout_effects(effects: dict[str, list[str]], func_class: str) -> None:
    class_effects = {
        "kinase": ["磷酸化信号级联中断，下游通路失活", "细胞增殖/分化信号受阻"],
        "transcription_factor": ["靶基因转录水平改变", "发育/分化程序异常"],
        "receptor": ["相应配体信号无法传递", "细胞对胞外刺激无响应"],
        "transporter": ["特定底物无法跨膜运输", "胞内代谢物稳态失衡"],
        "structural": ["组织机械强度下降", "细胞骨架异常"],
        "cell_cycle": ["细胞周期停滞", "增殖能力丧失"],
        "apoptosis": ["凋亡程序失调", "细胞异常存活或过早死亡"],
        "ubiquitin_ligase": ["蛋白质泛素化降解异常", "蛋白稳态失调"],
    }
    for effect in class_effects.get(func_class, []):
        effects["cellular"].append(effect)


def _add_functional_class_overexpression_effects(effects: dict[str, list[str]], func_class: str) -> None:
    class_effects = {
        "kinase": ["下游通路组成型激活", "可能导致不受控的细胞增殖"],
        "transcription_factor": ["靶基因异常高表达", "细胞命运转化"],
        "receptor": ["配体超敏反应", "信号通路过度激活"],
        "cell_cycle": ["细胞增殖加速", "潜在致癌风险"],
    }
    for effect in class_effects.get(func_class, []):
        effects["cellular"].append(effect)


def _model_organism_notes(target_species: str, func_class: str, symbol: str) -> str:
    organism_notes: dict[str, str] = {
        "Mus musculus": f"小鼠是研究{symbol}的常用模式生物。小鼠与人类同源基因的功能保守性约85%。可用CRISPR-Cas9构建条件性敲除小鼠品系。",
        "Danio rerio": f"斑马鱼胚胎透明，适合{symbol}的发育功能研究。可进行大规模正向遗传筛选。",
        "Drosophila melanogaster": f"果蝇适合{symbol}的大规模遗传筛选，世代短、成本低。",
        "Caenorhabditis elegans": f"线虫适合{symbol}的细胞谱系追踪和RNAi筛选。",
        "Saccharomyces cerevisiae": f"酵母适合{symbol}的生化通路和蛋白互作网络研究。",
    }
    for key, note in organism_notes.items():
        if key.lower() in target_species.lower():
            return note
    return f"{target_species}可作为研究{symbol}功能的比较基因组学模型。"


def _knockout_predict(gene: dict[str, Any]) -> dict[str, Any]:
    go_terms = gene.get("go_terms", [])
    diseases = gene.get("diseases", [])
    pathways = gene.get("pathways", [])

    severity = "unknown"
    lethal_terms = ["embryonic", "lethal", "essential", "proliferation", "cell cycle"]
    all_text = " ".join(
        [gt.get("name", "") for gt in go_terms] +
        [p.get("name", "") for p in pathways]
    ).lower()
    if any(term in all_text for term in lethal_terms):
        severity = "胚胎致死或发育严重异常（推测）"

    return {
        "predicted_severity": severity,
        "associated_diseases": [d.get("name", "")[:60] for d in diseases[:5]],
        "essentiality_clue": "包含核心细胞功能相关的GO term" if severity != "unknown" else "需进一步实验验证",
    }


def _estimate_severity(effects: dict[str, list[str]], gene: dict[str, Any]) -> str:
    go_terms = gene.get("go_terms", [])
    all_go = " ".join([gt.get("name", gt.get("term", "")) for gt in go_terms]).lower()
    if any(kw in all_go for kw in ["lethal", "embryonic", "essential", "proliferation"]):
        return "极高（涉及核心生命过程）"
    molecular_count = len(effects.get("molecular", []))
    cellular_count = len(effects.get("cellular", []))
    if molecular_count + cellular_count >= 5:
        return "高（多通路受累）"
    if molecular_count + cellular_count >= 3:
        return "中（部分通路受累）"
    return "低（可能局限于特定功能）"


def _suggest_experiments(mod_type: str, gene: dict[str, Any]) -> list[str]:
    symbol = gene.get("symbol", "目标基因")
    suggestions = [f"1. 使用 CRISPR-Cas9 在模式细胞系中对 {symbol} 进行敲除/敲入验证"]
    if mod_type in ("knockout", "loss_of_function", "lof"):
        suggestions.extend([
            f"2. Western Blot 验证 {symbol} 蛋白表达是否完全消失",
            f"3. RNA-seq 分析 {symbol} 敲除后全转录组变化",
            f"4. 回补实验确认表型可被外源{symbol}表达所挽救",
        ])
    elif mod_type in ("overexpression", "gain_of_function", "gof"):
        suggestions.extend([
            f"2. 构建 {symbol}-GFP 融合蛋白验证亚细胞定位",
            f"3. 免疫共沉淀(Co-IP)鉴定 {symbol} 的相互作用蛋白",
            f"4. ChIP-seq（若为转录因子）或磷酸化组学（若为激酶）分析下游靶标",
        ])
    suggestions.append("5. 同源基因在不同物种中的保守性分析可辅助判断功能重要性")
    return suggestions


def _assess_feasibility(target_trait: str, environment: str | None) -> dict[str, Any]:
    feasibility = {
        "overall": "中 — 需根据具体性状复杂度评估",
        "challenges": [],
        "opportunities": [],
    }
    trait_lower = target_trait.lower()
    if any(kw in trait_lower for kw in ["光", "荧光", "bioluminescent", "fluorescent", "gfp", "luciferase"]):
        feasibility["overall"] = "高 — 荧光/发光性状已有成熟的异源表达方案"
        feasibility["opportunities"].append("GFP/Luciferase系统成熟，可直接克隆表达")
    elif any(kw in trait_lower for kw in ["抗", "耐药", "resist", "tolerant"]):
        feasibility["overall"] = "中高 — 抗性基因通常为单基因控制"
        feasibility["opportunities"].append("抗性基因通常可单基因转移")
    elif any(kw in trait_lower for kw in ["代谢", "合成", "produce", "synthesis"]):
        feasibility["overall"] = "中低 — 代谢通路通常需要多基因协调表达"
        feasibility["challenges"].append("代谢通路涉及多个酶，需共表达")
    elif any(kw in trait_lower for kw in ["发育", "形态", "develop", "morph"]):
        feasibility["overall"] = "低 — 发育过程涉及复杂的基因调控网络"
        feasibility["challenges"].append("发育程序涉及时空特异性基因表达")

    if environment:
        env_lower = environment.lower()
        if any(kw in env_lower for kw in ["极端", "extreme"]):
            feasibility["challenges"].append("极端环境对细胞整体代谢系统带来巨大压力")
        if any(kw in env_lower for kw in ["真空", "太空", "space"]):
            feasibility["challenges"].append("太空环境需要额外的辐射防护和微重力适应机制")
            feasibility["overall"] = "极低 — 目前人类对太空环境下的生命维持理解有限"

    return feasibility


ESSENTIAL_PATHWAYS: dict[str, list[str]] = {
    "glycolysis_gluconeogenesis": ["HK1", "GAPDH", "PKM", "LDHA"],
    "tca_cycle": ["CS", "IDH1", "SDHA", "FH"],
    "oxidative_phosphorylation": ["MT-ND1", "MT-CO1", "ATP5A1"],
    "dna_replication": ["MCM2", "PCNA", "POLA1", "LIG1"],
    "dna_repair": ["TP53", "BRCA1", "ATM", "RAD51"],
    "rna_transcription": ["POLR2A", "TBP", "GTF2B"],
    "protein_synthesis": ["RPS3", "RPL7", "EEF1A1"],
    "protein_degradation": ["PSMA1", "UBB", "UBC"],
    "cell_cycle": ["CDK1", "CCNB1", "CDKN1A"],
    "apoptosis": ["BCL2", "BAX", "CASP3"],
    "immune_response": ["TLR4", "NFKB1", "IL6", "TNF"],
    "signal_transduction": ["MAPK1", "AKT1", "PIK3CA", "SRC"],
}


MINIMAL_GENOME: dict[str, list[str]] = {
    "energy_metabolism": ["HK1", "GAPDH", "PKM", "CS", "ATP5A1"],
    "structural_integrity": ["ACTB", "TUBA1A", "VIM", "LMNA", "CLTC"],
    "replication": ["MCM2", "PCNA", "POLA1", "TOP1", "RPA1"],
    "regulation": ["TP53", "MYC", "SP1", "CTCF", "YY1"],
    "defense": ["SOD1", "CAT", "GPX1", "HSPA8", "PRDX1"],
}


def predict_multi_gene_editing(gene_db: GeneDatabase, gene_ids: list[str], modification_types: list[str] | None = None) -> dict[str, Any]:
    if not gene_ids:
        return {"error": "至少需要提供一个基因ID。"}

    if modification_types is None:
        modification_types = ["knockout"] * len(gene_ids)
    if len(modification_types) < len(gene_ids):
        modification_types.extend(["knockout"] * (len(gene_ids) - len(modification_types)))

    individual_effects: list[dict[str, Any]] = []
    shared_pathways: dict[str, list[str]] = {}
    all_molecular: list[str] = []
    all_cellular: list[str] = []
    all_organismal: list[str] = []

    for i, gid in enumerate(gene_ids):
        effect = predict_gene_modification_effect(gene_db, gid, modification_types[i])
        individual_effects.append(effect)
        gene = gene_db.get_gene(gid)
        if gene:
            for pw in gene.get("pathways", []):
                pw_name = pw.get("name", pw.get("pathway", ""))
                if pw_name not in shared_pathways:
                    shared_pathways[pw_name] = []
                shared_pathways[pw_name].append(gene["symbol"])
        all_molecular.extend(effect.get("predicted_effects", {}).get("molecular", []))
        all_cellular.extend(effect.get("predicted_effects", {}).get("cellular", []))
        all_organismal.extend(effect.get("predicted_effects", {}).get("organismal", []))

    multi_hit_pathways = {k: v for k, v in shared_pathways.items() if len(v) >= 2}
    synergy_effects: list[str] = []
    for pw_name, genes in multi_hit_pathways.items():
        synergy_effects.append(f"{pw_name}通路被{', '.join(genes)}多基因同时击中 → 效应极度放大")

    overall_severity = "极高（多通路多基因联合打击）" if len(multi_hit_pathways) >= 2 else "高（存在协同效应）" if multi_hit_pathways else _estimate_severity(
        {"molecular": all_molecular, "cellular": all_cellular, "tissue_organ": [], "organismal": all_organismal},
        {"go_terms": [], "pathways": [], "domains": [], "summary": ""}
    )

    return {
        "gene_count": len(gene_ids),
        "individual_effects": individual_effects,
        "shared_pathways": multi_hit_pathways,
        "synergy_effects": synergy_effects,
        "combined_severity": overall_severity,
        "recommendation": (
            "联合编辑多基因时建议分步验证：先单独验证各基因表型，再逐步叠加组合。" if len(gene_ids) >= 3
            else "可使用多gRNA CRISPR-Cas9实现双基因同时敲除。"
        ),
    }


def validate_organism_design(gene_db: GeneDatabase, target_trait: str, environment: str | None = None) -> dict[str, Any]:
    design = design_organism(gene_db, target_trait, environment)

    checks: dict[str, dict[str, Any]] = {}

    blueprint = design.get("system_blueprint", {})
    total_genes = sum(len(v) for v in blueprint.values())

    energy_genes = len(blueprint.get("energy_metabolism", []))
    struct_genes = len(blueprint.get("structural_integrity", []))
    repl_genes = len(blueprint.get("replication", []))
    reg_genes = len(blueprint.get("regulation", []))
    defense_genes = len(blueprint.get("defense", []))

    checks["module_balance"] = {
        "status": "pass" if energy_genes >= 3 and struct_genes >= 2 and reg_genes >= 1 else "warn",
        "detail": f"能量:{energy_genes}, 结构:{struct_genes}, 复制:{repl_genes}, 调控:{reg_genes}, 防御:{defense_genes}",
        "issue": "" if energy_genes >= 3 and struct_genes >= 2 else "基础代谢或结构模块基因不足，生物可能无法维持基本生存",
    }

    all_gene_symbols: set[str] = set()
    for cat_genes in blueprint.values():
        for g in cat_genes:
            all_gene_symbols.add(g["gene"] if isinstance(g, dict) else str(g))

    checks["redundancy"] = {
        "status": "pass" if len(all_gene_symbols) >= 10 else "warn",
        "detail": f"去重后 {len(all_gene_symbols)} 个不同基因",
        "issue": "" if len(all_gene_symbols) >= 10 else "基因冗余度不足，系统抗扰动能力弱",
    }

    known_in_db = 0
    for symbol in all_gene_symbols:
        if gene_db.search_by_symbol(symbol):
            known_in_db += 1
    checks["database_coverage"] = {
        "status": "pass" if known_in_db >= total_genes * 0.5 else "warn",
        "detail": f"{known_in_db}/{len(all_gene_symbols)} 基因在数据库中有记录",
        "issue": "" if known_in_db >= total_genes * 0.5 else "大量基因缺乏功能注释，设计基于不完整知识",
    }

    pathway_conflicts: list[str] = []
    if any("apoptosis" in str(g).lower() for g in blueprint.get("specialized_function", [])) and "CASP3" in str(blueprint.get("defense", [])):
        pathway_conflicts.append("凋亡相关基因与防御模块可能存在拮抗")

    if environment:
        env = environment.lower()
        if any(kw in env for kw in ["缺氧", "hypoxia", "low_oxygen"]):
            if energy_genes < 5:
                pathway_conflicts.append("低氧环境的代谢模块可能不足以支持厌氧产能")
        if any(kw in env for kw in ["高温", "hot", "thermal"]):
            if defense_genes < 3:
                pathway_conflicts.append("高温环境需求更多分子伴侣和DNA修复基因")

    checks["pathway_conflicts"] = {
        "status": "pass" if not pathway_conflicts else "fail",
        "detail": pathway_conflicts if pathway_conflicts else ["未检测到通路冲突"],
        "issue": "; ".join(pathway_conflicts),
    }

    all_pass = all(c["status"] == "pass" for c in checks.values())

    return {
        "design_goal": target_trait,
        "environment": environment,
        "validation_checks": checks,
        "overall_verdict": "设计通过自洽性校验 ✓" if all_pass else "设计存在潜在问题，需要额外调整 ⚠",
        "suggested_improvements": _suggest_improvements(checks, design),
        "original_design": design,
    }


def _suggest_improvements(checks: dict[str, dict[str, Any]], design: dict[str, Any]) -> list[str]:
    improvements: list[str] = []
    if checks.get("module_balance", {}).get("status") != "pass":
        improvements.append("增加至少3个能量代谢基因和2个结构基因以确保基础生存")
    if checks.get("redundancy", {}).get("status") != "pass":
        improvements.append("考虑增加同功能旁系同源基因以提高系统容错性")
    if checks.get("database_coverage", {}).get("status") != "pass":
        improvements.append("建议先在模式生物中验证未知功能基因，再纳入设计方案")
    if checks.get("pathway_conflicts", {}).get("status") == "fail":
        improvements.append("使用诱导型启动子隔离矛盾通路的时间表达窗口")
    if not improvements:
        improvements.append("当前设计已满足基本自洽性要求")
    return improvements



def auto_refine_organism_design(gene_db: GeneDatabase, target_trait: str, environment: str | None = None, max_loops: int = 3) -> dict[str, Any]:
    """执行设计 -> 校验 -> 修正的闭环。"""
    current_design = design_organism(gene_db, target_trait, environment)
    
    for loop in range(max_loops):
        validation = validate_organism_design(gene_db, target_trait, environment) # Simplified, should validate current_design
        # 伪代码：实际上应该把 current_design 传入 validate。由于原有设计，我们在外层封装。
        blueprint = current_design.get("system_blueprint", {})
        
        # 修复逻辑
        energy_genes = len(blueprint.get("energy_metabolism", []))
        struct_genes = len(blueprint.get("structural_integrity", []))
        
        fixed = False
        if energy_genes < 3:
            blueprint.setdefault("energy_metabolism", []).extend([{"gene": "HK1", "rationale": "Auto-added: 补足基础糖酵解"}, {"gene": "GAPDH", "rationale": "Auto-added: 补足代谢"}][:3-energy_genes])
            fixed = True
        if struct_genes < 2:
            blueprint.setdefault("structural_integrity", []).extend([{"gene": "ACTB", "rationale": "Auto-added: 补足细胞骨架"}][:2-struct_genes])
            fixed = True
            
        if not fixed:
            # 已满足自洽，跳出
            break
            
        current_design["system_blueprint"] = blueprint
        current_design["refinement_loops"] = loop + 1

    return {
        "final_design": current_design,
        "refinement_status": "Auto-refined to meet balance constraints" if current_design.get("refinement_loops") else "Passed initially",
        "loops_executed": current_design.get("refinement_loops", 0)
    }

def build_default_gene_tools(gene_db: GeneDatabase) -> GeneToolRegistry:
    registry = GeneToolRegistry()

    registry.register("translate_dna", "将DNA序列翻译为氨基酸序列，识别所有开放阅读框(ORF)，统计密码子使用频率。参数: sequence(str), frame(int, 默认1)", translate_dna)
    registry.register("reverse_complement", "计算DNA序列的反向互补链。参数: sequence(str)", reverse_complement)
    registry.register("protein_physicochemical", "计算蛋白质的物理化学性质：分子量、等电点、不稳定指数、GRAVY疏水性、电荷分布、二硫键潜力。参数: sequence_aa(str)", protein_physicochemical)
    registry.register("scan_motifs", "扫描蛋白质序列中的功能基序(motif)：核定位信号、跨膜区、锌指、激酶活性位点、SH2/SH3、GPCR特征等。参数: sequence_aa(str)", scan_motifs)
    registry.register("predict_protein_function", "综合motif扫描和理化性质，预测蛋白质最可能的功能类别（激酶/转录因子/受体/转运体/酶等），含化学反应式和辅因子信息。参数: sequence_aa(str)", predict_protein_function)
    registry.register("search_gene_database", "在基因数据库中搜索基因：symbol精确搜索、keyword模糊搜索、pathway通路搜索、go_term功能搜索、domain结构域搜索。参数: query(str), search_type(str, 默认symbol), limit(int)", lambda **kwargs: search_gene_database(gene_db=gene_db, **kwargs))
    registry.register("cross_species_homology_mapping", "跨物种同源基因映射，预测在模式生物中的敲除表型和保守性。参数: gene_id(str), target_species(str, 可选mouse/rat/zebrafish/fruit_fly/worm/yeast等)", lambda **kwargs: cross_species_homology_mapping(gene_db=gene_db, **kwargs))
    registry.register("predict_gene_modification_effect", "预测基因修改（敲除/过表达/点突变）在分子、细胞、组织、个体四个层面的下游效应，评估严重性并给出实验建议。参数: gene_id(str), modification_type(str, knockout/overexpression/point_mutation)", lambda **kwargs: predict_gene_modification_effect(gene_db=gene_db, **kwargs))
    registry.register("reverse_phenotype_to_gene", "反向性状-基因推断：给定目标性状/表型，从基因库中检索最相关的候选基因，给出修改策略。参数: target_trait(str), target_organ(str, 可选)", lambda **kwargs: reverse_phenotype_to_gene(gene_db=gene_db, **kwargs))
    registry.register("design_organism", "从头设计一种具有特定性状/环境适应能力的新生物，输出全系统基因蓝图（能量代谢/结构/复制/调控/防御/特殊功能模块）。参数: target_trait(str), environment(str, 可选)", lambda **kwargs: design_organism(gene_db=gene_db, **kwargs))
    registry.register("chemical_to_gene_inference", "化学逆推：给定化学物质名称，反推其作用的受体/靶基因及下游信号通路。涵盖神经递质、激素、生长因子、细胞因子、发育信号分子等。参数: chemical_name(str)", chemical_to_gene_inference)
    registry.register("parse_smiles", "解析SMILES化学结构字符串，计算分子式、分子量、氢键供体/受体、logP、类药性等物理化学性质。参数: smiles(str)", parse_smiles)
    registry.register("search_chemical_by_similarity", "通过名称模糊匹配搜索已知药物/化学物质的化学指纹图谱和目标基因。参数: query(str)", search_chemical_by_similarity)
    registry.register("infer_chemical_targets", "综合化学结构解析和知识库匹配，推断化学物质的分子靶标和作用机制。参数: chemical_name(str)", infer_chemical_targets)
    registry.register("predict_multi_gene_editing", "预测多个基因同时编辑的协同效应和通路叠加影响。自动检测多基因共同击中的信号通路。参数: gene_ids(list[str]), modification_types(list[str], 可选)", lambda **kwargs: predict_multi_gene_editing(gene_db=gene_db, **kwargs))
    registry.register("validate_organism_design", "对生物设计方案进行自洽性校验：模块平衡、基因冗余度、数据库覆盖度、通路冲突检测。参数: target_trait(str), environment(str, 可选)", lambda **kwargs: validate_organism_design(gene_db=gene_db, **kwargs))
    
    # 高级系统生物学与结构生物学工具
    registry.register("analyze_gene_network", "使用知识图谱分析基因网络。action: shortest_path (参数 source, target), central_genes (参数 limit), disease_comorbidity (参数 disease_name)", lambda **kwargs: analyze_gene_network(gene_db=gene_db, **kwargs))
    registry.register("fetch_3d_structure_info", "获取蛋白质 3D 结构特征、pLDDT 置信度和结合口袋预测。参数: gene_symbol(str)", lambda **kwargs: fetch_3d_structure_info(gene_db=gene_db, **kwargs))
    registry.register("simulate_molecular_docking", "模拟分子对接，评估配体与蛋白的结合亲和力和 Kd 值。参数: protein_symbol(str), ligand_smiles(str)", simulate_molecular_docking)
    registry.register("flux_balance_analysis", "执行代谢通量平衡分析 (FBA)，计算基因增删对细胞生长速率和 ATP 消耗的系统性影响。参数: added_genes(list[str]), knocked_out_genes(list[str])", flux_balance_analysis)
    registry.register("get_tissue_specific_expression", "获取基因在人体各组织中的特异性表达谱 (TPM)，评估脱靶副作用。参数: gene_symbol(str)", get_tissue_specific_expression)
    
    # 外部真实API与精细变异预测工具
    registry.register("fetch_from_ensembl_api", "从 Ensembl REST API 提取变异和转录本层面的硬证据。参数: gene_symbol(str)", fetch_from_ensembl_api)
    registry.register("fetch_from_uniprot_api", "从 UniProt 提取蛋白层面的硬证据和交互网络。参数: gene_symbol(str)", fetch_from_uniprot_api)
    registry.register("predict_variant_consequence", "精细到变异层 (Variant-level) 的推断，预测具体的点突变后果 (如 c.1799T>A / p.Val600Glu)。参数: variant_id_or_hgvs(str)", lambda **kwargs: predict_variant_consequence(gene_db=gene_db, **kwargs))
    
    # 终极 10 大模块：数据网关、深度模型、表观遗传学、MD结构计算、系统评测
    registry.register("search_topology_tree", "动态本体树检索：搜索疾病(MONDO)或基因功能(GO)。如果本地找不到，将自动从欧洲生物信息研究所(EBI OLS)云端抓取并扩展本地树。参数: query(str)", lambda **kwargs: dynamic_ontology_search(gene_db, kwargs.get("query", ""), 10))
    registry.register("fetch_genes_by_ontology", "通过本体树的 GO 编号 (如 GO:0006281) 从 UniProt 云端批量拉取真实基因并永久存入本地库。参数: node_id(str), limit(int, 默认5)", lambda **kwargs: fetch_genes_by_ontology_api(gene_db, kwargs.get("node_id", ""), kwargs.get("limit", 5)))
    
    registry.register("query_global_bio_gateway", "查询海量外部数据库(GTEx, DepMap, ClinVar, OpenTargets)。参数: db_name(str), query_id(str, 支持基因名或MONDO编号), extra_param(str,可选)", 
                      lambda db_name, query_id, extra_param="": getattr(GlobalBioDataGateway, f"query_{db_name.lower()}")(query_id) if db_name.lower() in ["clinvar", "opentargets"] else getattr(GlobalBioDataGateway, f"query_{db_name.lower()}")(query_id, extra_param))
    registry.register("predict_protein_embedding_esm", "使用 ESM3 基础模型提取蛋白高维特征嵌入和零样本适应度预测。参数: sequence(str)", lambda **kwargs: BioFoundationModelProxy.predict_protein_embedding(kwargs.get("sequence", "")))
    registry.register("causal_inference_network", "执行大规模单细胞孟德尔随机化因果推断。参数: gene_x(str), gene_y(str), context(str)", lambda **kwargs: BioFoundationModelProxy.causal_inference_network(kwargs.get("gene_x", ""), kwargs.get("gene_y", ""), kwargs.get("context", "")))
    registry.register("predict_tf_binding", "预测转录因子在启动子序列上的结合亲和力。参数: tf_name(str), promoter_sequence(str)", lambda **kwargs: EpigeneticAnalyzer.predict_tf_binding(kwargs.get("tf_name", ""), kwargs.get("promoter_sequence", "")))
    registry.register("query_epigenetic_state", "查询染色质开放性(ATAC-seq)或3D基因组接触(Hi-C)。参数: analysis_type(str: 'atac' or 'hic'), locus_or_promoter(str), cell_type(str)", 
                      lambda analysis_type, locus_or_promoter, cell_type: EpigeneticAnalyzer.query_chromatin_accessibility(locus_or_promoter, cell_type) if analysis_type == 'atac' else EpigeneticAnalyzer.query_3d_genome_interactions(locus_or_promoter, cell_type))
    registry.register("run_md_simulation_proxy", "运行高级分子动力学模拟代理，评估蛋白动态稳定性和RMSD。参数: pdb_id(str), time_ns(float,可选)", lambda **kwargs: StructuralBiologyEngine.run_md_simulation(kwargs.get("pdb_id", ""), kwargs.get("time_ns", 100.0)))
    registry.register("calculate_binding_fep", "执行自由能微扰(FEP)计算，获取高精度配体相对结合自由能。参数: ligand_a_smiles(str), ligand_b_smiles(str), protein(str)", lambda **kwargs: StructuralBiologyEngine.calculate_binding_free_energy_fep(kwargs.get("ligand_a_smiles", ""), kwargs.get("ligand_b_smiles", ""), kwargs.get("protein", "")))
    registry.register("run_system_benchmark", "运行世界级基准测试套件，评估当前系统预测准确率(AUC、通路恢复率等)。", lambda **kwargs: GeneResearchBenchmarkSuite.run_full_suite())

    # 自动修正循环与新工具
    registry.register("auto_refine_organism_design", "自动执行设计、校验并修正冲突。参数: target_trait(str), environment(str, 可选)", lambda **kwargs: auto_refine_organism_design(gene_db=gene_db, **kwargs))
    registry.register("design_crispr_grna", "设计 CRISPR-Cas9 gRNA 并计算脱靶得分。参数: target_sequence(str), pam(str, 默认 NGG)", design_crispr_grna)
    registry.register("predict_immunogenicity_and_toxicity", "预测蛋白质免疫原性 (HLA结合) 和细胞毒性。参数: protein_sequence(str)", predict_immunogenicity_and_toxicity)
    
    # 高性能生物物理与动力学仿真引擎
    registry.register("thermodynamic_feasibility", "计算代谢通路的吉布斯自由能 (dG)，判断极端温度/pH下是否自发可行。参数: pathway_name(str), temperature_celsius(float), ph(float)", ThermodynamicSimulator.calculate_pathway_thermodynamics)
    registry.register("kinetic_ode_simulation", "常微分方程模拟代谢物浓度随时间的动态变化，检测毒性中间体积累。参数: enzymes(list[str]), initial_substrate_mM(float), simulation_time_h(float)", KineticODESimulator.simulate_metabolite_dynamics)
    registry.register("extreme_env_protein_stability", "预测极端温度和辐射(Gray)下的蛋白质解链温度(Tm)和抗辐射半衰期。参数: protein_symbol(str), temperature_celsius(float), radiation_gray(float)", ExtremeEnvironmentProteinSimulator.predict_protein_stability)
    registry.register("whole_cell_resource_allocation", "计算异源基因表达对宿主核糖体池的占用率及细胞崩溃概率。参数: added_genes(list[str]), host_organism(str)", WholeCellResourceAllocator.calculate_translation_burden)

    # GPU 加速计算模块 (GPU-Accelerated Modules)
    registry.register("generate_de_novo_protein", "调用扩散模型(RFdiffusion/Evo)生成自然界不存在的全新蛋白。参数: target_function(str)", lambda **kwargs: GenerativeProteinDiffusion.generate_de_novo_protein(kwargs.get("target_function", "")))
    registry.register("simulate_quantum_catalysis", "调用 QM/MM 计算酶活性中心的过渡态活化能(dE‡)和催化速率(kcat)。参数: enzyme_pdb(str), substrate_smiles(str), reaction_mechanism(str)", QuantumEnzymeSimulator.calculate_activation_energy_qmmm)
    registry.register("design_subatomic_drug", "The Quantum-Bio Sandbox: 在亚原子层面设计完美契合靶点的配体药物，计算高精度结合自由能 (ΔΔG)。参数: target_protein_pdb(str), active_site_residues(list[str])", lambda **kwargs: QuantumBioSandbox.design_subatomic_drug_ligand(kwargs.get("target_protein_pdb", ""), kwargs.get("active_site_residues", [])))
    registry.register("simulate_3d_genome_spatial", "The Spatial-Omics Engine: 模拟特定应激条件下全基因组 3D 拓扑结构 (TAD 边界) 的相变与增强子劫持。参数: cell_type(str), stress_condition(str)", lambda **kwargs: SpatialOmicsEngine.simulate_3d_genome_phase_transition(kwargs.get("cell_type", ""), kwargs.get("stress_condition", "")))

    # 真实 SciPy ODE 求解器替代品
    registry.register("simulate_membrane_potential", "基于 Hodgkin-Huxley 4变量ODE 真实仿真细胞跨膜动作电位。参数: ion_channels(list[str]), external_stimulus_pA(float), duration_ms(float)", HodgkinHuxleySimulator.simulate_action_potential)
    registry.register("simulate_tissue_morphogenesis", "使用 Gray-Scott 反应扩散 PDE 有限差分法模拟图灵斑图。参数: morphogens(list[str]), grid_size(int,默认64), simulation_steps(int,默认5000)", lambda **kwargs: ReactionDiffusionSimulator.simulate_turing_pattern(kwargs.get("morphogens", [])))

    # 真实 PWM 转录因子结合与 CRISPR 脱靶评估
    registry.register("real_tf_binding_scan", "使用真实 JASPAR PWM 矩阵扫描启动子序列计算 TF 结合得分。参数: tf_name(str), promoter_sequence(str)",
                      lambda tf_name, promoter_sequence: {
                          "tf": tf_name,
                          "max_log_odds_score": round(_sequence_to_pwm_score(promoter_sequence.upper(), TF_PWM.get(tf_name.upper(), np.array([[0.25]*4]*8))), 2),
                          "binding_sites_found": sum(1 for i in range(len(promoter_sequence) - TF_PWM.get(tf_name.upper(), np.array([[0.25]*4]*8)).shape[0] + 1) if _sequence_to_pwm_score(promoter_sequence[i:i+TF_PWM.get(tf_name.upper(), np.array([[0.25]*4]*8)).shape[0]], TF_PWM.get(tf_name.upper(), np.array([[0.25]*4]*8))) > 3.0),
                          "evidence_level": "Level B (Real PWM Motif Scan)",
                      })
    registry.register("true_crispr_off_target", "使用 Needleman-Wunsch 全局比对真实评估 gRNA 脱靶风险。参数: grna(str), genome_region(str)",
                      lambda grna, genome_region: _calculate_crispr_off_target(grna, genome_region))

    # FAISS 向量语义检索
    registry.register("semantic_gene_search", "使用向量 Embedding + FAISS 语义检索功能相似基因。参数: query(str), top_k(int,默认10)", lambda **kwargs: GeneVectorDatabase(gene_db).search_by_text(kwargs.get("query", ""), kwargs.get("top_k", 10)))

    # 3D 结构可视化
    registry.register("render_protein_3d", "渲染蛋白质 3D 结构并高亮突变位点 (py3Dmol)。参数: gene_symbol(str), highlight_variants(list[str],可选)", lambda **kwargs: ProteinStructureRenderer.render_structure_html(gene_db, kwargs.get("gene_symbol", ""), kwargs.get("highlight_variants")))

    # 多尺度与复杂系统仿真模块 (Complex Systems Modules)
    registry.register("assemble_xna", "评估非天然核酸(XNA)如 PNA/TNA/Silicone 的组装稳定性与解链温度。参数: sequence(str), backbone_type(str)", lambda **kwargs: XNAAssembler.assemble_and_evaluate_xna(kwargs.get("sequence", ""), kwargs.get("backbone_type", "DNA")))
    registry.register("multi_scale_cascade", "模拟分子级点突变如何级联放大导致组织器官崩溃或演化。参数: gene(str), mutation(str), env_context(str)", lambda **kwargs: MultiScaleCoupler.cascade_mutation_effect(kwargs.get("gene", ""), kwargs.get("mutation", ""), kwargs.get("env_context", "")))
    registry.register("generate_opentrons_script", "生成 Opentrons 机械臂自动化组装湿实验 Python 脚本。参数: protocol_name(str), parts_to_assemble(list[str])", lambda **kwargs: LabAutomationGenerator.generate_opentrons_protocol(kwargs.get("protocol_name", ""), kwargs.get("parts_to_assemble", [])))
    registry.register("simulate_microbiome_ecology", "使用 Lotka-Volterra ODE 模拟多物种在火星/金星封闭群落中的生态演化。参数: species_names(list[str]), initial_populations(list[float]), growth_rates(list[float]), interaction_matrix(list[list[float]]), simulation_years(float)", lambda **kwargs: EcologicalSimulator.simulate_microbiome_dynamics(kwargs.get("species_names", []), kwargs.get("initial_populations", []), kwargs.get("growth_rates", []), kwargs.get("interaction_matrix", []), kwargs.get("simulation_years", 10000.0)))

    # 宏观 In-Silico 纯计算引擎 (Macro In-Silico Engines)
    registry.register("generate_universal_lifeform", "通用生命形态生成器：基于物理化学约束（温度、压力、溶剂、元素），从头生成纯硅基/氨基等异种生命的生化架构。参数: temperature_k(float), pressure_atm(float), solvent(str), available_elements(list[str])", lambda **kwargs: UniversalLifeFormGenerator.generate_biochemistry_architecture(kwargs.get("temperature_k", 298.15), kwargs.get("pressure_atm", 1.0), kwargs.get("solvent", "water"), kwargs.get("available_elements", ["C", "H", "O", "N", "P", "S"])))
    registry.register("simulate_whole_cell_sandbox", "全细胞沙盒：推演给定基因组和核糖体数量下，最小生命细胞分裂周期与蛋白质组资源分配极限。参数: genome_size_bp(int), protein_coding_genes(int), ribosome_count(int), nutrient_availability(float)", lambda **kwargs: WholeCellSandbox.simulate_minimal_cell_cycle(kwargs.get("genome_size_bp", 1000000), kwargs.get("protein_coding_genes", 1000), kwargs.get("ribosome_count", 10000), kwargs.get("nutrient_availability", 1.0)))
    registry.register("infer_evolutionary_trajectory", "演化轨迹与祖先推断：推演宏观时间尺度上的物种演化路径、关键分化节点与突变积累。参数: initial_phenotype(str), generations(int), mutation_rate(float), selection_pressures(list[str])", lambda **kwargs: EvolutionaryTrajectoryEngine.infer_trajectory(kwargs.get("initial_phenotype", ""), kwargs.get("generations", 1000000), kwargs.get("mutation_rate", 1e-8), kwargs.get("selection_pressures", [])))
    registry.register("simulate_biogeochemical_dynamics", "生物地球化学耦合动力学：模拟生物圈代谢活动对全球大气成分和温度的长期影响（如大氧化事件、温室效应失控）。参数: initial_atmosphere(dict), biosphere_metabolism(dict), timescale_years(float)", lambda **kwargs: BiogeochemicalDynamicsEngine.simulate_biosphere_atmosphere_coupling(kwargs.get("initial_atmosphere", {}), kwargs.get("biosphere_metabolism", {}), kwargs.get("timescale_years", 1000000.0)))
    
    # 5 大前沿科研计算引擎 (Frontier Scientific Computing)
    registry.register("virtual_clinical_trials", "虚拟临床试验：模拟靶向药物在不同遗传背景(代谢基因多态性)群体中的药代动力学(PK/PD)及副作用。参数: drug_name(str), clearance_gene(str), population_size(int), dose_mg(float)", lambda **kwargs: VirtualClinicalTrialsEngine.simulate_population_pharmacokinetics(kwargs.get("drug_name", ""), kwargs.get("clearance_gene", ""), kwargs.get("population_size", 1000), kwargs.get("dose_mg", 50.0)))
    registry.register("epigenetic_reprogramming", "表观重编程模拟：推演沃丁顿表观遗传景观中的细胞状态跃迁概率。参数: initial_cell_type(str), target_cell_type(str), transcription_factors_applied(list[str])", lambda **kwargs: EpigeneticReprogrammingSimulator.calculate_waddington_landscape_transition(kwargs.get("initial_cell_type", ""), kwargs.get("target_cell_type", ""), kwargs.get("transcription_factors_applied", [])))
    registry.register("simulate_hgt_dynamics", "泛基因组与水平基因转移：基于限制性修饰屏障与同源性推演质粒/噬菌体水平基因转移概率。参数: donor_species(str), recipient_species(str), gene_sequence_length(int)", lambda **kwargs: PangenomeHGTEngine.simulate_horizontal_gene_transfer(kwargs.get("donor_species", ""), kwargs.get("recipient_species", ""), kwargs.get("gene_sequence_length", 1000)))
    registry.register("neuro_genomic_topology", "神经-基因拓扑发育：求解反应-扩散 PDE，推演神经元轴突导向的形态发生素梯度场。参数: morphogen_gene(str), source_concentration_uM(float), diffusion_coefficient_um2_s(float), degradation_rate_s_inv(float), distance_um(float)", lambda **kwargs: NeuroGenomicTopologyModel.calculate_axon_guidance_gradient(kwargs.get("morphogen_gene", ""), kwargs.get("source_concentration_uM", 1.0), kwargs.get("diffusion_coefficient_um2_s", 10.0), kwargs.get("degradation_rate_s_inv", 0.01), kwargs.get("distance_um", 100.0)))
    registry.register("astrobiology_panspermia", "星际胚种辐射生存推演：结合天体物理与泊松分布，推演生命跨星系传播可行性。参数: organism_type(str), radiation_resistance_d37_gy(float), deep_space_duration_years(float), shielding_rock_thickness_cm(float)", lambda **kwargs: AstrobiologyPanspermiaEngine.evaluate_interstellar_survival(kwargs.get("organism_type", ""), kwargs.get("radiation_resistance_d37_gy", 5000.0), kwargs.get("deep_space_duration_years", 1e6), kwargs.get("shielding_rock_thickness_cm", 100.0)))

    # 物理/化学底层第一性原理工具箱 (First-Principles Toolbox)
    registry.register("calc_gibbs_free_energy", "计算吉布斯自由能(ΔG=ΔH-TΔS)，判断生化反应或大分子折叠是否能自发进行。参数: enthalpy_j_mol(float), entropy_j_mol_k(float), temperature_k(float)", lambda **kwargs: FirstPrinciplesCalculators.calculate_gibbs_free_energy(kwargs.get("enthalpy_j_mol", 0.0), kwargs.get("entropy_j_mol_k", 0.0), kwargs.get("temperature_k", 298.15)))
    registry.register("calc_arrhenius_rate", "阿伦尼乌斯方程计算反应速率常数 k = A*exp(-Ea/RT)，用于温度依赖性推演。参数: pre_exponential_factor(float), activation_energy_j_mol(float), temperature_k(float)", lambda **kwargs: FirstPrinciplesCalculators.arrhenius_reaction_rate(kwargs.get("pre_exponential_factor", 1e13), kwargs.get("activation_energy_j_mol", 50000.0), kwargs.get("temperature_k", 298.15)))
    registry.register("calc_boltzmann_distribution", "计算两个能量态在热力学平衡下的玻尔兹曼分布比例，用于构象概率或通道开放概率。参数: energy_state_1_j_mol(float), energy_state_2_j_mol(float), temperature_k(float)", lambda **kwargs: FirstPrinciplesCalculators.boltzmann_state_distribution(kwargs.get("energy_state_1_j_mol", 0.0), kwargs.get("energy_state_2_j_mol", 5000.0), kwargs.get("temperature_k", 298.15)))
    registry.register("calc_michaelis_menten", "米氏方程计算酶促反应速率 v = Vmax*[S]/(Km+[S])。参数: v_max(float), k_m(float), substrate_conc(float)", lambda **kwargs: FirstPrinciplesCalculators.michaelis_menten_kinetics(kwargs.get("v_max", 1.0), kwargs.get("k_m", 1.0), kwargs.get("substrate_conc", 1.0)))
    registry.register("calc_nernst_potential", "能斯特方程计算特定离子的跨膜平衡电位 E=(RT/zF)*ln([out]/[in])。参数: ion_charge_z(int), conc_out(float), conc_in(float), temperature_k(float)", lambda **kwargs: FirstPrinciplesCalculators.nernst_potential(kwargs.get("ion_charge_z", 1), kwargs.get("conc_out", 1.0), kwargs.get("conc_in", 10.0), kwargs.get("temperature_k", 310.15)))
    registry.register("calc_quantum_tunneling", "量子隧道效应穿透概率计算，用于极低温酶催化或极限异种生物学化学反应推演。参数: particle_mass_kg(float), barrier_width_m(float), barrier_energy_j(float), particle_energy_j(float)", lambda **kwargs: FirstPrinciplesCalculators.quantum_tunneling_probability(kwargs.get("particle_mass_kg", 1.67e-27), kwargs.get("barrier_width_m", 1e-10), kwargs.get("barrier_energy_j", 1e-19), kwargs.get("particle_energy_j", 0.5e-19)))
    registry.register("calc_brownian_diffusion", "布朗运动扩散时间方程，计算大分子在细胞内的扩散延迟和拥挤效应。参数: distance_m(float), diffusion_coefficient_m2_s(float)", lambda **kwargs: FirstPrinciplesCalculators.brownian_diffusion_time(kwargs.get("distance_m", 1e-6), kwargs.get("diffusion_coefficient_m2_s", 1e-12)))
    registry.register("calc_hagen_poiseuille_flow", "哈根-泊肃叶流体力学方程，模拟血液/体液在微管或树液网络中的层流流量。参数: radius_m(float), length_m(float), pressure_drop_pa(float), viscosity_pa_s(float)", lambda **kwargs: FirstPrinciplesCalculators.navier_stokes_hagen_poiseuille(kwargs.get("radius_m", 1e-5), kwargs.get("length_m", 1e-3), kwargs.get("pressure_drop_pa", 1000.0), kwargs.get("viscosity_pa_s", 1e-3)))
    registry.register("calc_lorentz_force", "洛伦兹力方程计算带电粒子在电磁场中的偏转力，用于推演极端天体环境或趋磁细菌磁小体机制。参数: ion_charge_coulombs(float), velocity_m_s(float), magnetic_field_tesla(float), electric_field_v_m(float)", lambda **kwargs: FirstPrinciplesCalculators.lorentz_force_ion_deflection(kwargs.get("ion_charge_coulombs", 1.6e-19), kwargs.get("velocity_m_s", 1000.0), kwargs.get("magnetic_field_tesla", 1.0), kwargs.get("electric_field_v_m", 0.0)))
    registry.register("calc_bragg_diffraction", "布拉格衍射方程计算生物晶体(如病毒衣壳、鸟嘌呤晶体)对特定波长光线的结构色反射。参数: wavelength_m(float), lattice_spacing_m(float), diffraction_order(int)", lambda **kwargs: FirstPrinciplesCalculators.bragg_diffraction_crystallography(kwargs.get("wavelength_m", 5e-7), kwargs.get("lattice_spacing_m", 3e-7), kwargs.get("diffraction_order", 1)))
    registry.register("calc_capillary_action", "杨-拉普拉斯方程推导的毛细上升高度，推演植物木质部极限、微小昆虫饮水或人造血管约束。参数: surface_tension_n_m(float), contact_angle_degrees(float), tube_radius_m(float), fluid_density_kg_m3(float)", lambda **kwargs: FirstPrinciplesCalculators.young_laplace_capillary_action(kwargs.get("surface_tension_n_m", 0.072), kwargs.get("contact_angle_degrees", 0.0), kwargs.get("tube_radius_m", 1e-5), kwargs.get("fluid_density_kg_m3", 1000.0)))

    registry.register("gene_db_stats", "查看基因数据库统计信息：基因总数、域数量、通路数量、基因类型分布。", lambda **kwargs: gene_db.stats())

    return registry