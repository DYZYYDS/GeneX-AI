from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any

ATOMIC_MASSES: dict[str, float] = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "P": 30.974, "S": 32.065, "Cl": 35.453, "Br": 79.904, "I": 126.904,
    "Na": 22.990, "K": 39.098, "Mg": 24.305, "Ca": 40.078, "Fe": 55.845,
    "Zn": 65.380, "Mn": 54.938, "Cu": 63.546, "Se": 78.971, "B": 10.811,
    "Si": 28.085, "Ge": 72.630, "W": 183.84, # 增加硅、锗、钨等高温耐受元素
}

CHEMICAL_FINGERPRINTS: dict[str, dict[str, Any]] = {
    "caffeine": {
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "mw": 194.19,
        "logp": -0.07,
        "hbd": 0,
        "hba": 3,
        "tpsa": 58.4,
        "rotatable_bonds": 0,
        "rings": 2,
        "fragments": ["purine", "methylxanthine", "amide"],
        "target_genes": ["ADORA1", "ADORA2A", "PDE4D"],
        "mechanism": "腺苷受体拮抗剂 + 磷酸二酯酶抑制剂",
    },
    "aspirin": {
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "mw": 180.16,
        "logp": 1.19,
        "hbd": 1,
        "hba": 4,
        "tpsa": 63.6,
        "rotatable_bonds": 3,
        "rings": 1,
        "fragments": ["acetyl", "salicylate", "carboxylic_acid"],
        "target_genes": ["PTGS1", "PTGS2", "NFKB1"],
        "mechanism": "不可逆 COX-1/COX-2 乙酰化抑制剂",
    },
    "ibuprofen": {
        "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "mw": 206.28,
        "logp": 3.97,
        "hbd": 1,
        "hba": 2,
        "tpsa": 37.3,
        "rotatable_bonds": 4,
        "rings": 1,
        "fragments": ["propionic_acid", "isobutylbenzene"],
        "target_genes": ["PTGS1", "PTGS2"],
        "mechanism": "可逆非选择性 COX 抑制剂",
    },
    "metformin": {
        "smiles": "CN(C)C(=N)NC(=N)N",
        "mw": 129.16,
        "logp": -2.6,
        "hbd": 4,
        "hba": 3,
        "tpsa": 88.9,
        "rotatable_bonds": 2,
        "rings": 0,
        "fragments": ["biguanide"],
        "target_genes": ["PRKAA1", "PRKAA2", "MTOR", "SLC22A1"],
        "mechanism": "AMPK激活剂 → 抑制肝脏糖异生",
    },
    "warfarin": {
        "smiles": "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
        "mw": 308.33,
        "logp": 2.70,
        "hbd": 1,
        "hba": 4,
        "tpsa": 63.6,
        "rotatable_bonds": 4,
        "rings": 3,
        "fragments": ["coumarin", "phenyl", "ketone"],
        "target_genes": ["VKORC1", "CYP2C9"],
        "mechanism": "维生素K环氧化物还原酶抑制剂 → 抗凝",
    },
    "rapamycin": {
        "smiles": "CO[C@H]1C[C@@H]2CC[C@@H](C)[C@@](O)(O2)C(=O)C(=O)N3CCCC[C@H]3C(=O)O[C@H]([C@H](C)C[C@@H]4CC[C@@H](O)[C@H](OC)C4)CC(=O)[C@H](C)/C=C(\\C)[C@@H](O)[C@@H](OC)C(=O)[C@H](C)C[C@H](C)/C=C/C=C/C=C/1",
        "mw": 914.17,
        "logp": 4.3,
        "hbd": 3,
        "hba": 13,
        "tpsa": 178.4,
        "rotatable_bonds": 6,
        "rings": 4,
        "fragments": ["macrolide", "pipecolate", "triene"],
        "target_genes": ["MTOR", "FKBP1A"],
        "mechanism": "与FKBP12形成复合物抑制mTORC1",
    },
    "cisplatin": {
        "smiles": "N[Pt](N)(Cl)Cl",
        "mw": 300.05,
        "logp": -2.19,
        "hbd": 2,
        "hba": 2,
        "tpsa": 52.0,
        "rotatable_bonds": 0,
        "rings": 0,
        "fragments": ["platinum_amine"],
        "target_genes": ["TP53", "ATM", "ERCC1"],
        "mechanism": "DNA交联 → 复制停滞 → 凋亡",
    },
    "doxorubicin": {
        "smiles": "COC1=CC=CC2=C1C(=O)C1=C(C2=O)C(O)=C2C[C@@](O)(C(=O)CO)C[C@H](O[C@H]3C[C@H](N)[C@H](O)[C@H](C)O3)C2=C1O",
        "mw": 543.52,
        "logp": 1.27,
        "hbd": 6,
        "hba": 12,
        "tpsa": 206.1,
        "rotatable_bonds": 5,
        "rings": 5,
        "fragments": ["anthracycline", "amino_sugar"],
        "target_genes": ["TOP2A", "TP53"],
        "mechanism": "拓扑异构酶II抑制剂 + DNA嵌入 + ROS产生",
    },
    "nicotine": {
        "smiles": "CN1CCC[C@H]1C1=CC=CN=C1",
        "mw": 162.23,
        "logp": 1.17,
        "hbd": 0,
        "hba": 2,
        "tpsa": 16.1,
        "rotatable_bonds": 1,
        "rings": 2,
        "fragments": ["pyridine", "pyrrolidine"],
        "target_genes": ["CHRNA7", "CHRNA4", "CHRNB2"],
        "mechanism": "烟碱型乙酰胆碱受体激动剂",
    },
    "thalidomide": {
        "smiles": "O=C1CCC(N2C(=O)C3=CC=CC=C3C2=O)C(=O)N1",
        "mw": 258.23,
        "logp": 0.33,
        "hbd": 1,
        "hba": 5,
        "tpsa": 83.6,
        "rotatable_bonds": 1,
        "rings": 3,
        "fragments": ["phthalimide", "glutarimide"],
        "target_genes": ["CRBN", "IKZF1", "IKZF3"],
        "mechanism": "CRBN E3泛素连接酶调节剂 → 靶向降解IKZF1/3",
    },
    "penicillin_g": {
        "smiles": "CC1(C)S[C@@H]2[C@H](NC(=O)CC3=CC=CC=C3)C(=O)N2[C@H]1C(=O)O",
        "mw": 334.39,
        "logp": 1.83,
        "hbd": 2,
        "hba": 5,
        "tpsa": 112.0,
        "rotatable_bonds": 4,
        "rings": 3,
        "fragments": ["beta_lactam", "thiazolidine", "phenylacetyl"],
        "target_genes": ["PBP1A", "PBP2", "PBP3"],
        "mechanism": "转肽酶抑制剂 → 细菌细胞壁合成阻断",
    },
}


def parse_smiles(smiles: str) -> dict[str, Any]:
    atoms: list[str] = []
    bonds: list[dict[str, Any]] = []

    element_pattern = re.compile(
        r"Cl|Br|Na|Mg|Ca|Fe|Zn|Mn|Cu|Se|Pt|Si|Ge|W|"
        r"B|C|N|O|F|P|S|K|I|H(?!e)"
    )
    pos = 0
    atom_idx = 0
    atom_stack: list[int] = []
    ring_marks: dict[int, list[tuple[int, str]]] = {}
    branch_stack: list[int] = []

    while pos < len(smiles):
        ch = smiles[pos]

        if ch == "[":
            end = smiles.index("]", pos)
            bracket_content = smiles[pos + 1:end]
            elem_match = element_pattern.match(bracket_content)
            if elem_match:
                atoms.append(elem_match.group())
            else:
                atoms.append(bracket_content[0] if bracket_content else "C")
            atom_idx = len(atoms) - 1
            pos = end + 1
        elif ch in "B" or (ch == "C" and pos + 1 < len(smiles) and smiles[pos + 1] == "l"):
            if ch == "C" and pos + 1 < len(smiles) and smiles[pos + 1] == "l":
                atoms.append("Cl")
                pos += 2
            else:
                atoms.append(ch)
                pos += 1
            atom_idx = len(atoms) - 1
        elif ch == "B" and pos + 1 < len(smiles) and smiles[pos + 1] == "r":
            atoms.append("Br")
            pos += 2
            atom_idx = len(atoms) - 1
        elif ch == "S" and pos + 1 < len(smiles) and smiles[pos + 1] == "i":
            atoms.append("Si")
            pos += 2
            atom_idx = len(atoms) - 1
        elif ch == "G" and pos + 1 < len(smiles) and smiles[pos + 1] == "e":
            atoms.append("Ge")
            pos += 2
            atom_idx = len(atoms) - 1
        elif ch in "CNOFPSIH":
            atoms.append(ch)
            pos += 1
            atom_idx = len(atoms) - 1
            if ch == "N" and pos < len(smiles) and smiles[pos] == "a":
                atoms[-1] = "Na"
                pos += 1
            elif ch == "M" and pos < len(smiles) and smiles[pos] == "g":
                atoms[-1] = "Mg"
                pos += 1
            elif ch == "K" and pos < len(smiles):
                pass
        elif ch == "(":
            branch_stack.append(atom_idx)
            pos += 1
        elif ch == ")":
            if branch_stack:
                branch_stack.pop()
            pos += 1
        elif ch.isdigit():
            ring_num = int(ch)
            if ring_num in ring_marks:
                other = ring_marks[ring_num]
                if len(other) == 1:
                    bonds.append({"from": other[0][0], "to": atom_idx, "type": "single", "ring": ring_num})
                    ring_marks[ring_num].append((atom_idx, "close"))
                else:
                    pass
            else:
                ring_marks[ring_num] = [(atom_idx, "open")]
            pos += 1
        elif ch in "=#$/\\@":
            pos += 1
        elif ch == ".":
            pos += 1
        else:
            pos += 1

    mw = _calculate_mw(atoms)
    formula = _get_formula(atoms)
    logp_est = 0.5 * len(atoms) - sum(1 for a in atoms if a in "NO") * 0.7
    hbd = sum(1 for a in atoms if a in "NO") + atoms.count("N") + atoms.count("O")
    hba = sum(1 for a in atoms if a in "NOF")
    rotatable = max(0, len(atoms) - sum(1 for a in atoms if a in "CFNOS") // 2)

    return {
        "smiles": smiles,
        "atoms": atoms,
        "atom_count": len(atoms),
        "molecular_formula": formula,
        "molecular_weight": round(mw, 2),
        "estimated_logp": round(logp_est, 2),
        "h_bond_donors": hbd,
        "h_bond_acceptors": hba,
        "rotatable_bonds": rotatable,
        "drug_likeness": _lipinski_assessment(mw, logp_est, hbd, hba),
    }


def search_chemical_by_similarity(query: str) -> dict[str, Any]:
    query_lower = query.lower().replace(" ", "_").replace("-", "_")

    direct = CHEMICAL_FINGERPRINTS.get(query_lower)
    if direct:
        return {"found": True, "match_type": "exact", "chemical": direct}

    best = None
    best_score = 0.0
    for name, fp in CHEMICAL_FINGERPRINTS.items():
        score = _similarity_score(query_lower, name, fp)
        if score > best_score:
            best_score = score
            best = fp
            best["matched_name"] = name

    if best_score > 0.3:
        return {"found": True, "match_type": "fuzzy", "similarity_score": round(best_score, 3), "chemical": best}

    return {
        "found": False,
        "query": query,
        "suggestion": "未找到匹配的化学物质。请检查名称拼写，或提供 SMILES 字符串以进行结构解析。",
    }


def infer_chemical_targets(chemical_name: str) -> dict[str, Any]:
    result = search_chemical_by_similarity(chemical_name)
    if not result["found"]:
        return result

    chem = result["chemical"]
    targets = chem.get("target_genes", [])
    fragments = chem.get("fragments", [])

    return {
        "chemical": chemical_name,
        "matched_name": chem.get("matched_name", chemical_name),
        "smiles": chem["smiles"],
        "molecular_weight": chem["mw"],
        "target_genes": targets,
        "mechanism": chem["mechanism"],
        "structural_fragments": fragments,
        "physicochemical": {
            "logP": chem["logp"],
            "HBD": chem["hbd"],
            "HBA": chem["hba"],
            "TPSA": chem["tpsa"],
            "rotatable_bonds": chem["rotatable_bonds"],
        },
        "drug_likeness": _lipinski_assessment(chem["mw"], chem["logp"], chem["hbd"], chem["hba"]),
        "synthetic_accessibility": _estimate_synthetic_accessibility(chem["mw"], chem["rings"], chem["rotatable_bonds"]),
    }


def _calculate_mw(atoms: list[str]) -> float:
    total = 0.0
    for atom in atoms:
        total += ATOMIC_MASSES.get(atom, 12.0)
    return total


def _get_formula(atoms: list[str]) -> str:
    cnt = Counter(atoms)
    parts = []
    for elem in sorted(cnt.keys(), key=lambda e: (e != "C", e != "H", e)):
        parts.append(f"{elem}{cnt[elem]}" if cnt[elem] > 1 else elem)
    return "".join(parts)


def _lipinski_assessment(mw: float, logp: float, hbd: int, hba: int) -> dict[str, Any]:
    violations = 0
    details = []
    if mw > 500:
        violations += 1
        details.append(f"分子量 {mw:.1f} > 500")
    if logp > 5:
        violations += 1
        details.append(f"logP {logp:.1f} > 5")
    if hbd > 5:
        violations += 1
        details.append(f"氢键供体 {hbd} > 5")
    if hba > 10:
        violations += 1
        details.append(f"氢键受体 {hba} > 10")
    return {
        "violations": violations,
        "passes_lipinski": violations <= 1,
        "details": details if details else ["符合 Lipinski 类药性五规则"],
    }


def _estimate_synthetic_accessibility(mw: float, rings: int, rotatable: int) -> str:
    if mw < 300 and rings <= 2:
        return "容易（小分子、低环数）"
    if mw < 500 and rings <= 4:
        return "中等（常规化学合成可达）"
    return "困难（复杂天然产物或大环，可能需要生物合成途径）"


def _similarity_score(query: str, name: str, fp: dict[str, Any]) -> float:
    score = 0.0
    if query in name:
        score += 0.4
    for frag in fp.get("fragments", []):
        if frag in query or query in frag:
            score += 0.3
    mechanism = fp.get("mechanism", "").lower()
    for word in query.split("_"):
        if word in mechanism:
            score += 0.2
    return min(score, 1.0)
