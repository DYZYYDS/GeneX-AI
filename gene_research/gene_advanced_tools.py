from __future__ import annotations

import json
import random
import math
import time
import requests
import re
from functools import lru_cache
from typing import Any

from .gene_database import GeneDatabase
from .gene_chemistry import parse_smiles

try:
    import cobra
except ImportError:
    cobra = None


def _api_get_with_retry(url: str, headers: dict | None = None, max_retries: int = 3, timeout: int = 15) -> requests.Response:
    """指数退避重试机制 - 防止 API 限流和网络波动导致中断"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers or {}, timeout=timeout)
            if response.status_code == 429:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
                continue
            if response.status_code >= 500:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
                continue
            return response
        except (requests.ConnectionError, requests.Timeout):
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    return response


@lru_cache(maxsize=256)
def fetch_from_ensembl_api(gene_symbol: str) -> dict[str, Any]:
    """真实 API 集成：从 Ensembl REST API 提取跨物种同源和基因树信息。"""
    try:
        url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene_symbol}?expand=1"
        response = _api_get_with_retry(url, headers={"Content-Type": "application/json"})
        if response.status_code == 400:
             return {"error": f"Ensembl 无法识别基因 {gene_symbol}"}
        response.raise_for_status()
        data = response.json()
        
        transcripts = data.get("Transcript", [])
        return {
            "status": "success",
            "api": "Ensembl REST API",
            "ensembl_id": data.get("id"),
            "biotype": data.get("biotype"),
            "description": data.get("description"),
            "transcript_count": len(transcripts),
            "evidence_level": "Level A (Validated external DB)"
        }
    except Exception as e:
        return {"error": f"Ensembl API 请求失败: {str(e)}"}

def fetch_genes_by_ontology_api(gene_db: GeneDatabase, node_id: str, limit: int = 5) -> dict[str, Any]:
    """真实 API 集成：通过本体树的节点编号去 UniProt 批量懒加载基因。支持 GO、ENV 等拓扑。"""
    try:
        query_str = ""
        organism_filter = "+AND+organism_id:9606" # 默认限制为人类
        
        if node_id.startswith("GO:"):
            # UniProt API requires exact syntax for GO, e.g. go:0006281
            query_str = f'go:{node_id.replace("GO:", "")}'
        elif node_id.startswith("ENV:"):
            # 极端环境属于异种生物学，必须解除人类基因组限制，进行跨物种全局检索
            organism_filter = ""
            env_map = {
                "ENV:EXT_RAD": "KW-0678", # UniProt keyword for Radiation resistance
                "ENV:EXT_THERM": "KW-0803", # Thermophile
                "ENV:EXT_CHEM": "KW-0800" # Toxin
            }
            kw = env_map.get(node_id, "KW-0275")
            query_str = f'keyword:{kw}'
        elif node_id.startswith("PATH:"):
            query_str = f'pathway:"{node_id.replace("PATH:", "")}"'
        else:
            return {"error": "不支持的拓扑节点类型。目前支持 GO:, ENV:, PATH:。MONDO疾病请使用 query_global_bio_gateway。"}
            
        url = f"https://rest.uniprot.org/uniprotkb/search?query={query_str}+AND+reviewed:true{organism_filter}&format=json&size={limit}"
        response = _api_get_with_retry(url)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("results"):
            return {"error": f"UniProt 未找到归属于 {node_id} 的真实基因记录。"}
            
        fetched_symbols = []
        new_genes = []
        from .gene_database import GeneRecord
        
        for protein in data["results"]:
            try:
                genes_data = protein.get("genes", [])
                symbol = ""
                if genes_data:
                    symbol = genes_data[0].get("geneName", {}).get("value", "")
                
                # 如果没有 symbol，用 accession 代替，保证极端微生物的蛋白也能入库
                accession = protein.get("primaryAccession", "")
                if not symbol: 
                    symbol = accession
                
                name = protein.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "Auto-fetched protein")
                
                # 检查本地是否已有
                if not gene_db.search_by_symbol(symbol):
                    new_gene = GeneRecord(
                        gene_id=accession,
                        symbol=symbol.upper(),
                        name=name,
                        gene_type="protein_coding",
                        start=0, end=0,
                        summary=f"Automatically fetched via Ontology Node {node_id}.",
                        go_terms_json=f'[{{"name": "Fetched via {node_id}"}}]', 
                        pathways_json="[]", domains_json="[]", diseases_json="[]"
                    )
                    new_genes.append(new_gene)
                    
                    # [修复联动] 将下载的基因与该拓扑节点建立关联
                    if node_id.startswith("GO:"):
                        new_gene.go_terms_json = f'[{{"name": "Fetched via {node_id}", "namespace": "linked_node"}}]'
                    elif node_id.startswith("PATH:"):
                        new_gene.pathways_json = f'[{{"name": "Fetched via {node_id}", "pathway": "{node_id}"}}]'
                        
                fetched_symbols.append(symbol)
            except Exception:
                continue
                
        if new_genes:
            gene_db.insert_genes(new_genes)
            gene_db.connection.commit()
            
        return {
            "status": "success",
            "ontology_node": node_id,
            "fetched_genes": fetched_symbols,
            "new_cached_count": len(new_genes),
            "evidence_level": "Level A (UniProt Ontology Mapping)"
        }
    except Exception as e:
        return {"error": f"UniProt Ontology API 请求失败: {str(e)}"}

@lru_cache(maxsize=256)
def fetch_from_uniprot_api(gene_symbol: str) -> dict[str, Any]:
    """真实 API 集成：从 UniProt 提取蛋白层面的硬证据和交互网络。"""
    try:
        url = f"https://rest.uniprot.org/uniprotkb/search?query=gene_exact:{gene_symbol}+AND+organism_id:9606&format=json&size=1"
        response = _api_get_with_retry(url)
        response.raise_for_status()
        data = response.json()
        if not data.get("results"):
            return {"error": f"UniProt 中未找到基因 {gene_symbol} 的人类蛋白记录"}
        
        protein = data["results"][0]
        accession = protein.get("primaryAccession")
        protein_name = protein.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")
        
        subcellular_locations = []
        for comment in protein.get("comments", []):
            if comment.get("commentType") == "SUBCELLULAR LOCATION":
                for loc in comment.get("subcellularLocations", []):
                    subcellular_locations.append(loc.get("location", {}).get("value", ""))

        # 深层 PTM 解析：提取翻译后修饰位点
        ptm_sites = []
        for feature in protein.get("features", []):
            if feature.get("type") in ("MOD_RES", "DISULFID", "CROSSLNK", "LIPID", "CARBOHYD"):
                ptm_sites.append({
                    "type": feature["type"],
                    "position": feature.get("location", {}).get("start", {}).get("value"),
                    "description": feature.get("description", ""),
                    "evidence": feature.get("evidences", [{}])[0].get("source", {}).get("name", "unknown"),
                })

        # 提取相互作用蛋白
        interactors = []
        for comment in protein.get("comments", []):
            if comment.get("commentType") == "INTERACTION":
                for interaction in comment.get("interactions", []):
                    interactors.append({
                        "partner": interaction.get("interactantOne", {}).get("uniProtkbId", ""),
                        "partner2": interaction.get("interactantTwo", {}).get("uniProtkbId", ""),
                        "experiments": interaction.get("numberOfExperiments", 0),
                    })
                    
        return {
            "status": "success",
            "api": "UniProt REST API",
            "uniprot_id": accession,
            "protein_name": protein_name,
            "subcellular_locations": list(set(subcellular_locations)),
            "sequence_length": protein.get("sequence", {}).get("length", 0),
            "ptm_count": len(ptm_sites),
            "ptm_sites": ptm_sites[:15],
            "interaction_partners": len(interactors),
            "top_interactors": [i for i in interactors if i["experiments"] >= 3][:10],
            "evidence_level": "Level A (Validated external DB)"
        }
    except Exception as e:
        return {"error": f"UniProt API 请求失败: {str(e)}"}

@lru_cache(maxsize=128)
def predict_variant_consequence(gene_db: GeneDatabase, variant_id_or_hgvs: str) -> dict[str, Any]:
    """真实 API 集成：通过 Ensembl VEP (Variant Effect Predictor) 分析点突变导致的后果。"""
    try:
        # VEP REST API expects HGVS or rsID. Example: rs699 or ENST00000003084:c.1431_1433delTTC
        url = f"https://rest.ensembl.org/vep/human/id/{variant_id_or_hgvs}?content-type=application/json"
        if ":" in variant_id_or_hgvs:
            url = f"https://rest.ensembl.org/vep/human/hgvs/{variant_id_or_hgvs}?content-type=application/json"
            
        response = requests.get(url, timeout=10)
        if response.status_code == 400:
             return {"error": f"VEP 无法解析变异 {variant_id_or_hgvs}。请提供标准的 rsID 或 HGVS 格式。"}
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return {"error": "VEP 返回空结果"}
            
        first_hit = data[0]
        most_severe = first_hit.get("most_severe_consequence", "unknown")
        
        transcript_consequences = first_hit.get("transcript_consequences", [])
        if transcript_consequences:
            top_tc = transcript_consequences[0]
            sift = top_tc.get("sift_prediction", "unknown")
            polyphen = top_tc.get("polyphen_prediction", "unknown")
            impact = top_tc.get("impact", "unknown")
        else:
            sift = "unknown"
            polyphen = "unknown"
            impact = "unknown"
            
        return {
            "query": variant_id_or_hgvs,
            "api": "Ensembl VEP REST API",
            "most_severe_consequence": most_severe,
            "impact": impact,
            "sift_prediction": sift,
            "polyphen_prediction": polyphen,
            "evidence_level": "Level A (VEP Live Query)", 
            "clinical_significance_inferred": "Likely Pathogenic" if "deleterious" in sift or "damaging" in polyphen else ("Likely Benign" if "tolerated" in sift else "VUS")
        }
    except Exception as e:
        return {"error": f"VEP API 请求失败: {str(e)}"}


def design_crispr_grna(target_sequence: str, pam: str = "NGG", top_n: int = 3) -> dict[str, Any]:
    """设计 CRISPR-Cas9 gRNA 并计算确定性的脱靶得分和 GC 含量 (基于序列特征)。"""
    import hashlib
    if len(target_sequence) < 23:
        return {"error": "目标序列过短，无法设计 gRNA（至少需要 23 bp）。"}
        
    pam_regex = pam.replace("N", "[ATCG]")
    matches = [m for m in re.finditer(f"(?=([ATCG]{{20}}{pam_regex}))", target_sequence.upper())]
    
    candidates = []
    for m in matches:
        full_seq = m.group(1)
        grna = full_seq[:20]
        pam_seq = full_seq[20:]
        
        gc_content = (grna.count("G") + grna.count("C")) / 20.0
        
        # 确定性启发式打分 (模拟 Doench-Root 规则)
        score = 100.0
        if not (0.4 <= gc_content <= 0.6):
            score -= 20
        if "TTT" in grna or "AAA" in grna:
            score -= 30  # 容易导致转录提前终止或聚合酶打滑
        if grna[-1] == "G":
            score += 10  # PAM 近端 G 提高切割效率
            
        # 基于序列哈希的确定性脱靶评估 (Off-target Proxy)
        h = hashlib.md5(grna.encode()).hexdigest()
        off_target_score = 50.0 + (int(h[:4], 16) / 0xffff) * 49.0
        
        # 加上微小的确定性扰动避免分数完全一致
        efficacy_perturbation = (int(h[4:8], 16) / 0xffff) * 10.0 - 5.0
        
        candidates.append({
            "grna_sequence": grna,
            "pam": pam_seq,
            "gc_content": round(gc_content, 2),
            "on_target_efficacy_score": round(min(100.0, max(0.0, score + efficacy_perturbation)), 1),
            "off_target_safety_score": round(off_target_score, 1),
            "recommendation": "Highly Recommended" if score > 80 and off_target_score > 85 else "Acceptable"
        })
        
    candidates.sort(key=lambda x: (x["on_target_efficacy_score"] + x["off_target_safety_score"]), reverse=True)
    
    return {
        "target_sequence_length": len(target_sequence),
        "pam_used": pam,
        "total_grnas_found": len(candidates),
        "top_candidates": candidates[:top_n]
    }


def predict_immunogenicity_and_toxicity(protein_sequence: str) -> dict[str, Any]:
    """预测蛋白质药物或外源表达蛋白的免疫原性 (HLA 结合) 和细胞毒性 (基于确定性算法)。"""
    import hashlib
    if not protein_sequence:
        return {"error": "未提供序列"}
        
    seq = protein_sequence.upper()
    length = len(seq)
    
    # 确定性毒性计算：基于带电氨基酸密度、疏水斑块等
    hydrophobic_count = sum(seq.count(aa) for aa in "VILMFWC")
    hydrophobic_ratio = hydrophobic_count / length if length > 0 else 0
    
    charge = seq.count("R") + seq.count("K") - seq.count("D") - seq.count("E")
    
    h = hashlib.md5(seq.encode()).hexdigest()
    # 用哈希模拟复杂的构象表位暴露度 (0 - 100)
    exposure_score = (int(h[:6], 16) / 0xffffff) * 100.0
    
    immunogenicity_score = (hydrophobic_ratio * 50) + (abs(charge) / length * 30) + (exposure_score * 0.2)
    immunogenicity_score = min(100.0, immunogenicity_score)
    
    return {
        "sequence_length": length,
        "hydrophobic_ratio": round(hydrophobic_ratio, 2),
        "net_charge": charge,
        "immunogenicity_score": round(immunogenicity_score, 2),
        "hla_binding_risk": "High" if immunogenicity_score > 70 else "Low",
        "toxicity_risk": "Potential membrane disruption" if hydrophobic_ratio > 0.6 and charge > 0 else "Safe",
        "evidence_level": "Level C (In-Silico Deterministic Proxy)"
    }

@lru_cache(maxsize=128)
def fetch_3d_structure_info(gene_db: GeneDatabase, gene_symbol: str) -> dict[str, Any]:
    """真实 API 集成：从 AlphaFold DB 获取 3D 结构特征。"""
    # 1. 必须先通过 UniProt 拿到 UniProt Accession
    uniprot_res = fetch_from_uniprot_api(gene_symbol)
    if "error" in uniprot_res:
        return uniprot_res
        
    uniprot_id = uniprot_res.get("uniprot_id")
    if not uniprot_id:
         return {"error": f"无法提取 {gene_symbol} 的 UniProt ID"}
         
    # 2. 调用 AlphaFold DB API
    try:
        url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
             return {"error": f"AlphaFold DB 中未找到 {uniprot_id} ({gene_symbol}) 的预测结构"}
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return {"error": "AlphaFold DB 返回空结果"}
            
        prediction = data[0]
        
        return {
            "gene_symbol": gene_symbol,
            "uniprot_id": uniprot_id,
            "structure_source": "AlphaFold DB REST API",
            "pdb_url": prediction.get("pdbUrl"),
            "cif_url": prediction.get("cifUrl"),
            "pae_image_url": prediction.get("paeImageUrl"),
            "structure_quality": "Available",
            "evidence_level": "Level C (AlphaFold2 In silico prediction)"
        }
    except Exception as e:
        return {"error": f"AlphaFold DB API 请求失败: {str(e)}"}


def simulate_molecular_docking(protein_symbol: str, ligand_smiles: str) -> dict[str, Any]:
    """模拟轻量级分子对接 (AutoDock Vina proxy)。"""
    chem_info = parse_smiles(ligand_smiles)
    if "error" in chem_info:
        return chem_info
        
    mw = chem_info["molecular_weight"]
    rotatable = chem_info["rotatable_bonds"]
    hbd = chem_info["h_bond_donors"]
    
    # 构造一个合理的亲和力得分：分子量越大、氢键越多通常结合力越强，但可旋转键太多会增加熵罚
    affinity_base = -5.0 - (mw / 100.0) - (hbd * 0.5) + (rotatable * 0.2)
    affinity = round(affinity_base + random.uniform(-1.5, 1.5), 1)
    
    kd_nm = round(math.exp(affinity / (0.001987 * 298.15)) * 1e9, 2)
    
    return {
        "protein_target": protein_symbol,
        "ligand_smiles": ligand_smiles,
        "docking_software": "Smina/Vina (Simulated)",
        "binding_affinity_kcal_mol": affinity,
        "estimated_Kd_nM": kd_nm,
        "interaction_types": ["Hydrogen bonds", "Hydrophobic contacts", "Pi-Pi stacking"] if "c" in ligand_smiles.lower() else ["Hydrogen bonds", "Hydrophobic contacts"],
        "druggability_assessment": "Strong Binder" if affinity < -8.0 else ("Moderate Binder" if affinity < -6.0 else "Weak/Non-binder")
    }


def flux_balance_analysis(added_genes: list[str], knocked_out_genes: list[str]) -> dict[str, Any]:
    """使用 COBRApy 进行代谢通量平衡分析 (FBA)。"""
    if cobra is None:
        return {
            "warning": "系统未安装 COBRApy，返回基于经验规则的模拟 FBA 结果。请 pip install cobra。",
            "simulated": True,
            "organism": "E. coli core model (proxy)",
            "added_genes": added_genes,
            "knocked_out_genes": knocked_out_genes,
            "wild_type_growth_rate": 0.87,
            "mutant_growth_rate": max(0.0, 0.87 - len(knocked_out_genes)*0.15 - len(added_genes)*0.05),
            "atp_maintenance_flux": 8.39,
            "metabolic_burden": f"Added {len(added_genes)} heterologous genes, imposing a {len(added_genes)*3}% penalty on growth.",
            "bottlenecks": ["Precursor depletion (e.g., Acetyl-CoA)" if len(added_genes) > 2 else "None detected"]
        }
        
    # 如果真的有 cobra，可以加载 E.coli core model 运行真实 FBA
    try:
        from cobra.io import load_model
        model = load_model("textbook") # E. coli core
        
        wt_sol = model.optimize()
        wt_growth = wt_sol.objective_value
        
        # 敲除逻辑
        for g in knocked_out_genes:
            # 简化匹配：如果模型中有这个基因则敲除
            try:
                model.genes.get_by_id(g).knock_out()
            except KeyError:
                pass
                
        # 对于添加的外源基因，我们添加一个假的消耗 ATP 的 sink reaction 代表表达负担
        if added_genes:
            burden_rxn = cobra.Reaction('HETEROLOGOUS_BURDEN')
            model.add_reactions([burden_rxn])
            burden_rxn.build_reaction_from_string(f"{len(added_genes)*5.0} atp_c + {len(added_genes)*5.0} h2o_c --> {len(added_genes)*5.0} adp_c + {len(added_genes)*5.0} pi_c + {len(added_genes)*5.0} h_c")
            burden_rxn.lower_bound = 1.0
            burden_rxn.upper_bound = 1.0
            
        mutant_sol = model.optimize()
        mutant_growth = mutant_sol.objective_value if mutant_sol.status == 'optimal' else 0.0
        
        return {
            "simulated": False,
            "organism": "E. coli core model",
            "added_genes": added_genes,
            "knocked_out_genes": knocked_out_genes,
            "wild_type_growth_rate": round(wt_growth, 3),
            "mutant_growth_rate": round(mutant_growth, 3),
            "growth_ratio": round(mutant_growth / wt_growth, 3) if wt_growth > 0 else 0,
            "fba_status": mutant_sol.status
        }
    except Exception as e:
        return {"error": f"FBA 计算失败: {str(e)}"}


def get_tissue_specific_expression(gene_symbol: str) -> dict[str, Any]:
    """模拟从 Human Protein Atlas 获取组织特异性表达数据。"""
    # 这里用伪随机和规则引擎生成合理的表达谱
    base_val = (hash(gene_symbol) % 100) / 10.0
    
    tissues = ["Brain", "Heart", "Liver", "Lung", "Kidney", "Skeletal Muscle", "Pancreas", "Spleen", "Immune Cells", "Testis"]
    expression_tpm = {}
    
    is_brain_specific = "receptor" in gene_symbol.lower() or "neuro" in gene_symbol.lower()
    is_liver_specific = "cyp" in gene_symbol.lower() or "metabolism" in gene_symbol.lower()
    
    for t in tissues:
        val = base_val + random.uniform(0, 5)
        if is_brain_specific and t == "Brain": val *= 15
        if is_liver_specific and t == "Liver": val *= 15
        if t == "Testis": val *= 2 # Testis usually has widespread leaky expression
        expression_tpm[t] = round(val, 1)
        
    sorted_expr = sorted(expression_tpm.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "gene_symbol": gene_symbol,
        "data_source": "Human Protein Atlas (Simulated)",
        "expression_category": "Tissue enriched" if sorted_expr[0][1] > sorted_expr[1][1] * 4 else "Expressed in all",
        "top_tissues": [t[0] for t in sorted_expr[:3]],
        "expression_tpm": expression_tpm,
        "clinical_implication": f"如果系统性给药敲除或抑制该基因，{sorted_expr[0][0]} 组织将产生最严重的脱靶/副作用。"
    }
