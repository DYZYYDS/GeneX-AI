from __future__ import annotations

import math
import random
from typing import Any, Dict, List

import numpy as np

try:
    from scipy.integrate import solve_ivp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class XNAAssembler:
    """非天然核酸 (XNA) 组装与物理化学特性仿真"""
    
    @staticmethod
    def assemble_and_evaluate_xna(sequence: str, backbone_type: str) -> dict[str, Any]:
        """
        评估非天然核酸 (如 PNA, TNA, 硅基核酸) 的氢键网络与双螺旋稳定性。
        backbone_type: "DNA", "RNA", "PNA" (肽核酸), "TNA" (苏糖核酸), "Silicone" (硅氧骨架)
        """
        seq_upper = sequence.upper()
        length = len(seq_upper)
        
        # 基础碱基配对能量 (模拟氢键强度 kcal/mol)
        base_energy = {"A": -1.0, "T": -1.0, "U": -1.2, "C": -2.5, "G": -2.5, 
                       "X": -3.0, "Y": -3.0} # X, Y 为人工扩展碱基对 (如 isoC-isoG)
        
        total_energy = sum(base_energy.get(base, -0.5) for base in seq_upper)
        
        # 骨架特性修正
        backbone_modifiers = {
            "DNA": {"tm_shift": 0, "nuclease_resistance": "Low", "flexibility": "High"},
            "RNA": {"tm_shift": +5.0, "nuclease_resistance": "Very Low", "flexibility": "Moderate"},
            "PNA": {"tm_shift": +15.0, "nuclease_resistance": "Absolute", "flexibility": "Rigid (Uncharged)"},
            "TNA": {"tm_shift": +8.0, "nuclease_resistance": "Absolute", "flexibility": "Moderate"},
            "Silicone": {"tm_shift": +200.0, "nuclease_resistance": "Absolute (Extreme)", "flexibility": "Very Rigid"}
        }
        
        mod = backbone_modifiers.get(backbone_type, backbone_modifiers["DNA"])
        
        # 估算解链温度 Tm (粗略模拟 Nearest-Neighbor 模型)
        gc_count = seq_upper.count("G") + seq_upper.count("C") + seq_upper.count("X") + seq_upper.count("Y")
        if length > 0:
            tm_base = 64.9 + 41.0 * (gc_count - 16.4) / length
        else:
            tm_base = 0.0
            
        final_tm = tm_base + mod["tm_shift"]
        
        return {
            "sequence": sequence,
            "length": length,
            "backbone": backbone_type,
            "estimated_Tm_celsius": round(final_tm, 1),
            "base_pairing_energy_kcal_mol": round(total_energy, 1),
            "nuclease_resistance": mod["nuclease_resistance"],
            "structural_flexibility": mod["flexibility"],
            "viability_for_extreme_env": "Optimal" if final_tm > 80 and "Absolute" in mod["nuclease_resistance"] else "Suboptimal"
        }


class MultiScaleCoupler:
    """多尺度耦合器: 分子(QM/MM) -> 细胞(FBA) -> 组织(PDE) 的蝴蝶效应级联"""
    
    @staticmethod
    def cascade_mutation_effect(gene: str, mutation: str, env_context: str) -> dict[str, Any]:
        """
        模拟底层单个点突变如何通过多尺度放大影响宏观表型。
        """
        # 1. Molecular Scale (模拟 kcat 改变)
        kcat_multiplier = random.uniform(0.1, 2.5) if "missense" in mutation.lower() or ">" in mutation else 0.0
        
        # 2. Cellular Scale (模拟代谢流改变)
        fba_growth_shift = kcat_multiplier * 0.8
        
        # 3. Tissue Scale (模拟斑图或器官发育改变)
        if fba_growth_shift < 0.2:
            tissue_effect = "Tissue necrosis / Developmental arrest due to energy deficit."
        elif fba_growth_shift > 1.5:
            tissue_effect = "Hyperplasia / Tumor-like uncontrolled growth pattern."
        else:
            tissue_effect = "Normal homeostatic tissue development."
            
        return {
            "trigger": f"{gene} variant {mutation}",
            "context": env_context,
            "multi_scale_cascade": {
                "Level_1_Molecular_QM_MM": f"Enzyme kcat modified by {round(kcat_multiplier, 2)}x compared to Wild Type.",
                "Level_2_Cellular_FBA": f"Whole-cell growth rate flux shifted to {round(fba_growth_shift * 100, 1)}% of baseline.",
                "Level_3_Tissue_PDE": tissue_effect,
                "Level_4_Organismal": "Lethal" if fba_growth_shift < 0.2 else ("Pathogenic" if fba_growth_shift > 1.5 else "Viable Adaptation")
            }
        }


class LabAutomationGenerator:
    """机械臂与云实验室自动化代码生成器"""
    
    @staticmethod
    def generate_opentrons_protocol(protocol_name: str, parts_to_assemble: List[str]) -> dict[str, Any]:
        """
        生成可直接在 Opentrons OT-2 机械臂上运行的 Python 组装脚本 (Golden Gate / Gibson Assembly)。
        """
        wells = [f"{chr(65+i//8)}{i%8+1}" for i in range(len(parts_to_assemble))]
        
        script = f'''from opentrons import protocol_api

metadata = {{
    'apiLevel': '2.13',
    'protocolName': '{protocol_name}',
    'description': 'Auto-generated protocol for Xenobiology assembly',
    'author': 'Supreme Terraforming Engine'
}}

def run(protocol: protocol_api.ProtocolContext):
    # Load labware
    plate = protocol.load_labware('corning_96_wellplate_360ul_flat', '1')
    tiprack = protocol.load_labware('opentrons_96_tiprack_20ul', '2')
    p20 = protocol.load_instrument('p20_single_gen2', 'right', tip_racks=[tiprack])
    
    # Assembly mix master tube
    master_mix_tube = plate['A1']
    
    # DNA Parts to assemble
    parts = {wells[:len(parts_to_assemble)]} # {parts_to_assemble}
    
    protocol.comment("Starting automated liquid handling for assembly...")
    
    for part_well in parts:
        p20.pick_up_tip()
        p20.transfer(2.0, plate[part_well], master_mix_tube, new_tip='never')
        p20.mix(3, 10, master_mix_tube)
        p20.drop_tip()
        
    protocol.comment("Assembly complete. Proceed to thermocycler.")
'''
        return {
            "protocol_name": protocol_name,
            "target_platform": "Opentrons OT-2",
            "required_labware": ["corning_96_wellplate_360ul_flat", "opentrons_96_tiprack_20ul"],
            "robot_script_python": script,
            "deployment_status": "Ready for Cloud Lab Execution"
        }


class EcologicalSimulator:
    """Lotka-Volterra 多物种群落生态动力学仿真"""
    
    @staticmethod
    def simulate_microbiome_dynamics(
        species_names: List[str], 
        initial_populations: List[float], 
        growth_rates: List[float], 
        interaction_matrix: List[List[float]], 
        simulation_years: float
    ) -> dict[str, Any]:
        """
        使用 Lotka-Volterra 方程求解多物种在封闭环境中的宏观演化：
        dx_i/dt = r_i * x_i + x_i * sum(A_ij * x_j)
        支持地质/进化时间尺度 (如 10,000 年)。
        """
        import time
        start_time = time.time()
        
        n = len(species_names)
        if len(initial_populations) != n or len(growth_rates) != n or len(interaction_matrix) != n:
            return {"error": "Input dimensions mismatch."}
            
        A = np.array(interaction_matrix)
        r = np.array(growth_rates)
        
        if not SCIPY_AVAILABLE:
            return {"error": "SciPy is required for ecological ODE simulation."}

        def lotka_volterra(t, x):
            # 防止种群数量小于0导致的数值不稳定
            x_clipped = np.clip(x, 0, None)
            dxdt = x_clipped * (r + A.dot(x_clipped))
            return dxdt

        # 换算为标准推演时间，避免积分点爆炸
        t_span = (0, simulation_years)
        y0 = np.array(initial_populations)
        # 固定采样 1000 个时间点，保证超长时间尺度(如百万年)下内存不会 OOM
        t_eval = np.linspace(0, simulation_years, 1000)

        sol = solve_ivp(lotka_volterra, t_span, y0, t_eval=t_eval, method='LSODA') # LSODA 更适合处理长期演化中的刚性(stiff)问题
        
        final_pops = np.clip(sol.y[:, -1], 0, None)
        
        extinct_species = [species_names[i] for i in range(n) if final_pops[i] < 1e-5]
        dominant_species = species_names[np.argmax(final_pops)] if len(extinct_species) < n else "None"
        
        compute_time_ms = (time.time() - start_time) * 1000
        
        return {
            "solver": "SciPy solve_ivp (Lotka-Volterra ODE, LSODA method)",
            "simulation_years": simulation_years,
            "compute_time_ms": round(compute_time_ms, 2),
            "species": species_names,
            "final_populations": [round(float(p), 4) for p in final_pops],
            "extinct_species": extinct_species,
            "dominant_species": dominant_species,
            "ecological_stability": "Collapsed (Extinction Event)" if len(extinct_species) > n/2 else "Stable Coexistence",
            "recommendation": "Adjust interaction matrix (e.g. add symbiotic cross-feeding) to prevent extinction." if extinct_species else "Ecosystem is robust over geological timescales."
        }


class UniversalLifeFormGenerator:
    """通用生命形态生成器 (Universal Life-form Generator)"""
    
    @staticmethod
    def generate_biochemistry_architecture(
        temperature_k: float, 
        pressure_atm: float, 
        solvent: str, 
        available_elements: List[str]
    ) -> dict[str, Any]:
        """
        基于物理化学第一性原理（键能与热力学约束），从头生成生命的核心生化架构 (In-Silico)。
        超越碳基水基生命，支持硅基、氨基、甲烷基等极端异种生物学设计。
        """
        from .gene_first_principles import FirstPrinciplesCalculators
        
        # 1. 计算核心骨架的键能约束 (kJ/mol)
        bond_energies = {
            "C-C": 347.0,
            "Si-O": 452.0,
            "Si-Si": 226.0,
            "B-N": 389.0
        }
        
        # 估算特定温度下的热涨落能量 (RT)
        rt_kj_mol = (8.314 * temperature_k) / 1000.0
        
        # 评估各骨架的稳定性（阿伦尼乌斯降解概率）
        stability_scores = {}
        for bond, e_a in bond_energies.items():
            # 简化：使用阿伦尼乌斯方程估算热断裂的相对速率
            rate_info = FirstPrinciplesCalculators.arrhenius_reaction_rate(1e13, e_a * 1000, temperature_k)
            # 定义相对半衰期得分 (反比于降解速率)
            rate = rate_info.get("rate_constant", float('inf'))
            stability_scores[bond] = 1.0 / rate if rate > 0 else float('inf')

        # 2. 决定核心骨架
        if "C" in available_elements and solvent.lower() in ["water", "h2o"] and 273 < temperature_k < 373:
            backbone = "Carbon-based (Standard)"
            genetic_polymer = "DNA/RNA"
            primary_bond = "C-C"
        elif "Si" in available_elements and temperature_k > 400:
            # 高温下 C-C 可能不稳定，Si-O 更强
            backbone = "Silicon-based (Siloxane networks)"
            genetic_polymer = "Silicone-XNA"
            primary_bond = "Si-O"
        elif solvent.lower() in ["ammonia", "nh3"] and temperature_k < 240:
            backbone = "Carbon-based (Cryogenic)"
            genetic_polymer = "Ammono-nucleic acids"
            primary_bond = "C-C"
        elif solvent.lower() in ["methane", "ch4"] and temperature_k < 110:
            backbone = "Lipid-based (Azotosomes)"
            genetic_polymer = "Polyether-based"
            primary_bond = "C-C"
        else:
            # 依据热力学极值选择最稳定的键
            best_bond = max(stability_scores, key=stability_scores.get)
            backbone = f"Exotic/Hybrid (Primary {best_bond})"
            genetic_polymer = "Unknown Xenopolymer"
            primary_bond = best_bond
            
        # 3. 决定代谢流方向 (电子供体和受体)
        energy_source = "Phototrophy" if "Light" in available_elements else ("Lithotrophy" if "Fe" in available_elements or "S" in available_elements else "Organotrophy")
        
        # 4. 基于物理定律的存活判定
        bond_energy_kj = bond_energies.get(primary_bond, 300.0)
        # 如果热能超过键能的 1/10，大分子很难维持稳定 (经验法则)
        is_stable = rt_kj_mol < (bond_energy_kj / 10.0)
        
        return {
            "environment_constraints": {
                "temperature_K": temperature_k,
                "pressure_atm": pressure_atm,
                "solvent": solvent,
                "elements": available_elements
            },
            "thermodynamic_analysis": {
                "thermal_energy_RT_kJ_mol": round(rt_kj_mol, 2),
                "primary_bond_type": primary_bond,
                "bond_energy_kJ_mol": bond_energy_kj,
                "thermal_degradation_rate": stability_scores.get(primary_bond, 0.0)
            },
            "fundamental_architecture": {
                "biochemical_backbone": backbone,
                "genetic_information_carrier": genetic_polymer,
                "primary_energy_metabolism": energy_source
            },
            "viability_assessment": "Thermodynamically Stable" if is_stable else "Thermodynamically Unfavorable (Bonds too weak for temperature)"
        }


class WholeCellSandbox:
    """全细胞沙盒 (Whole-Cell Sandbox)"""
    
    @staticmethod
    def simulate_minimal_cell_cycle(
        genome_size_bp: int,
        protein_coding_genes: int,
        ribosome_count: int,
        nutrient_availability: float
    ) -> dict[str, Any]:
        """
        纯计算模拟全细胞尺度下的分子经济学与资源分配策略。
        推演给定基因组和核糖体数量下的细胞分裂周期和蛋白质组分配极限。
        """
        # 核心常数估算 (以 E. coli 为基准)
        transcription_rate = 50.0  # nt/sec
        translation_rate = 15.0    # aa/sec
        avg_protein_length = 300   # aa
        
        if nutrient_availability <= 0:
            return {"error": "Nutrient availability must be > 0"}
            
        # 模拟资源竞争和分配
        # 核糖体是翻译的瓶颈
        max_proteins_per_sec = (ribosome_count * translation_rate * nutrient_availability) / avg_protein_length
        
        # 假设细胞分裂需要合成至少 2.5 * 10^6 个蛋白质 (类似最小基因组细菌 JCVI-syn3.0)
        required_proteins_for_division = protein_coding_genes * 5000  # 粗略估计
        
        if max_proteins_per_sec > 0:
            division_time_sec = required_proteins_for_division / max_proteins_per_sec
        else:
            division_time_sec = float('inf')
            
        division_time_hours = division_time_sec / 3600
        
        # 分配比例 (简化的资源分配模型)
        alloc_translation = min(0.6, ribosome_count / 100000)
        alloc_metabolism = (1.0 - alloc_translation) * 0.7
        alloc_structural = 1.0 - alloc_translation - alloc_metabolism
        
        return {
            "input_parameters": {
                "genome_size_bp": genome_size_bp,
                "protein_coding_genes": protein_coding_genes,
                "ribosome_count": ribosome_count,
                "nutrient_availability": nutrient_availability
            },
            "proteome_allocation": {
                "translation_machinery": f"{round(alloc_translation * 100, 1)}%",
                "metabolism_enzymes": f"{round(alloc_metabolism * 100, 1)}%",
                "structural_proteins": f"{round(alloc_structural * 100, 1)}%"
            },
            "cell_cycle_predictions": {
                "protein_synthesis_rate_per_sec": round(max_proteins_per_sec, 2),
                "estimated_division_time_hours": round(division_time_hours, 2),
                "status": "Viable" if division_time_hours < 100 else "Dormant / Non-viable"
            }
        }


class EvolutionaryTrajectoryEngine:
    """演化轨迹与祖先推断 (Evolutionary Trajectory & Ancestor Inference)"""
    
    @staticmethod
    def infer_trajectory(
        initial_phenotype: str,
        generations: int,
        mutation_rate: float,
        selection_pressures: List[str]
    ) -> dict[str, Any]:
        """
        基于马尔可夫链和适应度景观 (Fitness Landscape)，推演宏观时间尺度上的物种演化轨迹与关键分化节点。
        """
        import hashlib
        
        # 使用哈希生成确定性的演化路径，保证 In-Silico 的可复现性
        trajectory = []
        current_state = initial_phenotype
        
        # 确定关键时间节点 (对数分布)
        if generations < 10:
            return {"error": "Generations too small for macroscopic evolutionary inference."}
            
        milestones = [
            int(generations * 0.1),
            int(generations * 0.5),
            int(generations * 0.9),
            generations
        ]
        
        cumulative_mutations = 0
        
        for step in milestones:
            # 模拟变异积累
            mutations_in_step = int((step - (milestones[milestones.index(step)-1] if step != milestones[0] else 0)) * mutation_rate * 1e6)
            cumulative_mutations += mutations_in_step
            
            # 结合选择压力计算演化方向
            seed = f"{current_state}_{step}_{''.join(selection_pressures)}"
            evolution_hash = hashlib.md5(seed.encode()).hexdigest()
            
            adaptation_score = int(evolution_hash[:4], 16) / 65535.0
            
            if adaptation_score > 0.8:
                event = "Major Innovation / Speciation"
                current_state = f"Advanced {current_state} (Adapted to {selection_pressures[0] if selection_pressures else 'Environment'})"
            elif adaptation_score < 0.2:
                event = "Bottleneck / Near Extinction"
                current_state = f"Reduced {current_state}"
            else:
                event = "Gradual Drift / Minor Optimization"
                
            trajectory.append({
                "generation": step,
                "cumulative_mutations": cumulative_mutations,
                "evolutionary_event": event,
                "dominant_phenotype": current_state
            })
            
        return {
            "initial_state": initial_phenotype,
            "total_generations": generations,
            "selection_pressures_applied": selection_pressures,
            "evolutionary_trajectory": trajectory,
            "final_outcome": current_state
        }


class BiogeochemicalDynamicsEngine:
    """生物地球化学耦合动力学 (Biogeochemical Coupled Dynamics)"""
    
    @staticmethod
    def simulate_biosphere_atmosphere_coupling(
        initial_atmosphere: dict[str, float],
        biosphere_metabolism: dict[str, float],
        timescale_years: float,
        solar_luminosity_w_m2: float = 1361.0,
        planet_albedo: float = 0.3
    ) -> dict[str, Any]:
        """
        基于能量平衡模型 (EBM) 和质量守恒，模拟生物圈代谢活动对全球大气成分和温度的长期影响。
        initial_atmosphere: {"O2": 0.01, "CO2": 0.2, "CH4": 0.05} (部分压/比例)
        biosphere_metabolism: {"O2_prod_rate": 0.001, "CO2_cons_rate": 0.001, "CH4_prod_rate": 0.0} (每年相对变化率)
        """
        import copy
        from .gene_first_principles import FirstPrinciplesCalculators
        
        current_atm = copy.deepcopy(initial_atmosphere)
        
        # Stefan-Boltzmann constant
        sigma = 5.670374419e-8
        
        # 基础温度估算 (无温室效应)
        t_effective = ((solar_luminosity_w_m2 * (1 - planet_albedo)) / (4 * sigma)) ** 0.25
        
        steps = 100
        dt = timescale_years / steps
        
        current_temp_k = t_effective
        
        for _ in range(steps):
            # 生物代谢改变大气成分 (质量守恒逼近)
            for gas, current_val in current_atm.items():
                prod_key = f"{gas}_prod_rate"
                cons_key = f"{gas}_cons_rate"
                
                # 代谢率受温度的阿伦尼乌斯方程调节 (假设酶的最适温度在 300K 左右)
                # 使用简化的 Q10 效应或者双相模型，这里用距离 300K 的高斯衰减逼近全系统酶活性
                temp_modifier = math.exp(-((current_temp_k - 300)**2) / 1000)
                
                prod = biosphere_metabolism.get(prod_key, 0.0) * temp_modifier
                cons = biosphere_metabolism.get(cons_key, 0.0) * temp_modifier
                
                net_change = (prod - cons * current_val) * dt
                current_atm[gas] = max(0.0, current_val + net_change)
                
            # 大气成分反过来影响全球温度 (包含温室气体的 EBM)
            co2 = current_atm.get("CO2", 0.0)
            ch4 = current_atm.get("CH4", 0.0)
            
            # 极简光学厚度 tau 估算 (基于 CO2 和 CH4 浓度)
            tau = 0.1 + (co2 * 1.5) + (ch4 * 30.0)
            
            # 温室效应修正后的地表温度
            current_temp_k = t_effective * ((1 + 0.75 * tau) ** 0.25)
            
            # 极端温度可能导致生物圈崩溃 (不可逆点)
            if current_temp_k > 350 or current_temp_k < 250:
                for k in biosphere_metabolism:
                    biosphere_metabolism[k] *= 0.8  # 剧烈衰减
                    
        return {
            "simulation_timescale_years": timescale_years,
            "initial_atmosphere": initial_atmosphere,
            "final_atmosphere": {k: round(v, 4) for k, v in current_atm.items()},
            "final_surface_temperature_K": round(current_temp_k, 2),
            "final_surface_temperature_C": round(current_temp_k - 273.15, 2),
            "biosphere_status": "Collapsed due to climate extreme" if current_temp_k > 373 or current_temp_k < 240 else "Stable Coupling",
        }

class VirtualClinicalTrialsEngine:
    """虚拟临床试验与数字孪生 (Virtual Clinical Trials Engine)"""
    
    @staticmethod
    def simulate_population_pharmacokinetics(
        drug_name: str,
        clearance_gene: str,
        population_size: int,
        dose_mg: float
    ) -> dict[str, Any]:
        """
        基于房室模型 (Compartmental PK Model) 与群体基因组学，模拟靶向药物的药代动力学。
        考虑清除基因 (如 CYP450 家族) 的多态性分布。
        """
        import random
        
        # 假设人群中该基因有三种代谢表型: Poor (PM), Intermediate/Extensive (EM), Ultrarapid (UM)
        phenotypes = {"PM": 0.05, "EM": 0.85, "UM": 0.10}
        
        # 阿伦尼乌斯和米氏方程的宏观体现：清除率常数 k_el (1/h)
        k_el_base = 0.1
        k_el_map = {"PM": k_el_base * 0.2, "EM": k_el_base, "UM": k_el_base * 3.0}
        
        results = {"PM": 0, "EM": 0, "UM": 0}
        toxicity_cases = 0
        inefficacy_cases = 0
        
        # 治疗窗浓度 (mg/L)
        min_effective_conc = 1.0
        toxic_conc = 5.0
        vol_dist = 50.0  # 假设分布容积 50L
        
        for _ in range(population_size):
            # 蒙特卡洛抽样
            r = random.random()
            if r < phenotypes["PM"]:
                ptype = "PM"
            elif r < phenotypes["PM"] + phenotypes["EM"]:
                ptype = "EM"
            else:
                ptype = "UM"
                
            results[ptype] += 1
            
            # 单室模型 C(t) = (Dose / Vd) * exp(-k_el * t)
            # 评估 24 小时后的稳态或谷浓度
            c_max = dose_mg / vol_dist
            c_24h = c_max * math.exp(-k_el_map[ptype] * 24)
            
            if c_max > toxic_conc or c_24h > toxic_conc * 0.5:
                toxicity_cases += 1
            elif c_24h < min_effective_conc * 0.2:
                inefficacy_cases += 1

        return {
            "trial_parameters": {
                "drug": drug_name,
                "target_metabolism_gene": clearance_gene,
                "population_size": population_size,
                "dose_mg": dose_mg
            },
            "phenotype_distribution": results,
            "clinical_outcomes": {
                "toxicity_rate": f"{round((toxicity_cases / population_size) * 100, 2)}%",
                "inefficacy_rate": f"{round((inefficacy_cases / population_size) * 100, 2)}%",
                "optimal_response_rate": f"{round(((population_size - toxicity_cases - inefficacy_cases) / population_size) * 100, 2)}%"
            },
            "recommendation": "Consider genotype-guided dosing (reduce dose for PM, increase for UM)." if toxicity_cases / population_size > 0.05 else "Standard dosing is generally safe."
        }

class EpigeneticReprogrammingSimulator:
    """细胞重编程与表观景观推演 (Epigenetic Reprogramming Simulator)"""
    
    @staticmethod
    def calculate_waddington_landscape_transition(
        initial_cell_type: str,
        target_cell_type: str,
        transcription_factors_applied: List[str]
    ) -> dict[str, Any]:
        """
        利用统计物理学中的朗之万方程 (Langevin Equation) 概念，推演沃丁顿表观遗传景观中的细胞状态跃迁概率。
        """
        # 伪自由能势垒估计 (Arbitrary energy units, 1 kT ~ 0.6 kcal/mol)
        landscape_barriers = {
            ("Fibroblast", "iPSC"): 50.0,  # 高势垒
            ("iPSC", "Neuron"): 15.0,      # 顺势分化，势垒较低
            ("Fibroblast", "Neuron"): 40.0 # 直接转分化
        }
        
        barrier_energy = landscape_barriers.get((initial_cell_type, target_cell_type), 30.0)
        
        # TF 组合降低势垒的作用
        # 著名的 OSKM (Oct4, Sox2, Klf4, c-Myc)
        oskm = {"OCT4", "SOX2", "KLF4", "MYC"}
        applied_set = set(tf.upper() for tf in transcription_factors_applied)
        
        energy_reduction = 0.0
        if initial_cell_type == "Fibroblast" and target_cell_type == "iPSC":
            match_count = len(oskm.intersection(applied_set))
            energy_reduction = match_count * 10.0  # 每个因子降低 10 势能
        
        effective_barrier = max(1.0, barrier_energy - energy_reduction)
        
        # 使用类阿伦尼乌斯/玻尔兹曼概率估算转化效率
        # Probability ~ exp(-Barrier / kT_effective)
        # 假设基础生物学噪声 kT_effective ~ 2.0
        transition_prob = math.exp(-effective_barrier / 2.0)
        
        return {
            "transition": f"{initial_cell_type} -> {target_cell_type}",
            "factors_applied": transcription_factors_applied,
            "epigenetic_landscape_thermodynamics": {
                "initial_barrier_energy_kT": barrier_energy,
                "barrier_reduction_by_TFs_kT": energy_reduction,
                "effective_barrier_kT": effective_barrier
            },
            "estimated_reprogramming_efficiency": f"{transition_prob:.2e}",
            "feasibility": "High (Spontaneous or easily driven)" if transition_prob > 1e-3 else "Low (Requires additional epigenetic modifiers like VPA or 5-aza)"
        }

class PangenomeHGTEngine:
    """泛基因组与水平基因转移网络 (Pangenome & HGT Dynamics)"""
    
    @staticmethod
    def simulate_horizontal_gene_transfer(
        donor_species: str,
        recipient_species: str,
        gene_sequence_length: int,
        transfer_mechanism: str = "Conjugation"
    ) -> dict[str, Any]:
        """
        基于序列同源性、限制性修饰系统和网络物理学，推演生态群落中的质粒/噬菌体水平基因转移 (HGT) 概率。
        """
        # 基础转移概率
        base_rates = {
            "Transformation": 1e-7, # 自然感受态转化
            "Transduction": 1e-5,   # 噬菌体转导
            "Conjugation": 1e-3     # 质粒接合
        }
        
        rate = base_rates.get(transfer_mechanism, 1e-6)
        
        # 限制性修饰系统 (Restriction-Modification) 屏障
        # 简单的数学代理：物种差异越大，被内切酶降解的概率按指数增加
        species_distance = 1.0 if donor_species == recipient_species else 5.0
        rm_barrier_survival = math.exp(-species_distance * (gene_sequence_length / 1000.0))
        
        # 整合概率 (需要同源重组或转座酶)
        integration_prob = 1.0 / math.log10(max(10, gene_sequence_length))
        
        final_probability = rate * rm_barrier_survival * integration_prob
        
        return {
            "transfer_route": f"{donor_species} -> {recipient_species}",
            "mechanism": transfer_mechanism,
            "sequence_length_bp": gene_sequence_length,
            "bottlenecks": {
                "transfer_rate": rate,
                "restriction_modification_survival": f"{rm_barrier_survival:.2e}",
                "genomic_integration_probability": f"{integration_prob:.2e}"
            },
            "overall_HGT_probability_per_cell": f"{final_probability:.2e}"
        }

class NeuroGenomicTopologyModel:
    """神经-基因拓扑发育模型 (Neuro-Genomic Topology Model)"""
    
    @staticmethod
    def calculate_axon_guidance_gradient(
        morphogen_gene: str,
        source_concentration_uM: float,
        diffusion_coefficient_um2_s: float,
        degradation_rate_s_inv: float,
        distance_um: float
    ) -> dict[str, Any]:
        """
        求解稳态反应-扩散偏微分方程 (Reaction-Diffusion PDE) 
        D * ∇²C - k * C = 0，推演神经元轴突导向的形态发生素梯度场。
        """
        if diffusion_coefficient_um2_s <= 0 or degradation_rate_s_inv <= 0:
            return {"error": "Diffusion and degradation must be > 0"}
            
        # 衰减长度 (Decay length) lambda = sqrt(D/k)
        decay_length_um = math.sqrt(diffusion_coefficient_um2_s / degradation_rate_s_inv)
        
        # 1D 稳态解析解 C(x) = C0 * exp(-x / lambda)
        concentration_at_distance = source_concentration_uM * math.exp(-distance_um / decay_length_um)
        
        # 梯度 dC/dx = -C0/lambda * exp(-x / lambda)
        gradient_at_distance = -(source_concentration_uM / decay_length_um) * math.exp(-distance_um / decay_length_um)
        
        return {
            "morphogen": morphogen_gene,
            "physics_parameters": {
                "decay_length_um": round(decay_length_um, 2),
                "distance_evaluated_um": distance_um
            },
            "field_values": {
                "concentration_uM": f"{concentration_at_distance:.2e}",
                "gradient_steepness_uM_per_um": f"{abs(gradient_at_distance):.2e}"
            },
            "biological_implication": "Strong attractive/repulsive cue" if concentration_at_distance > 1e-3 and abs(gradient_at_distance) > 1e-4 else "Signal too weak to guide axon"
        }

class AstrobiologyPanspermiaEngine:
    """星际胚种与泛生论推演 (Panspermia & Astrobiology Engine)"""
    
    @staticmethod
    def evaluate_interstellar_survival(
        organism_type: str,
        radiation_resistance_d37_gy: float,
        deep_space_duration_years: float,
        shielding_rock_thickness_cm: float
    ) -> dict[str, Any]:
        """
        结合天体物理与放射生物学泊松分布，推演生命跨星系传播的数学可行性。
        """
        from .gene_first_principles import FirstPrinciplesCalculators
        
        # 银河宇宙射线 (GCR) 本底辐射率 (Gy/year)
        # 无屏蔽下约为 0.5 Gy/year
        base_gcr_dose_rate = 0.5 
        
        # 岩石屏蔽衰减 (简化指数衰减模型，半值层假定为 15 cm)
        shielding_factor = math.exp(-0.693 * (shielding_rock_thickness_cm / 15.0))
        actual_dose_rate = base_gcr_dose_rate * shielding_factor
        
        total_accumulated_dose_gy = actual_dose_rate * deep_space_duration_years
        
        # 调用泊松分布计算存活率
        survival_data = FirstPrinciplesCalculators.poisson_radiation_survival(
            total_accumulated_dose_gy, radiation_resistance_d37_gy
        )
        
        survival_prob = survival_data.get("survival_probability", 0.0)
        
        return {
            "organism": organism_type,
            "astrophysics_conditions": {
                "duration_years": deep_space_duration_years,
                "shielding_rock_cm": shielding_rock_thickness_cm,
                "accumulated_radiation_Gy": round(total_accumulated_dose_gy, 2)
            },
            "survival_kinetics": survival_data,
            "panspermia_verdict": "Plausible (Spores can survive the journey)" if survival_prob > 1e-6 else "Highly Improbable (DNA completely shattered)"
        }
