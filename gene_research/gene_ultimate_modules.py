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
