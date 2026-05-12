from __future__ import annotations

import math
from typing import Any

# 物理与化学基础常数
R_GAS_CONSTANT = 8.314  # J/(mol*K)
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
PLANCK_CONSTANT = 6.62607015e-34  # J*s
AVOGADRO_CONSTANT = 6.02214076e23  # mol^-1
FARADAY_CONSTANT = 96485.3321  # C/mol

class FirstPrinciplesCalculators:
    """物理与化学第一性原理计算底层工具箱，供 AI 自由组合实现宏观仿真"""

    @staticmethod
    def calculate_gibbs_free_energy(enthalpy_j_mol: float, entropy_j_mol_k: float, temperature_k: float) -> dict[str, Any]:
        """
        计算吉布斯自由能 (ΔG = ΔH - TΔS)。
        用于判断在特定温度下，生化反应（或大分子折叠）是否能自发进行。
        """
        delta_g = enthalpy_j_mol - temperature_k * entropy_j_mol_k
        return {
            "temperature_K": temperature_k,
            "delta_G_J_mol": round(delta_g, 2),
            "delta_G_kJ_mol": round(delta_g / 1000, 3),
            "is_spontaneous": delta_g < 0
        }

    @staticmethod
    def arrhenius_reaction_rate(pre_exponential_factor: float, activation_energy_j_mol: float, temperature_k: float) -> dict[str, Any]:
        """
        阿伦尼乌斯方程计算反应速率常数 k = A * exp(-Ea / RT)。
        用于推演不同温度下的酶催化效率、大分子热降解速率等。
        """
        if temperature_k <= 0:
            return {"error": "Absolute temperature must be > 0 K."}
            
        exponent = -activation_energy_j_mol / (R_GAS_CONSTANT * temperature_k)
        try:
            rate_constant = pre_exponential_factor * math.exp(exponent)
        except OverflowError:
            rate_constant = float('inf') if exponent > 0 else 0.0
            
        return {
            "temperature_K": temperature_k,
            "activation_energy_kJ_mol": round(activation_energy_j_mol / 1000, 2),
            "rate_constant": rate_constant
        }

    @staticmethod
    def boltzmann_state_distribution(energy_state_1_j_mol: float, energy_state_2_j_mol: float, temperature_k: float) -> dict[str, Any]:
        """
        计算两个能量态在热力学平衡下的粒子分布比例 (N2/N1 = exp(-(E2-E1)/RT))。
        用于蛋白质构象概率分布、离子通道开放概率等。
        """
        if temperature_k <= 0:
            return {"error": "Absolute temperature must be > 0 K."}
            
        delta_e = energy_state_2_j_mol - energy_state_1_j_mol
        exponent = -delta_e / (R_GAS_CONSTANT * temperature_k)
        
        try:
            ratio = math.exp(exponent)
            p1 = 1.0 / (1.0 + ratio)
            p2 = ratio / (1.0 + ratio)
        except OverflowError:
            p1 = 0.0 if exponent > 0 else 1.0
            p2 = 1.0 if exponent > 0 else 0.0

        return {
            "temperature_K": temperature_k,
            "delta_E_J_mol": delta_e,
            "probability_state_1": round(p1, 6),
            "probability_state_2": round(p2, 6),
            "ratio_N2_to_N1": ratio
        }

    @staticmethod
    def michaelis_menten_kinetics(v_max: float, k_m: float, substrate_conc: float) -> dict[str, Any]:
        """
        米氏方程计算酶促反应速率 v = Vmax * [S] / (Km + [S])。
        """
        if k_m + substrate_conc == 0:
            return {"error": "Km and [S] cannot both be 0."}
            
        velocity = (v_max * substrate_conc) / (k_m + substrate_conc)
        return {
            "substrate_concentration": substrate_conc,
            "velocity": velocity,
            "fraction_of_vmax": round(velocity / v_max, 4) if v_max > 0 else 0.0
        }

    @staticmethod
    def nernst_potential(ion_charge_z: int, conc_out: float, conc_in: float, temperature_k: float = 310.15) -> dict[str, Any]:
        """
        能斯特方程计算特定离子的跨膜平衡电位 E = (RT/zF) * ln([out]/[in])。
        用于计算神经元静息电位和细胞膜电化学梯度。
        """
        if conc_in <= 0 or conc_out <= 0:
            return {"error": "Concentrations must be > 0."}
        if ion_charge_z == 0:
            return {"error": "Ion charge cannot be 0."}
            
        # 结果单位为伏特(V)，转换为毫伏(mV)
        e_volts = (R_GAS_CONSTANT * temperature_k) / (ion_charge_z * FARADAY_CONSTANT) * math.log(conc_out / conc_in)
        e_mv = e_volts * 1000
        
        return {
            "temperature_K": temperature_k,
            "ion_charge": ion_charge_z,
            "equilibrium_potential_mV": round(e_mv, 2)
        }

    @staticmethod
    def poisson_radiation_survival(dose_gray: float, d37_gray: float) -> dict[str, Any]:
        """
        泊松分布计算电离辐射下的细胞存活率 S = exp(-D/D37)。
        D37 为使 37% 细胞存活的致死剂量 (体现了 DNA 修复能力)。
        """
        if d37_gray <= 0:
            return {"error": "D37 must be > 0."}
            
        survival_fraction = math.exp(-dose_gray / d37_gray)
        return {
            "radiation_dose_Gy": dose_gray,
            "D37_tolerance_Gy": d37_gray,
            "survival_probability": survival_fraction,
            "log_kill": round(-math.log10(survival_fraction) if survival_fraction > 0 else float('inf'), 2)
        }

    @staticmethod
    def quantum_tunneling_probability(particle_mass_kg: float, barrier_width_m: float, barrier_energy_j: float, particle_energy_j: float) -> dict[str, Any]:
        """
        量子隧道效应穿透概率 (WKB 近似)：T ≈ exp(-2 * width * sqrt(2 * m * (V - E)) / h_bar)。
        用于极低温酶催化（如氢隧穿）和异种生物学极端环境下的化学反应分析。
        """
        if particle_energy_j >= barrier_energy_j:
            return {"error": "Particle energy is greater than or equal to barrier energy (Classical regime)."}
            
        h_bar = PLANCK_CONSTANT / (2 * math.pi)
        decay_constant = math.sqrt(2 * particle_mass_kg * (barrier_energy_j - particle_energy_j)) / h_bar
        exponent = -2 * barrier_width_m * decay_constant
        
        try:
            transmission_prob = math.exp(exponent)
        except OverflowError:
            transmission_prob = 0.0
            
        return {
            "particle_mass_kg": particle_mass_kg,
            "barrier_width_m": barrier_width_m,
            "barrier_energy_eV": round(barrier_energy_j / 1.602e-19, 2),
            "tunneling_probability": transmission_prob
        }

    @staticmethod
    def brownian_diffusion_time(distance_m: float, diffusion_coefficient_m2_s: float) -> dict[str, Any]:
        """
        布朗运动扩散时间方程：t = x^2 / (2 * D) （一维）或 t = x^2 / (6 * D) （三维）。
        用于计算大分子在细胞质内的平均扩散时间，评估信号传导延迟或空间大分子的拥挤效应。
        """
        if diffusion_coefficient_m2_s <= 0:
            return {"error": "Diffusion coefficient must be > 0."}
            
        time_1d = (distance_m ** 2) / (2 * diffusion_coefficient_m2_s)
        time_3d = (distance_m ** 2) / (6 * diffusion_coefficient_m2_s)
        
        return {
            "distance_um": distance_m * 1e6,
            "diffusion_coefficient_um2_s": diffusion_coefficient_m2_s * 1e12,
            "mean_time_1D_seconds": time_1d,
            "mean_time_3D_seconds": time_3d
        }

    @staticmethod
    def navier_stokes_hagen_poiseuille(radius_m: float, length_m: float, pressure_drop_pa: float, viscosity_pa_s: float) -> dict[str, Any]:
        """
        哈根-泊肃叶方程（纳维-斯托克斯方程在圆管层流中的特解）：Q = (π * r^4 * ΔP) / (8 * η * L)。
        用于模拟血液/体液在血管/导管网络中的流量，微流控芯片设计或树液流动。
        """
        if radius_m <= 0 or length_m <= 0 or viscosity_pa_s <= 0:
            return {"error": "Radius, length, and viscosity must be > 0."}
            
        flow_rate_m3_s = (math.pi * (radius_m ** 4) * pressure_drop_pa) / (8 * viscosity_pa_s * length_m)
        
        # 换算为 uL/min
        flow_rate_ul_min = flow_rate_m3_s * 1e9 * 60
        
        return {
            "radius_um": radius_m * 1e6,
            "pressure_drop_Pa": pressure_drop_pa,
            "flow_rate_uL_min": flow_rate_ul_min,
            "flow_rate_m3_s": flow_rate_m3_s
        }

    @staticmethod
    def lorentz_force_ion_deflection(ion_charge_coulombs: float, velocity_m_s: float, magnetic_field_tesla: float, electric_field_v_m: float = 0.0) -> dict[str, Any]:
        """
        洛伦兹力方程：F = q(E + v × B)。
        用于计算带电离子（如铁、钙）在强磁场或电场环境（如木星卫星、人造电磁发生器）中的偏转力。
        评估磁小体 (Magnetosome) 或离子通道在极端电磁环境下的功能状态。
        """
        # 简化计算：假设速度与磁场完全垂直 (sinθ = 1)
        force_magnetic_n = ion_charge_coulombs * velocity_m_s * magnetic_field_tesla
        force_electric_n = ion_charge_coulombs * electric_field_v_m
        
        total_force_n = math.sqrt(force_magnetic_n**2 + force_electric_n**2) # 假设正交
        
        return {
            "ion_charge_C": ion_charge_coulombs,
            "magnetic_force_N": f"{force_magnetic_n:.2e}",
            "electric_force_N": f"{force_electric_n:.2e}",
            "total_lorentz_force_N": f"{total_force_n:.2e}"
        }

    @staticmethod
    def bragg_diffraction_crystallography(wavelength_m: float, lattice_spacing_m: float, diffraction_order: int = 1) -> dict[str, Any]:
        """
        布拉格衍射方程：nλ = 2d * sin(θ)。
        用于 X 射线晶体学推演，或者评估生物晶体（如鸟嘌呤晶体、病毒衣壳结构色）对特定波长光线的结构性反射。
        """
        if lattice_spacing_m <= 0:
            return {"error": "Lattice spacing must be > 0."}
            
        sin_theta = (diffraction_order * wavelength_m) / (2 * lattice_spacing_m)
        
        if sin_theta > 1.0 or sin_theta < -1.0:
            return {
                "wavelength_nm": wavelength_m * 1e9,
                "lattice_spacing_nm": lattice_spacing_m * 1e9,
                "diffraction_possible": False,
                "message": "Wavelength too large for this lattice spacing."
            }
            
        theta_rad = math.asin(sin_theta)
        theta_deg = math.degrees(theta_rad)
        
        return {
            "wavelength_nm": wavelength_m * 1e9,
            "lattice_spacing_nm": lattice_spacing_m * 1e9,
            "diffraction_order": diffraction_order,
            "diffraction_possible": True,
            "diffraction_angle_degrees": round(theta_deg, 2)
        }

    @staticmethod
    def young_laplace_capillary_action(surface_tension_n_m: float, contact_angle_degrees: float, tube_radius_m: float, fluid_density_kg_m3: float, gravity_m_s2: float = 9.81) -> dict[str, Any]:
        """
        杨-拉普拉斯方程推导的毛细现象上升高度：h = (2 * γ * cos(θ)) / (ρ * g * r)。
        用于计算植物木质部液流极限高度、微小昆虫的饮水机制、或人造血管的物理约束。
        """
        if tube_radius_m <= 0 or fluid_density_kg_m3 <= 0 or gravity_m_s2 <= 0:
            return {"error": "Radius, density, and gravity must be > 0."}
            
        contact_angle_rad = math.radians(contact_angle_degrees)
        
        height_m = (2 * surface_tension_n_m * math.cos(contact_angle_rad)) / (fluid_density_kg_m3 * gravity_m_s2 * tube_radius_m)
        
        return {
            "surface_tension_N_m": surface_tension_n_m,
            "tube_radius_um": tube_radius_m * 1e6,
            "capillary_rise_height_m": round(height_m, 4),
            "capillary_rise_height_cm": round(height_m * 100, 2)
        }
