import unittest
import os
import tempfile
from pathlib import Path

from gene_research.gene_database import GeneDatabase
from gene_research.gene_tools import build_default_gene_tools
from gene_research.gene_ultimate_modules import XNAAssembler, EcologicalSimulator
from gene_research.gene_integrations import GlobalBioDataGateway
from gene_research.gene_advanced_tools import design_crispr_grna, predict_immunogenicity_and_toxicity

class GeneResearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "test_gene.db")
        self.db = GeneDatabase(self.db_path)
        self.registry = build_default_gene_tools(self.db)

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_registry_contains_core_tools(self) -> None:
        names = self.registry.names()
        self.assertIn("assemble_xna", names)
        self.assertIn("simulate_microbiome_ecology", names)
        self.assertIn("design_crispr_grna", names)
        self.assertIn("query_global_bio_gateway", names)

    def test_xna_assembler_determinism(self) -> None:
        """测试 XNA 组装器的确定性与极端环境稳定性计算"""
        res1 = XNAAssembler.assemble_and_evaluate_xna("ATGCATGC", "Silicone")
        res2 = XNAAssembler.assemble_and_evaluate_xna("ATGCATGC", "Silicone")
        self.assertEqual(res1, res2, "XNA 组装器应当具备完全确定性")
        self.assertGreater(res1["estimated_Tm_celsius"], 200.0)
        self.assertEqual(res1["viability_for_extreme_env"], "Optimal")

    def test_ecological_simulator_geological_timescale(self) -> None:
        """测试生态模拟器是否支持万年级地质时间尺度，且不会发生数值溢出"""
        res = EcologicalSimulator.simulate_microbiome_dynamics(
            species_names=["A", "B"],
            initial_populations=[100.0, 50.0],
            growth_rates=[1.0, 0.5],
            interaction_matrix=[[-0.05, -0.01], [0.05, -0.05]],
            simulation_years=10000.0 # 模拟一万年
        )
        self.assertNotIn("error", res)
        self.assertEqual(res["simulation_years"], 10000.0)
        self.assertIn("compute_time_ms", res)
        # LSODA 应当很快完成积分
        self.assertLess(res["compute_time_ms"], 2000.0) 

    def test_deterministic_bio_gateway(self) -> None:
        """测试底层数据网关已去除 Random，对于相同输入具备一致重现性"""
        res1 = GlobalBioDataGateway.query_depmap("TP53", "A549")
        res2 = GlobalBioDataGateway.query_depmap("TP53", "A549")
        self.assertEqual(res1["chronos_score"], res2["chronos_score"])
        self.assertEqual(res1["evidence_level"], "Level C (Deterministic In-Silico Proxy)")

    def test_crispr_design_determinism(self) -> None:
        """测试 CRISPR 设计算法去除了随机性，且评分合理"""
        target_seq = "ATGCGTACGTACGTAGCTAGCTAGCTAGGG" # len > 23 with PAM (NGG)
        res = design_crispr_grna(target_seq, "NGG")
        self.assertNotIn("error", res)
        res2 = design_crispr_grna(target_seq, "NGG")
        if len(res.get("top_candidates", [])) > 0:
            self.assertEqual(res["top_candidates"][0]["on_target_efficacy_score"], 
                             res2["top_candidates"][0]["on_target_efficacy_score"])

    def test_toxicity_prediction(self) -> None:
        """测试毒性预测算法对于极端序列的捕捉能力"""
        res = predict_immunogenicity_and_toxicity("VILMFWCVILMFWC") # 全疏水极性
        self.assertGreater(res["hydrophobic_ratio"], 0.9)
        self.assertEqual(res["evidence_level"], "Level C (In-Silico Deterministic Proxy)")

if __name__ == "__main__":
    unittest.main()
