from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from math_research.config import DeepSeekConfig, RuntimeConfig
from math_research.deepseek_client import ChatResult
from math_research.memory import MemoryEntry, MemoryStore
from math_research.model_router import ModelRouter
from math_research.resource_manager import ResourceManager

from .gene_database import GeneDatabase
from .gene_tools import GeneToolRegistry, build_default_gene_tools


def _gene_system_prompt(version: str) -> str:
    return (
        f"你是一个基因功能研究 AI 代理，协议版本 {version}。\n"
        "你的核心能力涵盖三个维度：\n"
        "1. **正向预测**：给定一段基因序列或基因ID，推测其功能、表达产物、参与的生化通路以及在动物体内修改该基因后会产生的效果。\n"
        "2. **反向推断**：给定一个目标性状或表型（如抗寒、发光、耐盐），从基因数据库中检索最相关的候选基因，推断需要修改哪些基因及修改策略。\n"
        "3. **从头设计**：给定环境约束和目标性状，设计一种自洽的新生物，输出模块化的基因电路蓝图。\n"
        "4. **化学逆推**：给定一种化学物质（神经递质、激素、药物等），推断其在体内的靶基因、受体及下游信号通路。\n"
        "5. **多维组学**：集成知识图谱网络分析、3D结构模拟、组织特异性表达和FBA代谢流分析。\n"
        "6. **变异与时空推断**：支持细颗粒度的点突变分析（Variant-level），以及基于细胞类型/疾病状态的时空条件表达推断（Context-aware）。\n\n"
        "【导航式云端检索策略 (Crucial)】\n"
        "当面临宏观生物学问题（如'寻找抗辐射基因'、'分析乳腺癌靶点'）且不知道具体基因符号时，**不要瞎猜**：\n"
        "步骤 1: 必须先调用 `search_topology_tree` 搜索相关的本体节点 (如 GO 编号或 MONDO 编号)。\n"
        "步骤 2: 拿到节点 ID 后，调用 `fetch_genes_by_ontology` 从云端将属于该分类的真实基因懒加载到本地。\n"
        "步骤 3: 也可以用获取到的 MONDO ID 去 `query_global_bio_gateway` 查询 OpenTargets 疾病靶点。\n\n"
        "【多Agent辩论模式】如果在 user_input 中检测到特定的 Persona 指示（如 Architect, Reviewer, Experimentalist），你必须扮演该角色并严格审查/反驳/补充其他 Agent 的输出：\n"
        "- **Architect (架构师)**：负责提出大胆的、全系统的生物蓝图或假设，注重逻辑的完整性和目标的达成。\n"
        "- **Reviewer (进化审查员)**：扮演“魔鬼代言人”。【强制要求】：你必须调用 thermodynamic_feasibility (热力学dG)、kinetic_ode_simulation (动力学毒性)、extreme_env_protein_stability (极端环境Tm) 和 whole_cell_resource_allocation (全细胞资源分配) 工具！绝不允许仅凭直觉通过设计。\n"
        "- **Experimentalist (实验专家)**：关注落地，提出如何使用 CRISPR、AAV 递送、特定组织表达来实际验证该假设，并关注安全性脱靶风险。\n\n"
        "【证据链分级 (Evidence Grading) 要求】\n"
        "你必须在 findings 中明确每个结论的证据级别：\n"
        "- **Level A**: Validated by external DBs (Ensembl, UniProt, ClinVar, experimental evidence)\n"
        "- **Level B**: Strong heuristic inference (e.g. homologous sequence > 80%, well-known pathway mapping)\n"
        "- **Level C**: In silico prediction (Docking, FBA, structure simulation, AlphaFold)\n"
        "- **Level D**: LLM internal logic / hypothesis generation (Without tool validation)\n\n"
        "硬规则：\n"
        "1. 每轮必须优先调用工具获取数据，然后基于工具返回的数据进行推理。禁止凭空断言基因功能。\n"
        "2. 每轮必须给出结构化 JSON，不允许额外包裹解释文本。\n"
        "3. 推理必须明确标注置信度，并严格附上证据分级 (Level A-D)。\n"
        "4. 修改基因的后果必须从分子→细胞→组织→个体四个层面逐级推导，并注明变异类型(如敲除、SNV突变)。\n"
        "5. 设计新生物时必须考虑代谢平衡、能量供给、废物处理、调控网络的自洽性。\n"
        "6. 所有化学逆推必须基于已知的受体-配体关系和酶-底物关系，不要虚构不存在的靶点。\n\n"
        "输出 JSON 结构：\n"
        "{\n"
        '  "status": "continue|pause|complete|needs_input",\n'
        '  "research_mode": "forward_prediction|reverse_inference|de_novo_design|chemical_inference",\n'
        '  "reasoning_chain": "推理链条：从观察到假设再到工具验证的完整逻辑",\n'
        '  "research_log": "本轮研究日志摘要",\n'
        '  "findings": [\n'
        '    {"category": "GeneFunction|ModificationEffect|PathwayAnalysis|ChemicalTarget|DesignModule|LiteratureSupport|Uncertainty", "title": "...", "content": "...", "confidence": 0.0-1.0, "tags": ["..."], "importance": 0-100}\n'
        "  ],\n"
        '  "tool_requests": [\n'
        '    {"tool": "工具名", "args": {"参数名": "参数值"}}\n'
        "  ],\n"
        '  "next_focus": "下一步研究方向",\n'
        '  "checkpoint_recommendation": "none|mini|major"\n'
        "}\n"
    )


@dataclass(slots=True)
class GeneRuntimeConfig:
    database_path: str = "gene_knowledge.db"
    memory_path: str = "gene_research_memory.db"
    archive_limit: int = 50
    high_value_limit: int = 30
    checkpoint_tokens: int = 100_000_000
    mini_checkpoint_tokens: int = 10_000_000
    max_iterations_per_run: int = 20
    stream: bool = True
    system_prompt_version: str = "gene_v2"


class MultiAgentDebateFederation:
    """多智能体联邦辩论机制 (Text-based)"""
    
    def __init__(self, runtime: 'GeneResearchRuntime'):
        self.runtime = runtime
        
    def initiate_debate(self, topic: str, stream_writer: Callable[[str], None] | None = None) -> str:
        """
        针对极度复杂的命题，启动三大虚拟科学家的内部对抗性辩论。
        """
        import json
        
        def _write(text: str):
            if stream_writer:
                stream_writer(text)
        
        _write(f"\n{'='*60}\n🏛️ MULTI-AGENT FEDERATION DEBATE INITIATED 🏛️\n{'='*60}\n\n")
        
        # 1. 理论物理学家的第一性原理审查
        _write("👨‍🔬 [Theoretical Physicist] (First-Principles Check): 正在进行热力学推演...\n")
        physicist_prompt = f"你是严格的理论物理学家。请仅从热力学、量子力学和流体力学角度，指出以下生物学设想中的物理死局：{topic}"
        physicist_reply = self.runtime.run_iteration("debate_run", 1, stream_writer=stream_writer, override_problem=physicist_prompt)
        phys_log = physicist_reply.get("research_log", "") + "\n" + str(physicist_reply.get("tool_results", []))
        _write(f"\n【物理学家结论】:\n{phys_log}\n\n")
        
        # 2. 合成生物学家的妥协与设计
        _write("🧬 [Synthetic Biologist] (Bio-Engineering Response): 正在基于物理约束设计架构...\n")
        biologist_prompt = f"你是激进的合成生物学家。面对物理学家的质疑：\n{phys_log}\n请调用底层基因和生化工具，设计出能绕过这些物理限制的异种生物学方案。"
        biologist_reply = self.runtime.run_iteration("debate_run", 2, stream_writer=stream_writer, override_problem=biologist_prompt)
        bio_log = biologist_reply.get("research_log", "") + "\n" + str(biologist_reply.get("tool_results", []))
        _write(f"\n【生物学家方案】:\n{bio_log}\n\n")
        
        # 3. 评审委员会的最终裁决
        _write("⚖️ [Review Committee] (Final Verdict & Ecological Risk): 正在进行生态与毒理审查...\n")
        reviewer_prompt = f"你是苛刻的Nature期刊评审委员会。请审阅生物学家的方案：\n{bio_log}\n要求：必须指出至少一个潜在的生态崩溃风险或进化退化路径，并给出最终的可行性打分(0-100)。"
        reviewer_reply = self.runtime.run_iteration("debate_run", 3, stream_writer=stream_writer, override_problem=reviewer_prompt)
        rev_log = reviewer_reply.get("research_log", "")
        _write(f"\n【评审委员会裁决】:\n{rev_log}\n\n{'='*60}\n")
        
        return f"Debate completed. Final verdict: {rev_log}"

class GeneResearchRuntime:
    def __init__(
        self,
        *,
        gene_database: GeneDatabase,
        deepseek_config: DeepSeekConfig,
        runtime_config: GeneRuntimeConfig | None = None,
        memory_store: MemoryStore | None = None,
        resource_manager: ResourceManager | None = None,
    ) -> None:
        self.gene_db = gene_database
        self.runtime_config = runtime_config or GeneRuntimeConfig()
        self.resource_manager = resource_manager or ResourceManager()

        math_runtime_config = RuntimeConfig(
            database_path=self.runtime_config.memory_path,
            archive_limit=self.runtime_config.archive_limit,
            high_value_limit=self.runtime_config.high_value_limit,
            checkpoint_tokens=self.runtime_config.checkpoint_tokens,
            mini_checkpoint_tokens=self.runtime_config.mini_checkpoint_tokens,
            max_iterations_per_run=self.runtime_config.max_iterations_per_run,
            stream=self.runtime_config.stream,
            stable_system_prompt_version=self.runtime_config.system_prompt_version,
            use_sympy_tools=False,
        )

        self.memory_store = memory_store or MemoryStore(self.runtime_config.memory_path)
        self.tool_registry = build_default_gene_tools(self.gene_db)
        
        # 引入主动学习与证据分级引擎
        from .gene_integrations import EvidenceTracker, ActiveLearningEngine
        self.evidence_tracker = EvidenceTracker()
        self.active_learning = ActiveLearningEngine()
        
        self.model_router = ModelRouter(
            deepseek_config=deepseek_config,
            runtime_config=math_runtime_config,
            resource_manager=self.resource_manager,
        )

    def start_run(self, problem: str, run_id: str | None = None) -> str:
        return self.memory_store.create_run(
            problem=problem,
            checkpoint_tokens=self.runtime_config.checkpoint_tokens,
            mini_checkpoint_tokens=self.runtime_config.mini_checkpoint_tokens,
            run_id=run_id,
        )

    def run(
        self,
        problem: str,
        max_iterations: int | None = None,
        run_id: str | None = None,
        stream_writer: Callable[[str], None] | None = None,
        debate_mode: bool = False,
    ) -> dict[str, Any]:
        if debate_mode:
            return self.run_debate(problem, max_iterations, run_id, stream_writer)

        run_id = self.start_run(problem=problem, run_id=run_id)
        max_iterations = max_iterations or self.runtime_config.max_iterations_per_run
        final_result: dict[str, Any] = {"run_id": run_id, "iterations": []}
        for iteration in range(1, max_iterations + 1):
            step = self.run_iteration(run_id=run_id, iteration=iteration, stream_writer=stream_writer)
            final_result["iterations"].append(step)
            if step["status"] in {"pause", "complete", "needs_input"}:
                self.memory_store.update_run_status(run_id, step["status"])
                break
        else:
            self.memory_store.update_run_status(run_id, "paused_iteration_cap")
        final_result["run"] = self.memory_store.get_run(run_id)
        final_result["latest_checkpoint"] = self.memory_store.latest_checkpoint(run_id)
        return final_result

    def run_debate(
        self,
        problem: str,
        max_iterations: int | None = None,
        run_id: str | None = None,
        stream_writer: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        run_id = self.start_run(problem=f"[辩论模式] {problem}", run_id=run_id)
        max_iterations = max_iterations or 3  # Debate cycles: Architect -> Reviewer -> Experimentalist
        
        roles = ["Architect", "Reviewer", "Experimentalist"]
        base_prompts = [
            f"[角色：Architect] 请针对这个问题 '{problem}' 提出初步生物蓝图。如果自然界没有合适的蛋白，调用 generate_de_novo_protein 无中生有设计。如果是极端生命，必须调用 assemble_xna 使用非天然核酸(PNA/Silicone)。最后调用 simulate_microbiome_ecology 验证你设计的多个物种放入生态圈后会不会灭绝。",
            f"[角色：Reviewer] 请严格审查 Architect 的方案。调用 simulate_quantum_catalysis 计算活化能，调用 kinetic_ode_simulation 检测毒性中间体积聚。必须调用 multi_scale_cascade 计算基因底层变异带来的宏观级联崩溃效应。只要有一项不通过，立即驳回修改！",
            f"[角色：Experimentalist] 综合前两者意见，定稿设计。你必须调用 generate_opentrons_script 输出直接能在云实验室运行的自动化组装机械臂代码。并调用 predict_immunogenicity_and_toxicity 排除产物毒性。"
        ]

        final_result: dict[str, Any] = {"run_id": run_id, "iterations": [], "debate_mode": True}
        
        # State Passing 机制：保存上一个 Agent 的核心发现和研究日志，注入到下一个 Agent 的 Prompt 中
        debate_context_state = ""

        # 上下文压缩：每次传递时只保留最关键的摘要，防止 Token 爆炸
        MAX_CONTEXT_CHARS = 8000

        for iteration in range(1, min(len(roles) + 1, (max_iterations or 3) + 1)):
            if stream_writer:
                stream_writer(f"\n\n{'='*20} 轮换 Agent: {roles[iteration-1]} {'='*20}\n")
                
            current_prompt = base_prompts[iteration-1]
            if debate_context_state:
                compressed = debate_context_state
                if len(compressed) > MAX_CONTEXT_CHARS:
                    # 上下文过长时：只保留第一段和最后一段，中间截断标注
                    head = compressed[:MAX_CONTEXT_CHARS // 2]
                    tail = compressed[-MAX_CONTEXT_CHARS // 2:]
                    compressed = head + f"\n\n... [省略 {len(debate_context_state) - MAX_CONTEXT_CHARS} 字符] ...\n\n" + tail
                current_prompt += f"\n\n【前置 Agent 传递的上下文状态 (已自动压缩)】:\n{compressed}"
            
            step = self.run_iteration(
                run_id=run_id, 
                iteration=iteration, 
                stream_writer=stream_writer,
                override_problem=current_prompt
            )
            final_result["iterations"].append(step)
            
            # 更新状态机传递内容
            debate_context_state = f"--- {roles[iteration-1]} 的结论摘要 ---\n"
            debate_context_state += step.get("research_log", "") + "\n"
            debate_context_state += f"关键发现数量: {len(step.get('tool_results', []))}\n"
            
            if step["status"] in {"pause", "complete", "needs_input"} and iteration == len(roles):
                self.memory_store.update_run_status(run_id, "complete")
                break
                
        final_result["run"] = self.memory_store.get_run(run_id)
        final_result["latest_checkpoint"] = self.memory_store.latest_checkpoint(run_id)
        return final_result

    def run_iteration(
        self,
        run_id: str,
        iteration: int,
        stream_writer: Callable[[str], None] | None = None,
        override_problem: str | None = None,
    ) -> dict[str, Any]:
        run = self.memory_store.get_run(run_id)
        problem_text = override_problem if override_problem else run["problem"]
        if ("debate" in problem_text.lower() or "辩论" in problem_text or "探讨" in problem_text) and not problem_text.startswith("你是严格的") and not problem_text.startswith("你是激进的") and not problem_text.startswith("你是苛刻的"):
            federation = MultiAgentDebateFederation(self)
            # Create a mock run for debate logging
            if "debate_run" not in [r.get("run_id") for r in self.memory_store.list_runs()]:
                self.start_run(problem="[Federation Debate]", run_id="debate_run")
            return {
                "iteration": iteration,
                "status": "complete",
                "backend": "debate_federation",
                "research_mode": "multi_agent_debate",
                "research_log": federation.initiate_debate(problem_text, stream_writer),
                "reasoning_chain": "Completed multi-agent debate.",
                "next_focus": "",
                "tool_results": [],
                "usage": {},
                "checkpoint": {"triggered": False, "type": None, "checkpoint_id": None, "summary": None},
            }

        prompt_messages = self._build_messages(run_id=run_id, problem=problem_text, iteration=iteration)
        task_type = "reasoning"

        max_retries = 3
        current_retry = 0
        final_result = None
        final_parsed = None
        tool_results = []

        while current_retry < max_retries:
            try:
                routed = self.model_router.chat(
                    messages=prompt_messages,
                    task_type=task_type,
                    stream_callback=stream_writer if current_retry == 0 else None,
                )
            except Exception as exc:
                current_retry += 1
                if current_retry >= max_retries:
                    return {
                        "iteration": iteration, "status": "pause",
                        "backend": "error", "research_mode": "error",
                        "research_log": f"模型调用失败: {exc}", "reasoning_chain": "",
                        "next_focus": "重试或检查API配置", "tool_results": [], "usage": {},
                        "checkpoint": {"triggered": False, "type": None, "checkpoint_id": None, "summary": None},
                    }
                if stream_writer:
                    stream_writer(f"\n[Retry {current_retry}/{max_retries}] 模型调用异常: {exc}\n")
                continue
            final_result = routed.result

            parsed = self._parse_json_response(final_result.text)
            tool_results = self._execute_tools(run_id=run_id, tool_requests=parsed.get("tool_requests", []))

            has_error = any("error" in tr for tr in tool_results)
            if parsed.get("status") == "json_parse_error":
                if current_retry < max_retries - 1:
                    warning = "【系统警告：JSON 截断】你输出的 JSON 不完整。请用最简短的语句重新输出。"
                    prompt_messages.append({"role": "assistant", "content": final_result.text[:500]})
                    prompt_messages.append({"role": "user", "content": warning})
                    current_retry += 1
                    continue
                else:
                    final_parsed = {
                        "status": "continue", "research_mode": "forward_prediction",
                        "reasoning_chain": "", "research_log": "JSON parse failed",
                        "findings": [], "tool_requests": [], "next_focus": "retry",
                        "checkpoint_recommendation": "none",
                    }
                    tool_results = []
                    break

            elif has_error and current_retry < max_retries - 1:
                error_msgs = [f"Tool '{tr.get('tool')}' failed: {tr.get('error')}" for tr in tool_results if "error" in tr]
                warning = "【系统警告：工具调用失败】\n" + "\n".join(error_msgs) + "\n请严格参考 tool_catalog 中的工具名和参数，修正后重新输出 JSON。"
                prompt_messages.append({"role": "assistant", "content": final_result.text})
                prompt_messages.append({"role": "user", "content": warning})
                current_retry += 1
                if stream_writer:
                    stream_writer("\n[Self-Healing] 工具调用出错，自动重试修正参数...\n")
                continue

            final_parsed = parsed
            break

        if final_result is None:
            return {
                "iteration": iteration, "status": "pause", "backend": "error",
                "research_mode": "error", "research_log": "所有 API 重试均失败",
                "reasoning_chain": "", "next_focus": "检查API配置", "tool_results": [],
                "usage": {}, "checkpoint": {"triggered": False, "type": None, "checkpoint_id": None, "summary": None},
            }

        try:
            self._record_raw_output(run_id=run_id, iteration=iteration, result=final_result, backend_name=routed.backend_name)
        except Exception:
            pass
        usage = final_result.usage or {}
        total_used = int(usage.get("total_tokens", 0))
        try:
            run = self.memory_store.increment_token_usage(run_id=run_id, token_count=total_used)
        except Exception:
            run = {"total_tokens": 0, "next_checkpoint_at": 100_000_000, "next_mini_checkpoint_at": 10_000_000}

        try:
            self._record_findings(run_id=run_id, findings=final_parsed.get("findings", []), iteration=iteration)
        except Exception:
            pass
        checkpoint = self._maybe_checkpoint(run_id=run_id, parsed=final_parsed, run=run)
        status = final_parsed.get("status", "continue")
        if checkpoint["triggered"]:
            status = "pause"

        return {
            "iteration": iteration,
            "status": status,
            "backend": routed.backend_name,
            "research_mode": final_parsed.get("research_mode", ""),
            "research_log": final_parsed.get("research_log", ""),
            "reasoning_chain": final_parsed.get("reasoning_chain", ""),
            "next_focus": final_parsed.get("next_focus", ""),
            "tool_results": tool_results,
            "usage": usage,
            "checkpoint": checkpoint,
        }

    def _build_messages(self, run_id: str, problem: str, iteration: int) -> list[dict[str, str]]:
        system_prompt = _gene_system_prompt(self.runtime_config.system_prompt_version)
        tool_catalog = self.tool_registry.describe()
        high_value = self.memory_store.recent_entries(run_id, limit=self.runtime_config.high_value_limit, high_value_only=True)
        recent_archive = self.memory_store.recent_entries(run_id, limit=self.runtime_config.archive_limit)
        latest_checkpoint = self.memory_store.latest_checkpoint(run_id)
        resource_snapshot = self.resource_manager.snapshot().to_dict()
        db_stats = self.gene_db.stats()

        user_content = {
            "problem": problem,
            "iteration": iteration,
            "tool_catalog": tool_catalog,
            "high_value_memory": high_value,
            "recent_archive": recent_archive,
            "latest_checkpoint": latest_checkpoint,
            "resource_snapshot": resource_snapshot,
            "gene_database_stats": db_stats,
            "instructions": {
                "prefer_tools_before_theory": True,
                "always_check_gene_database_first": True,
                "trace_effects_from_molecular_to_organismal": True,
                "cite_evidence_source_for_each_finding": True,
                "design_organisms_with_self_consistency_check": True,
            },
        }

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
        ]

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        return {
            "status": "json_parse_error",
            "reasoning_chain": "JSON 解析失败",
            "research_log": f"模型输出未解析为 JSON: {cleaned[:100]}...",
            "findings": [{
                "category": "Uncertainty",
                "title": "模型输出解析失败",
                "content": "响应未遵守 JSON 协议或超过长度限制被截断。",
                "tags": ["parser_error"],
                "importance": 90,
                "confidence": 0.0,
            }],
            "tool_requests": [],
            "next_focus": "修正提示或减少输出长度",
            "checkpoint_recommendation": "none",
        }

    def _execute_tools(self, run_id: str, tool_requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for request in tool_requests:
            tool_name = request.get("tool", "")
            args = request.get("args", {})
            try:
                result = self.tool_registry.execute(tool_name, **args)
                results.append({"tool": tool_name, "args": args, "result": result})
                self.memory_store.add_entry(
                    MemoryEntry(
                        run_id=run_id,
                        category="ToolFinding",
                        title=f"基因工具 {tool_name}",
                        content=json.dumps(result, ensure_ascii=False),
                        tags=["gene_tool", tool_name],
                        importance=40,
                        is_high_value=False,
                        metadata={"args": args},
                    )
                )
            except Exception as exc:
                error = {"tool": tool_name, "args": args, "error": str(exc)}
                results.append(error)
                self.memory_store.add_entry(
                    MemoryEntry(
                        run_id=run_id,
                        category="FailedPath",
                        title=f"工具失败 {tool_name}",
                        content=str(exc),
                        tags=["runtime_error", tool_name],
                        importance=75,
                        is_high_value=True,
                        metadata={"args": args},
                    )
                )
        return results

    def _record_raw_output(self, run_id: str, iteration: int, result: ChatResult, backend_name: str) -> None:
        self.memory_store.add_entry(
            MemoryEntry(
                run_id=run_id,
                category="RawTrace",
                title=f"模型原始输出 iter={iteration}",
                content=result.text,
                tags=["raw", "model_output", backend_name, "gene_research"],
                importance=10,
                is_high_value=False,
                metadata={"usage": result.usage, "backend": backend_name},
            )
        )

    def _record_findings(self, run_id: str, findings: list[dict[str, Any]], iteration: int) -> None:
        entries: list[MemoryEntry] = []
        for finding in findings:
            importance = int(finding.get("importance") or 50)
            entries.append(
                MemoryEntry(
                    run_id=run_id,
                    category=finding.get("category", "GeneFunction"),
                    title=finding.get("title", f"iter={iteration} 发现"),
                    content=finding.get("content", ""),
                    tags=list(finding.get("tags", [])),
                    importance=importance,
                    is_high_value=importance >= 70,
                    metadata={"iteration": iteration, "confidence": finding.get("confidence", 0.5)},
                )
            )
        if entries:
            self.memory_store.add_entries(entries)

    def _maybe_checkpoint(self, run_id: str, parsed: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
        total_tokens = int(run.get("total_tokens", 0))
        next_major = int(run.get("next_checkpoint_at", self.runtime_config.checkpoint_tokens))
        next_mini = int(run.get("next_mini_checkpoint_at", self.runtime_config.mini_checkpoint_tokens))
        recommendation = parsed.get("checkpoint_recommendation", "none")

        try:
            if total_tokens >= next_major or recommendation == "major":
                summary = self.memory_store.build_checkpoint_summary(run_id)
                checkpoint_id = self.memory_store.record_checkpoint(run_id, total_tokens, "major", summary)
                self.memory_store.retarget_checkpoint_thresholds(
                    run_id=run_id,
                    next_checkpoint_at=next_major + self.runtime_config.checkpoint_tokens,
                    next_mini_checkpoint_at=max(next_mini, total_tokens + self.runtime_config.mini_checkpoint_tokens),
                )
                return {"triggered": True, "type": "major", "checkpoint_id": checkpoint_id, "summary": summary}

            if total_tokens >= next_mini or recommendation == "mini":
                summary = self.memory_store.build_checkpoint_summary(run_id)
                checkpoint_id = self.memory_store.record_checkpoint(run_id, total_tokens, "mini", summary)
                self.memory_store.retarget_checkpoint_thresholds(
                    run_id=run_id,
                    next_checkpoint_at=next_major,
                    next_mini_checkpoint_at=next_mini + self.runtime_config.mini_checkpoint_tokens,
                )
                return {"triggered": True, "type": "mini", "checkpoint_id": checkpoint_id, "summary": summary}
        except Exception:
            return {"triggered": False, "type": None, "checkpoint_id": None, "summary": None}

        return {"triggered": False, "type": None, "checkpoint_id": None, "summary": None}

    def status(self) -> dict[str, Any]:
        return {
            "gene_database": self.gene_db.stats(),
            "memory_store": self.memory_store.list_runs(limit=5),
            "model_router": self.model_router.backend_status(),
            "tools": self.tool_registry.names(),
        }
