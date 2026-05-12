from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from .config import RuntimeConfig
from .deepseek_client import ChatResult
from .memory import MemoryEntry, MemoryStore
from .model_router import ModelRouter
from .resource_manager import ResourceManager
from .tools import ToolRegistry


def _stable_system_prompt(version: str, failure_categories: tuple[str, ...]) -> str:
    categories = ", ".join(failure_categories)
    return (
        f"你是一个通用数学难题研究代理，协议版本 {version}。\n"
        "目标：在尽量少浪费 token 的前提下，优先依赖工具、再依赖推理，持续产生可追溯的研究资产。\n"
        "硬规则：\n"
        "1. 能调用工具验证的内容，禁止仅凭直觉断言。\n"
        "2. 每轮必须给出结构化 JSON，不允许额外包裹解释文本。\n"
        "3. 【禁止数值穷举】你的目标是寻找严格的数学证明或结构性反证。绝对禁止使用任何工具进行大规模的数值扫表或穷举。所有数值工具只能用于小范围验证一个局部代数猜想的正确性。\n"
        "4. 【引理驱动证明】将大问题拆解为清晰的子引理（Lemmas）。每个子引理必须有明确的逻辑推导链。使用 proof_tree_manager 工具来注册和验证你的引理结构。\n"
        "5. 失败路径必须分类，并明确失败原因（如逻辑断裂、反例存在、循环论证等）。\n"
        "6. 优先复用稳定前缀和已有高价值记忆，降低缓存失效率。\n"
        "失败原因类别限定为："
        f"{categories}。\n"
        "输出 JSON 结构：\n"
        "{\n"
        '  "status": "continue|pause|complete|needs_input",\n'
        '  "logical_motivation": "调用工具前必须声明的逻辑推理与结构假设，严禁写无逻辑的盲目试错",\n'
        '  "research_log": "本轮研究日志摘要",\n'
        '  "high_value_findings": [\n'
        '    {"category": "Heuristic|Lemma|Conjecture|FailedPath|SearchFrontier|ToolFinding", "title": "...", "content": "...", "tags": ["..."], "importance": 0-100}\n'
        "  ],\n"
        '  "tool_requests": [\n'
        '    {"tool": "工具名", "args": {"参数名": "参数值"}}\n'
        "  ],\n"
        '  "next_focus": "下一步焦点",\n'
        '  "checkpoint_recommendation": "none|mini|major"\n'
        "}\n"
    )


@dataclass(slots=True)
class ResearchRuntime:
    memory_store: MemoryStore
    tool_registry: ToolRegistry
    model_router: ModelRouter
    runtime_config: RuntimeConfig
    resource_manager: ResourceManager

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
    ) -> dict[str, Any]:
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

    def run_iteration(
        self,
        run_id: str,
        iteration: int,
        stream_writer: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        run = self.memory_store.get_run(run_id)
        
        # 获取人类干预信息
        human_interventions = self.memory_store.search_entries(query="", run_id=run_id, categories=["UserOverride"], limit=5)
        
        prompt_messages = self._build_messages(run_id=run_id, problem=run["problem"], iteration=iteration, interventions=human_interventions)
        task_type = self._task_type_for_iteration(iteration)
        
        max_retries = 3
        current_retry = 0
        final_result = None
        final_parsed = None
        tool_results = []
        
        while current_retry < max_retries:
            routed = self.model_router.chat(
                messages=prompt_messages,
                task_type=task_type,
                stream_callback=stream_writer if current_retry == 0 else None,
            )
            final_result = routed.result
            
            parsed = self._parse_json_response(final_result.text)
            tool_results = self._execute_tools(run_id=run_id, tool_requests=parsed.get("tool_requests", []))
            
            # 自我修复回路 (Self-Healing Loop)
            has_error = any("error" in tr for tr in tool_results)
            if parsed.get("status") == "json_parse_error":
                if current_retry < max_retries - 1:
                    warning_content = "【系统警告：JSON 截断重试】你输出的 JSON 被截断了。请用最简短的语句重新输出，确保 research_log 不超过 100 字。"
                    prompt_messages.append({"role": "assistant", "content": final_result.text[:500]})
                    prompt_messages.append({"role": "user", "content": warning_content})
                    current_retry += 1
                    continue
                else:
                    # 重试耗尽：强行使用空结果继续，不挂起
                    final_parsed = {"status": "continue", "logical_motivation": "(parse failed)", "research_log": "JSON parse failed after retries", "high_value_findings": [], "tool_requests": [], "next_focus": "retry", "checkpoint_recommendation": "none"}
                    tool_results = []
                    break
                    
            elif has_error and current_retry < max_retries - 1:
                # 把错误信息作为系统警告追加到 messages 里，让 AI 重试
                error_msgs = [f"Tool '{tr.get('tool')}' failed: {tr.get('error')}" for tr in tool_results if "error" in tr]
                warning_content = "【系统警告：工具调用失败】\n" + "\n".join(error_msgs) + "\n请严格参考 user 提供的 tool_catalog 中的工具名和参数说明。务必修正参数签名，然后重新输出 JSON。"
                prompt_messages.append({"role": "assistant", "content": final_result.text})
                prompt_messages.append({"role": "user", "content": warning_content})
                current_retry += 1
                if stream_writer:
                    stream_writer("\n[Self-Healing] 工具调用出错，正在底层自动重试修正参数...\n")
                continue
                
            final_parsed = parsed
            break

        self._record_raw_output(run_id=run_id, iteration=iteration, result=final_result, backend_name=routed.backend_name, task_type=routed.task_type)
        usage = final_result.usage or {}
        total_used = int(usage.get("total_tokens", 0))
        run = self.memory_store.increment_token_usage(run_id=run_id, token_count=total_used)
        
        self._record_findings(run_id=run_id, findings=final_parsed.get("high_value_findings", []), iteration=iteration)
        checkpoint = self._maybe_checkpoint(run_id=run_id, parsed=final_parsed, run=run)
        status = final_parsed.get("status", "continue")
        if checkpoint["triggered"]:
            status = "pause"
            
        # 清除已被采纳的人类干预记录，避免重复提示
        for intervention in human_interventions:
            try:
                self.memory_store.connection.execute("DELETE FROM entries WHERE entry_id = ?", (intervention["entry_id"],))
                self.memory_store.connection.commit()
            except Exception:
                pass
                
        return {
            "iteration": iteration,
            "status": status,
            "backend": routed.backend_name,
            "task_type": task_type,
            "research_log": parsed.get("research_log", ""),
            "next_focus": parsed.get("next_focus", ""),
            "tool_results": tool_results,
            "usage": usage,
            "checkpoint": checkpoint,
        }

    def _build_messages(self, run_id: str, problem: str, iteration: int, interventions: list[dict[str, Any]] | None = None) -> list[dict[str, str]]:
        stable_prefix = _stable_system_prompt(
            self.runtime_config.stable_system_prompt_version,
            self.runtime_config.failure_categories,
        )
        tool_catalog = self.tool_registry.describe()
        high_value = self.memory_store.recent_entries(run_id, limit=self.runtime_config.high_value_limit, high_value_only=True)
        recent_archive = self.memory_store.recent_entries(run_id, limit=self.runtime_config.archive_limit)
        latest_checkpoint = self.memory_store.latest_checkpoint(run_id)
        resource_snapshot = self.resource_manager.snapshot().to_dict()
        
        # 附上最新的 ProofTree 状态
        try:
            with open("proof_tree_state.json", "r", encoding="utf-8") as f:
                proof_tree = json.load(f)
        except Exception:
            proof_tree = {}
            
        user_content = {
            "problem": problem,
            "iteration": iteration,
            "tool_catalog": tool_catalog,
            "high_value_memory": high_value,
            "recent_archive": recent_archive,
            "latest_checkpoint": latest_checkpoint,
            "resource_snapshot": resource_snapshot,
            "current_proof_tree": proof_tree,
            "instructions": {
                "prefer_tools_before_theory": True,
                "summarize_failed_paths": True,
                "preserve_cacheable_prefix": True,
                "keep_json_stable": True,
            },
        }
        
        if interventions:
            user_content["HUMAN_INTERVENTION"] = [inv["content"] for inv in interventions]
            user_content["instructions"]["MUST_ADDRESS_HUMAN_INTERVENTION"] = "人类研究员给了你最新指示，本轮必须优先响应并调整研究方向。"
            
        return [
            {"role": "system", "content": stable_prefix},
            {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
        ]

    def _task_type_for_iteration(self, iteration: int) -> str:
        if iteration % 10 == 0:
            return "summary"
        if iteration % 3 == 0:
            return "exploration"
        return "reasoning"

    def _record_raw_output(self, run_id: str, iteration: int, result: ChatResult, backend_name: str, task_type: str) -> None:
        self.memory_store.add_entry(
            MemoryEntry(
                run_id=run_id,
                category="RawTrace",
                title=f"模型原始输出 iter={iteration}",
                content=result.text,
                tags=["raw", "model_output", backend_name, task_type],
                importance=10,
                is_high_value=False,
                metadata={"usage": result.usage, "backend": backend_name, "task_type": task_type},
            )
        )

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        # 尝试清理 markdown 代码块标记
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
            "logical_motivation": "JSON 解析失败",
            "research_log": f"模型输出未能解析为 JSON，已尝试自我修复但失败。原始输出片段: {cleaned[:100]}...",
            "high_value_findings": [
                {
                    "category": "FailedPath",
                    "title": "模型输出解析失败",
                    "content": "响应未遵守 JSON 协议，或因超过 max_tokens 被截断。",
                    "tags": ["parser_error", "json"],
                    "importance": 90,
                }
            ],
            "tool_requests": [],
            "next_focus": "修正提示或扩大 max_tokens 限制。",
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
                        title=f"工具调用 {tool_name}",
                        content=json.dumps(result, ensure_ascii=False),
                        tags=["tool", tool_name],
                        importance=40,
                        is_high_value=False,
                        metadata={"args": args},
                    )
                )
            except Exception as exc:  # noqa: BLE001
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

    def _record_findings(self, run_id: str, findings: list[dict[str, Any]], iteration: int) -> None:
        entries: list[MemoryEntry] = []
        for finding in findings:
            entries.append(
                MemoryEntry(
                    run_id=run_id,
                    category=finding.get("category", "Heuristic"),
                    title=finding.get("title", f"iter={iteration} 未命名发现"),
                    content=finding.get("content", ""),
                    tags=list(finding.get("tags", [])),
                    importance=int(finding.get("importance", 50)),
                    is_high_value=int(finding.get("importance", 50)) >= 70,
                    metadata={"iteration": iteration},
                )
            )
        if entries:
            self.memory_store.add_entries(entries)

    def _maybe_checkpoint(self, run_id: str, parsed: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
        total_tokens = int(run["total_tokens"])
        next_major = int(run["next_checkpoint_at"])
        next_mini = int(run["next_mini_checkpoint_at"])
        recommendation = parsed.get("checkpoint_recommendation", "none")

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

        return {"triggered": False, "type": None, "checkpoint_id": None, "summary": None}
