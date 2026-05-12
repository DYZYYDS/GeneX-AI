from __future__ import annotations

# Copyright (C) 2024 XenoGenesis Project
# This file is part of XenoGenesis.
# 
# XenoGenesis is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Commercial licensing is available. Contact: 2141595982@qq.com
# 
import json
import os
import sys
import time
from pathlib import Path

# 强制 stdout 使用 utf-8 编码，防止 Windows 终端在打印 Emoji 时报 UnicodeEncodeError
sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent))

from math_research.config import DeepSeekConfig, _load_dotenv
from gene_research.gene_database import GeneDatabase
from gene_research.gene_runtime import GeneResearchRuntime, GeneRuntimeConfig
from gene_research.gene_data.seed_knowledge import seed_database


def color(text: str, code: int) -> str:
    return f"\033[{code}m{text}\033[0m"


def banner() -> None:
    print(color("╔══════════════════════════════════════════════════════════╗", 36))
    print(color("║        🧬 Gene Research AI · 基因功能研究系统            ║", 36))
    print(color("║  正向预测 · 反向推断 · 化学逆推 · 从头设计                ║", 36))
    print(color("╚══════════════════════════════════════════════════════════╝", 36))
    print()


def init_database(db_path: str, force_reinit: bool = False) -> GeneDatabase:
    if force_reinit and os.path.exists(db_path):
        os.remove(db_path)
        print(color(f"[INIT] 已删除旧数据库 {db_path}", 33))

    is_new = not os.path.exists(db_path)
    db = GeneDatabase(db_path)

    if is_new or db.gene_count() == 0:
        print(color("[INIT] 正在导入基础种子知识库 (Local Seed)...", 33))
        seed_database(db)
        print(color("[INIT] 基础种子导入完成，图计算测试骨架已就绪。", 32))
        print(color("[INIT] ✅ 懒加载缓存 (Lazy-Loading Cache) 机制启动，未命中基因将自动云端拉取并建库。", 32))
    else:
        stats = db.stats()
        print(color(f"[INIT] 挂载本地缓存数据库: 已缓存 {stats['gene_count']} 个基因节点。", 32))
        print(color("[INIT] ✅ 懒加载缓存 (Lazy-Loading Cache) 机制启动，云端网关随时待命。", 32))
    return db


def show_status(runtime: GeneResearchRuntime) -> None:
    s = runtime.status()
    print(color("\n--- 系统状态 ---", 36))
    print(f"  基因库: {s['gene_database']['gene_count']} 基因")
    print(f"  可用工具: {len(s['tools'])} 个")
    backends = s['model_router']['backends']
    for b in backends:
        status_icon = "✓" if b.get("ok") else "✗"
        print(f"  模型后端 [{b.get('backend', '?')}]: {status_icon}")


def interactive_mode(runtime: GeneResearchRuntime) -> None:
    print(color("\n--- 交互模式 ---", 36))
    print("支持的问题类型:")
    print("  1. 正向预测: 「TP53 敲除会有什么效果？」")
    print("  2. 反向推断: 「想要生物发光，需要修改哪些基因？」")
    print("  3. 化学逆推: 「多巴胺通过哪些受体发挥作用？」")
    print("  4. 从头设计: 「在深海高压环境中设计一种能高效合成氢气的微生物」")
    print("  5. 序列分析: 直接粘贴 DNA 或蛋白质序列")
    print("  输入 'quit' 退出, 'status' 查看状态, 'reinit' 重新导入数据库\n")

    def stream_writer(text: str) -> None:
        print(text, end="", flush=True)

    iteration = 0
    while True:
        try:
            user_input = input(color(f"\n[Iter {iteration}] 请输入研究问题 > ", 33)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("退出。")
            break
        if user_input.lower() == "status":
            show_status(runtime)
            continue
        if user_input.lower() == "reinit":
            global _gene_db
            _gene_db = init_database(runtime.gene_db.database_path, force_reinit=True)
            runtime.gene_db = _gene_db
            from gene_research.gene_tools import build_default_gene_tools
            runtime.tool_registry = build_default_gene_tools(_gene_db)
            print(color("[INIT] 数据库已重建。", 32))
            continue

        iteration += 1
        print(color(f"\n{'='*60}", 36))
        print(color(f"研究问题: {user_input}", 37))
        print(color(f"{'='*60}\n", 36))

        try:
            result = runtime.run(
                problem=user_input,
                max_iterations=3,
                stream_writer=stream_writer,
            )
            print(color(f"\n\n--- 研究完成 (共 {len(result['iterations'])} 轮) ---", 32))
            for it in result["iterations"]:
                print(color(f"\n[第 {it['iteration']} 轮] 模式: {it.get('research_mode', 'N/A')} | 状态: {it['status']}", 35))
                if it.get("reasoning_chain"):
                    print(f"  推理链: {it['reasoning_chain'][:200]}")
                if it.get("research_log"):
                    print(f"  日志: {it['research_log'][:300]}")
                if it.get("tool_results"):
                    print(f"  工具调用: {len(it['tool_results'])} 次")
        except Exception as e:
            print(color(f"\n[ERROR] 研究过程出错: {e}", 31))


def main() -> None:
    banner()

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        print(color("="*60, 33))
        print(color("  【首次运行配置】", 33))
        print(color("  本项目支持任意兼容 OpenAI 格式的 API (如 DeepSeek, 硅基流动, 阿里云, OpenAI 等)。", 36))
        print(color("="*60, 33))
        api_key = input("请输入您的 API Key: ").strip()
        base_url = input("请输入 API Base URL (默认: https://api.openai.com/v1/chat/completions): ").strip()
        if not base_url: base_url = "https://api.openai.com/v1/chat/completions"
        model = input("请输入您要使用的模型名称 (如 gpt-5.5, gemini-3.1-pro 等，默认: gpt-5.5): ").strip()
        if not model: model = "gpt-5.5"
        
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(f"DEEPSEEK_API_KEY={api_key}\n")
            f.write(f"DEEPSEEK_BASE_URL={base_url}\n")
            f.write(f"DEEPSEEK_MODEL={model}\n")
        print(color(f"\n✅ 配置已保存至 {env_path}\n", 32))

    _load_dotenv()
    # 开源版：如果没有配置密钥，仅作警告，允许用户使用本地数学求解器等无需联网的功能
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print(color("[WARN] 未检测到 DEEPSEEK_API_KEY。系统的 AI 辩论和推理功能将受限，但核心硬核求解器仍可被 API 调用。", 33))

    db_path = os.environ.get("GENE_DB_PATH", "gene_knowledge.db")

    print(color("[START] 初始化基因知识库...", 36))
    db = init_database(db_path)

    deepseek_config = DeepSeekConfig(
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions"),
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        max_tokens=int(os.environ.get("DEEPSEEK_MAX_TOKENS", "4096")),
        stream=True,
    )

    runtime_config = GeneRuntimeConfig()

    print(color("[START] 初始化 AI 引擎...", 36))
    runtime = GeneResearchRuntime(
        gene_database=db,
        deepseek_config=deepseek_config,
        runtime_config=runtime_config,
    )

    show_status(runtime)
    interactive_mode(runtime)

    db.close()
    print(color("\n[EXIT] 基因研究系统已关闭。", 36))


if __name__ == "__main__":
    main()
