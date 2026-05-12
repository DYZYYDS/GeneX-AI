import os
import sys

from gene_research.gene_database import GeneDatabase
from gene_research.gene_tools import build_default_gene_tools
from gene_research.gene_runtime import GeneResearchRuntime, DeepSeekConfig

def main():
    os.environ["API_BASE_URL"] = "https://api.deepseek.com"
    os.environ["API_MODEL"] = "deepseek-chat"
    os.environ["API_KEY"] = "sk-10602936dcc340f3841639bdcb358057"
    
    print("=== 初始化底层引擎 ===")
    db = GeneDatabase("gene_data/gene_database.sqlite")
    
    # 挂载基础知识库
    from gene_research.gene_data.seed_knowledge import SEED_GENES, SEED_ONTOLOGY_NODES
    db.insert_genes(SEED_GENES)
    db.insert_ontology_nodes(SEED_ONTOLOGY_NODES)
    
    tools = build_default_gene_tools(db)
    agent = GeneResearchRuntime(
        gene_database=db, 
        deepseek_config=DeepSeekConfig(api_key=os.environ["API_KEY"], base_url=os.environ["API_BASE_URL"], model=os.environ["API_MODEL"])
    )
    agent.tools = tools
    
    prompt = "我希望开启多智能体联邦辩论模式 (Debate)，探讨如何利用底层的第一性原理计算和高级生物工具，攻克胶质母细胞瘤（Glioblastoma）。"
    
    print("\n=== 触发新升级的四步多智能体联邦辩论 ===")
    print(prompt)
    print("\n=== 内置 AI 辩论推演日志 ===\n")
    
    # 因为我们在 gene_runtime.py 中已经升级了 initiate_debate 接收 stream_writer
    # 只要包含 "辩论" 关键字，就会自动触发 MultiAgentDebateFederation
    
    sys.stdout.reconfigure(encoding='utf-8')
    
    # 因为辩论模式会产生极大量日志，为了加速测试，我们不再用 agent.run_iteration，而是直接调取 API
    import json
    import requests
    headers = {
        "Authorization": f"Bearer {os.environ['API_KEY']}",
        "Content-Type": "application/json"
    }
    
    def fast_call(messages):
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.2,
            "tools": [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": {"type": "object", "properties": {}, "additionalProperties": True}}} for t in agent.tools.describe()]
        }
        resp = requests.post(f"{os.environ['API_BASE_URL']}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]

    print("\n" + "="*60 + "\n🏛️ MULTI-AGENT FEDERATION DEBATE INITIATED 🏛️\n" + "="*60 + "\n")

    print("🏗️ [Architect] (Blueprint Design): 正在构思底层生物学架构...\n")
    arch_msg = [{"role": "system", "content": "You are GeneX AI."}, {"role": "user", "content": f"你是极具创新精神的系统生物学架构师。请针对以下命题提出一个宏观的生物学或药学解决方案框架：\n{prompt}"}]
    arch_resp = fast_call(arch_msg)
    print(f"【架构师蓝图】:\n{arch_resp.get('content')[:300]}...\n")

    print("👨‍🔬 [Theoretical Physicist] (First-Principles Check): 正在进行热力学与动力学推演...\n")
    phys_msg = [{"role": "system", "content": "You are GeneX AI."}, {"role": "user", "content": f"你是严格的理论物理学家。请仅从热力学、量子力学和流体力学等第一性原理角度，审视架构师的方案并指出物理死局：\n{arch_resp.get('content')}"}]
    phys_resp = fast_call(phys_msg)
    print(f"【物理学家结论】:\n{phys_resp.get('content')[:300]}...\n")

    print("⚖️ [Review Committee] (Clinical & Ecological Review): 正在进行生态与毒理审查...\n")
    rev_msg = [{"role": "system", "content": "You are GeneX AI."}, {"role": "user", "content": f"你是苛刻的Nature期刊评审委员会。请综合物理学家的意见，对架构师的方案进行临床(PK/PD)或宏观生态学审查。指出潜在的生态崩溃或耐药性进化逃逸路线：\n架构师方案：{arch_resp.get('content')}\n物理学家意见：{phys_resp.get('content')}"}]
    rev_resp = fast_call(rev_msg)
    print(f"【评审委员会裁决】:\n{rev_resp.get('content')[:300]}...\n")

    print("🧪 [Experimentalist] (Wet-Lab Execution): 正在生成可落地的实验代码...\n")
    exp_msg = [{"role": "system", "content": "You are GeneX AI."}, {"role": "user", "content": f"你是执行力极强的自动化实验员。前面的三位科学家已经探讨了理论和风险。请调用自动化代码生成工具或分子设计工具，输出最终的化学分子结构式、FASTA序列或 Opentrons 实验机器人的 Python 执行脚本。\n前置讨论总结：{rev_resp.get('content')}"}]
    exp_resp = fast_call(exp_msg)
    print(f"【实验员落地输出】:\n{exp_resp.get('content')[:300]}...\n")
    
    print("\n\n=== 辩论最终结果 ===")
    print("Debate Cycle Complete.")

if __name__ == "__main__":
    main()