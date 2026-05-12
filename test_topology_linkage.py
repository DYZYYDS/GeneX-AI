import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gene_research.gene_database import GeneDatabase
from gene_research.gene_data.seed_knowledge import seed_database
from gene_research.gene_tools import build_default_gene_tools

print("\n" + "="*80)
print("  拓扑树全链路联动集成测试 (Ontology Linkage Integration Test)")
print("="*80 + "\n")

db_path = '_test_topology_linkage.db'
if os.path.exists(db_path):
    os.remove(db_path)

db = GeneDatabase(db_path)
seed_database(db)
registry = build_default_gene_tools(db)

print(">>> [测试 1] 疾病节点 (MONDO) 联动 OpenTargets >>>")
# AI 寻找癌症相关的节点
ontology_res = registry.execute("search_topology_tree", query="cancer")
mondo_node = [n for n in ontology_res if n['node_id'].startswith("MONDO")][0]
print(f"找到拓扑节点: {mondo_node['node_id']} ({mondo_node['name']})")

# AI 根据 MONDO 节点去查询 OpenTargets 获取靶点
ot_res = registry.execute("query_global_bio_gateway", db_name="OpenTargets", query_id=mondo_node['node_id'])
top_targets = ot_res['top_targets']
print(f"提取到云端靶点: {top_targets}")

# AI 拿着靶点触发本地库懒加载
gene_res = registry.execute("search_gene_database", query=top_targets[0], search_type="symbol")
print(f"懒加载查询靶点 {top_targets[0]}: {'成功' if gene_res.get('found') else '失败'}")


print("\n>>> [测试 2] 极端环境节点 (ENV) 联动 UniProt 跨物种抓取 >>>")
ontology_res = registry.execute("search_topology_tree", query="radiation")
env_node = [n for n in ontology_res if n['node_id'].startswith("ENV")][0]
print(f"找到拓扑节点: {env_node['node_id']} ({env_node['name']})")

# AI 根据 ENV 节点去 UniProt 抓取非人类物种基因
fetch_res = registry.execute("fetch_genes_by_ontology", node_id=env_node['node_id'], limit=2)
print(f"跨物种拉取状态: {fetch_res.get('status', fetch_res.get('error'))}")
if 'fetched_genes' in fetch_res:
    print(f"抓取到的抗辐射基因: {fetch_res['fetched_genes']}")


print("\n>>> [测试 3] 生物过程节点 (GO) 联动 UniProt 抓取 >>>")
ontology_res = registry.execute("search_topology_tree", query="DNA repair")
go_node = [n for n in ontology_res if n['node_id'].startswith("GO")][0]
print(f"找到拓扑节点: {go_node['node_id']} ({go_node['name']})")

fetch_res = registry.execute("fetch_genes_by_ontology", node_id=go_node['node_id'], limit=3)
print(f"拉取状态: {fetch_res.get('status', fetch_res.get('error'))}")
if 'fetched_genes' in fetch_res:
    print(f"抓取到的修复基因: {fetch_res['fetched_genes']}")


print("\n" + "="*80)
print("所有联动测试执行完毕！")
print("="*80)

db.close()
if os.path.exists(db_path):
    os.remove(db_path)
