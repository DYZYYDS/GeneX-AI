from __future__ import annotations

import json
from typing import Any

from .gene_database import GeneDatabase

try:
    import networkx as nx
except ImportError:
    nx = None


class GeneKnowledgeGraph:
    """基于 NetworkX 的基因调控网络知识图谱。"""

    def __init__(self, gene_db: GeneDatabase) -> None:
        self.gene_db = gene_db
        self.graph = nx.DiGraph() if nx else None
        if self.graph is not None:
            self._build_graph()

    def _build_graph(self) -> None:
        if self.graph is None:
            return

        genes = self.gene_db.search_genes("", limit=1000)
        for g in genes:
            symbol = g["symbol"]
            self.graph.add_node(symbol, type="gene", name=g["name"])

            for pw in g.get("pathways", []):
                pw_name = pw.get("name")
                if pw_name:
                    self.graph.add_node(pw_name, type="pathway")
                    self.graph.add_edge(symbol, pw_name, relation="part_of_pathway")
                    self.graph.add_edge(pw_name, symbol, relation="contains_gene")

            for go in g.get("go_terms", []):
                go_name = go.get("name")
                if go_name:
                    self.graph.add_node(go_name, type="go_term", namespace=go.get("namespace"))
                    self.graph.add_edge(symbol, go_name, relation="has_function")

            for dis in g.get("diseases", []):
                dis_name = dis.get("name")
                if dis_name:
                    self.graph.add_node(dis_name, type="disease")
                    self.graph.add_edge(symbol, dis_name, relation="associated_with_disease")
                    self.graph.add_edge(dis_name, symbol, relation="caused_by_gene")

    def find_shortest_path(self, source: str, target: str) -> dict[str, Any]:
        if self.graph is None:
            return {"error": "缺少 networkx 库，无法进行图谱计算。请 pip install networkx。"}
        
        if source not in self.graph or target not in self.graph:
            return {"error": f"节点 '{source}' 或 '{target}' 不在知识图谱中。"}

        try:
            path = nx.shortest_path(self.graph, source=source, target=target)
            edges = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                rel = self.graph[u][v].get("relation", "connected_to")
                edges.append(f"{u} -[{rel}]-> {v}")
            return {
                "source": source,
                "target": target,
                "path_length": len(path) - 1,
                "path_nodes": path,
                "path_edges": edges,
                "biological_meaning": "此路径揭示了两个看似无关的生物学实体之间的隐藏分子级联联系。"
            }
        except nx.NetworkXNoPath:
            return {"error": f"在 '{source}' 和 '{target}' 之间没有找到已知通路。"}

    def find_central_genes(self, limit: int = 10) -> dict[str, Any]:
        if self.graph is None:
            return {"error": "缺少 networkx 库。"}
        
        gene_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get("type") == "gene"]
        if not gene_nodes:
            return {"error": "图谱中没有基因节点。"}

        degree_dict = dict(self.graph.degree(gene_nodes))
        sorted_genes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for symbol, degree in sorted_genes[:limit]:
            results.append({"gene": symbol, "connectivity_degree": degree, "role": "Hub Gene (关键枢纽)"})
            
        return {"central_genes": results, "analysis": "这些高连通性基因通常是生物网络的脆弱点或关键调控因子(Hubs)。"}

    def predict_disease_comorbidity(self, disease_name: str) -> dict[str, Any]:
        if self.graph is None:
            return {"error": "缺少 networkx 库。"}
            
        if disease_name not in self.graph:
            return {"error": f"疾病 '{disease_name}' 不在图谱中。"}
            
        genes = [v for u, v, d in self.graph.edges(disease_name, data=True) if d.get("relation") == "caused_by_gene"]
        
        comorbidities: dict[str, int] = {}
        shared_genes: dict[str, list[str]] = {}
        
        for g in genes:
            for _, dis, d in self.graph.edges(g, data=True):
                if d.get("relation") == "associated_with_disease" and dis != disease_name:
                    comorbidities[dis] = comorbidities.get(dis, 0) + 1
                    if dis not in shared_genes:
                        shared_genes[dis] = []
                    shared_genes[dis].append(g)
                    
        sorted_comorb = sorted(comorbidities.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "query_disease": disease_name,
            "associated_genes": genes,
            "predicted_comorbidities": [
                {"disease": dis, "shared_gene_count": count, "shared_genes": shared_genes[dis]}
                for dis, count in sorted_comorb[:5]
            ],
            "conclusion": "具有大量共享致病基因的疾病可能在临床上表现为并发症，或具有相似的潜在分子病理机制。"
        }

def analyze_gene_network(gene_db: GeneDatabase, action: str, **kwargs: Any) -> dict[str, Any]:
    kg = GeneKnowledgeGraph(gene_db)
    if action == "shortest_path":
        return kg.find_shortest_path(kwargs.get("source", ""), kwargs.get("target", ""))
    elif action == "central_genes":
        return kg.find_central_genes(int(kwargs.get("limit", 10)))
    elif action == "disease_comorbidity":
        return kg.predict_disease_comorbidity(kwargs.get("disease_name", ""))
    else:
        return {"error": f"未知的网络分析操作: {action}"}
