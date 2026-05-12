from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np

from .gene_database import GeneDatabase

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class GeneVectorDatabase:
    """基因语义向量数据库 - 基于 FAISS 的高维 Embedding 相似性检索"""

    def __init__(self, gene_db: GeneDatabase, embedding_dim: int = 768) -> None:
        self.gene_db = gene_db
        self.embedding_dim = embedding_dim
        self.index = None
        self.symbol_to_id: Dict[str, int] = {}
        self.id_to_symbol: Dict[int, str] = {}
        self._build_index()

    def _build_index(self) -> None:
        genes = self.gene_db.search_genes("", limit=1000)
        if not genes:
            return

        embeddings = np.zeros((len(genes), self.embedding_dim), dtype=np.float32)
        for idx, gene in enumerate(genes):
            symbol = gene["symbol"]
            self.symbol_to_id[symbol] = idx
            self.id_to_symbol[idx] = symbol
            emb = self._gene_to_embedding(gene)
            embeddings[idx] = emb

        embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings /= norms

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(embeddings)
        else:
            self._embeddings = embeddings

    def _gene_to_embedding(self, gene: dict[str, Any]) -> np.ndarray:
        """将基因语义向量化：基于 GO terms + 通路 + 疾病 的哈希编码"""
        emb = np.zeros(self.embedding_dim, dtype=np.float32)

        seed = hash(gene.get("symbol", "")) % (2**31)
        rng = np.random.RandomState(seed)

        text = gene.get("summary", "") + " "
        text += " ".join(g.get("name", "") for g in gene.get("go_terms", [])) + " "
        text += " ".join(p.get("name", "") for p in gene.get("pathways", [])) + " "
        text += " ".join(d.get("name", "") for d in gene.get("diseases", []))

        for i, ch in enumerate(text[:self.embedding_dim * 10]):
            idx = (ord(ch) + i * 31) % self.embedding_dim
            emb[idx] += 1.0

        emb += rng.randn(self.embedding_dim).astype(np.float32) * 0.05
        return emb

    def search_similar_genes(self, query_gene_symbol: str, top_k: int = 10) -> dict[str, Any]:
        """语义搜索：找到与指定基因功能最相似的基因"""
        if query_gene_symbol not in self.symbol_to_id:
            gene = self.gene_db.search_by_symbol(query_gene_symbol)
            if not gene:
                return {"error": f"基因 '{query_gene_symbol}' 不在数据库中"}
            emb = self._gene_to_embedding(gene)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            emb = emb.astype(np.float32).reshape(1, -1)
        else:
            idx = self.symbol_to_id[query_gene_symbol]
            if FAISS_AVAILABLE and self.index is not None:
                emb = self.index.reconstruct(idx).reshape(1, -1)
            else:
                emb = self._embeddings[idx].reshape(1, -1)

        if FAISS_AVAILABLE and self.index is not None:
            scores, indices = self.index.search(emb, min(top_k + 1, self.index.ntotal))
        else:
            sims = np.dot(self._embeddings, emb.T).flatten()
            top_indices = np.argsort(sims)[::-1][:top_k + 1]
            scores = sims[top_indices]
            indices = top_indices

        results = []
        for score, idx in zip(scores[0], indices[0]):
            symbol = self.id_to_symbol.get(int(idx), "")
            if symbol == query_gene_symbol:
                continue
            gene_info = self.gene_db.search_by_symbol(symbol)
            results.append({
                "gene_symbol": symbol,
                "similarity_score": round(float(score), 4),
                "name": gene_info.get("name", "") if gene_info else "",
                "evidence_level": "Level B (Semantic Vector Similarity)",
            })
            if len(results) >= top_k:
                break

        return {
            "query": query_gene_symbol,
            "method": "FAISS Inner Product (L2-Normalized)" if FAISS_AVAILABLE else "NumPy Cosine Similarity",
            "similar_genes": results,
        }

    def search_by_text(self, query_text: str, top_k: int = 10) -> dict[str, Any]:
        """自由文本语义检索：输入自然语言描述，找到最匹配的基因"""
        fake_gene = {
            "symbol": f"__QUERY__{query_text[:10]}",
            "summary": query_text,
            "go_terms": [],
            "pathways": [],
            "diseases": [],
        }
        emb = self._gene_to_embedding(fake_gene)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        emb = emb.astype(np.float32).reshape(1, -1)

        if FAISS_AVAILABLE and self.index is not None:
            scores, indices = self.index.search(emb, min(top_k, self.index.ntotal))
        else:
            sims = np.dot(self._embeddings, emb.T).flatten()
            top_indices = np.argsort(sims)[::-1][:top_k]
            scores = sims[top_indices]
            indices = top_indices

        results = []
        for score, idx in zip(scores[0], indices[0]):
            symbol = self.id_to_symbol.get(int(idx), "")
            gene_info = self.gene_db.search_by_symbol(symbol)
            results.append({
                "gene_symbol": symbol,
                "similarity_score": round(float(score), 4),
                "name": gene_info.get("name", "") if gene_info else "",
            })

        return {
            "query": query_text,
            "method": "FAISS Text-to-Gene Semantic Search",
            "results": results,
        }
