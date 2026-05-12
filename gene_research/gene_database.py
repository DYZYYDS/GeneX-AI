from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class GeneRecord:
    gene_id: str
    symbol: str
    name: str
    aliases: str
    chromosome: str
    start_pos: int
    end_pos: int
    strand: str
    gene_type: str
    summary: str
    go_terms_json: str
    pathways_json: str
    domains_json: str
    diseases_json: str
    sequence_nt: str
    sequence_aa: str


@dataclass(slots=True)
class VariantRecord:
    variant_id: str          # e.g., rs113488022
    gene_id: str
    hgvs_c: str              # e.g., c.1799T>A
    hgvs_p: str              # e.g., p.Val600Glu
    variant_type: str        # missense, nonsense, frameshift, etc.
    clinical_significance: str # Pathogenic, Benign, VUS
    evidence_level: str      # e.g., "Level A: Validated by multiple independent studies"
    phenotype_json: str      # JSON list of phenotypes/diseases


@dataclass(slots=True)
class ExpressionContext:
    context_id: str          # Unique ID
    gene_id: str
    cell_type: str           # e.g., "Hepatocyte", "Neuron"
    developmental_stage: str # e.g., "Embryonic", "Adult"
    disease_state: str       # e.g., "Normal", "Hepatocellular Carcinoma"
    expression_level: float  # TPM or FPKM
    source_db: str           # e.g., "GTEx", "Human Protein Atlas"
    evidence_url: str        # Link to actual data source


@dataclass(slots=True)
class ProteinDomain:
    domain_id: str
    accession: str
    name: str
    description: str
    source_db: str
    consensus_sequence: str


@dataclass(slots=True)
class PathwayRecord:
    pathway_id: str
    name: str
    source_db: str
    description: str
    gene_ids_json: str


@dataclass(slots=True)
class OntologyNode:
    node_id: str             # e.g., GO:0006281, PATH:hsa04110
    name: str                # e.g., DNA repair
    node_type: str           # e.g., GO_BP, GO_MF, GO_CC, PATHWAY, DISEASE, EXTREME_ENV
    description: str         # e.g., The process of restoring DNA after damage.
    parent_id: str | None    # e.g., GO:0006950 (response to stress)


class GeneDatabase:
    def __init__(self, database_path: str) -> None:
        self.database_path = str(Path(database_path))
        self.connection = sqlite3.connect(self.database_path)
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS genes (
                gene_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                name TEXT NOT NULL,
                aliases_json TEXT NOT NULL DEFAULT '[]',
                chromosome TEXT NOT NULL,
                start_pos INTEGER NOT NULL,
                end_pos INTEGER NOT NULL,
                strand TEXT NOT NULL DEFAULT '+',
                gene_type TEXT NOT NULL DEFAULT 'protein_coding',
                summary TEXT NOT NULL DEFAULT '',
                go_terms_json TEXT NOT NULL DEFAULT '[]',
                pathways_json TEXT NOT NULL DEFAULT '[]',
                domains_json TEXT NOT NULL DEFAULT '[]',
                diseases_json TEXT NOT NULL DEFAULT '[]',
                sequence_nt TEXT NOT NULL DEFAULT '',
                sequence_aa TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS protein_domains (
                domain_id TEXT PRIMARY KEY,
                accession TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                source_db TEXT NOT NULL DEFAULT 'Pfam',
                consensus_sequence TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS pathways (
                pathway_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source_db TEXT NOT NULL DEFAULT 'KEGG',
                description TEXT NOT NULL DEFAULT '',
                gene_ids_json TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS variants (
                variant_id TEXT PRIMARY KEY,
                gene_id TEXT NOT NULL,
                hgvs_c TEXT NOT NULL,
                hgvs_p TEXT NOT NULL,
                variant_type TEXT NOT NULL,
                clinical_significance TEXT NOT NULL,
                evidence_level TEXT NOT NULL,
                phenotype_json TEXT NOT NULL DEFAULT '[]',
                FOREIGN KEY(gene_id) REFERENCES genes(gene_id)
            );

            CREATE TABLE IF NOT EXISTS expression_contexts (
                context_id TEXT PRIMARY KEY,
                gene_id TEXT NOT NULL,
                cell_type TEXT NOT NULL,
                developmental_stage TEXT NOT NULL,
                disease_state TEXT NOT NULL,
                expression_level REAL NOT NULL,
                source_db TEXT NOT NULL,
                evidence_url TEXT NOT NULL,
                FOREIGN KEY(gene_id) REFERENCES genes(gene_id)
            );

            CREATE TABLE IF NOT EXISTS go_terms (
                go_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                namespace TEXT NOT NULL,
                definition TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS ontology_nodes (
                node_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                node_type TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                parent_id TEXT
            );

            CREATE TABLE IF NOT EXISTS gene_domain_links (
                gene_id TEXT NOT NULL,
                domain_id TEXT NOT NULL,
                start_pos INTEGER NOT NULL DEFAULT 0,
                end_pos INTEGER NOT NULL DEFAULT 0,
                evalue REAL NOT NULL DEFAULT 0.0,
                PRIMARY KEY (gene_id, domain_id, start_pos),
                FOREIGN KEY (gene_id) REFERENCES genes(gene_id),
                FOREIGN KEY (domain_id) REFERENCES protein_domains(domain_id)
            );

            CREATE TABLE IF NOT EXISTS research_notes (
                note_id TEXT PRIMARY KEY,
                gene_id TEXT,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                evidence_json TEXT NOT NULL DEFAULT '{}',
                confidence REAL NOT NULL DEFAULT 0.5,
                created_at REAL NOT NULL,
                FOREIGN KEY (gene_id) REFERENCES genes(gene_id)
            );

            CREATE INDEX IF NOT EXISTS idx_genes_symbol ON genes(symbol);
            CREATE INDEX IF NOT EXISTS idx_genes_chromosome ON genes(chromosome, start_pos);
            CREATE INDEX IF NOT EXISTS idx_genes_type ON genes(gene_type);
            CREATE INDEX IF NOT EXISTS idx_domains_source ON protein_domains(source_db);
            CREATE INDEX IF NOT EXISTS idx_research_notes_gene ON research_notes(gene_id, category);
            """
        )
        self.connection.commit()

    @staticmethod
    def _escape_like(value: str) -> str:
        return value.replace("%", "\\%").replace("_", "\\_")

    def insert_gene(self, gene: GeneRecord) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO genes(
                gene_id, symbol, name, aliases_json, chromosome, start_pos, end_pos,
                strand, gene_type, summary, go_terms_json, pathways_json, domains_json,
                diseases_json, sequence_nt, sequence_aa
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                gene.gene_id, gene.symbol, gene.name, gene.aliases,
                gene.chromosome, gene.start_pos, gene.end_pos, gene.strand,
                gene.gene_type, gene.summary, gene.go_terms_json, gene.pathways_json,
                gene.domains_json, gene.diseases_json, gene.sequence_nt, gene.sequence_aa,
            ),
        )
        self.connection.commit()

    def insert_genes(self, genes: Iterable[GeneRecord]) -> None:
        payload = [
            (
                g.gene_id, g.symbol, g.name, g.aliases,
                g.chromosome, g.start_pos, g.end_pos, g.strand,
                g.gene_type, g.summary, g.go_terms_json, g.pathways_json,
                g.domains_json, g.diseases_json, g.sequence_nt, g.sequence_aa,
            )
            for g in genes
        ]
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO genes(
                gene_id, symbol, name, aliases_json, chromosome, start_pos, end_pos,
                strand, gene_type, summary, go_terms_json, pathways_json, domains_json,
                diseases_json, sequence_nt, sequence_aa
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        self.connection.commit()

    def insert_domain(self, domain: ProteinDomain) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO protein_domains(domain_id, accession, name, description, source_db, consensus_sequence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (domain.domain_id, domain.accession, domain.name, domain.description, domain.source_db, domain.consensus_sequence),
        )
        self.connection.commit()

    def insert_domains(self, domains: Iterable[ProteinDomain]) -> None:
        payload = [
            (d.domain_id, d.accession, d.name, d.description, d.source_db, d.consensus_sequence)
            for d in domains
        ]
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO protein_domains(domain_id, accession, name, description, source_db, consensus_sequence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        self.connection.commit()

    def insert_pathway(self, pathway: PathwayRecord) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO pathways(pathway_id, name, source_db, description, gene_ids_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (pathway.pathway_id, pathway.name, pathway.source_db, pathway.description, pathway.gene_ids_json),
        )
        self.connection.commit()

    def insert_pathways(self, pathways: Iterable[PathwayRecord]) -> None:
        payload = [
            (p.pathway_id, p.name, p.source_db, p.description, p.gene_ids_json)
            for p in pathways
        ]
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO pathways(pathway_id, name, source_db, description, gene_ids_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            payload,
        )
        self.connection.commit()

    def insert_ontology_nodes(self, nodes: Iterable[OntologyNode]) -> None:
        for node in nodes:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO ontology_nodes
                (node_id, name, node_type, description, parent_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (node.node_id, node.name, node.node_type, node.description, node.parent_id)
            )
        self.connection.commit()

    def search_ontology(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        cursor = self.connection.execute(
            """
            SELECT node_id, name, node_type, description, parent_id
            FROM ontology_nodes
            WHERE name LIKE ? OR description LIKE ? OR node_id LIKE ?
            LIMIT ?
            """,
            (f"%{query}%", f"%{query}%", f"%{query}%", limit)
        )
        return [dict(row) for row in cursor.fetchall()]

    def insert_variant(self, variant: VariantRecord) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO variants(variant_id, gene_id, hgvs_c, hgvs_p, variant_type, clinical_significance, evidence_level, phenotype_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (variant.variant_id, variant.gene_id, variant.hgvs_c, variant.hgvs_p, variant.variant_type, variant.clinical_significance, variant.evidence_level, variant.phenotype_json)
        )
        self.connection.commit()

    def insert_expression_context(self, ctx: ExpressionContext) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO expression_contexts(context_id, gene_id, cell_type, developmental_stage, disease_state, expression_level, source_db, evidence_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ctx.context_id, ctx.gene_id, ctx.cell_type, ctx.developmental_stage, ctx.disease_state, ctx.expression_level, ctx.source_db, ctx.evidence_url)
        )
        self.connection.commit()

    def get_variants_for_gene(self, gene_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM variants WHERE gene_id = ?", (gene_id,)).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["phenotypes"] = json.loads(d.pop("phenotype_json"))
            except (json.JSONDecodeError, TypeError):
                d["phenotypes"] = []
            results.append(d)
        return results

    def get_expression_for_gene(self, gene_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM expression_contexts WHERE gene_id = ?", (gene_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_gene(self, gene_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM genes WHERE gene_id = ?", (gene_id,)).fetchone()
        if row is None:
            return None
        return self._gene_row_to_dict(row)

    def search_by_symbol(self, symbol: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM genes WHERE symbol = ? COLLATE NOCASE", (symbol,)
        ).fetchone()
        if row is None:
            return None
        return self._gene_row_to_dict(row)

    def search_genes(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        pattern = f"%{self._escape_like(query)}%"
        rows = self.connection.execute(
            """
            SELECT * FROM genes
            WHERE symbol LIKE ? ESCAPE '\\' OR name LIKE ? ESCAPE '\\' OR summary LIKE ? ESCAPE '\\'
            LIMIT ?
            """,
            (pattern, pattern, pattern, limit),
        ).fetchall()
        return [self._gene_row_to_dict(row) for row in rows]

    def genes_by_chromosome(self, chromosome: str, start: int = 0, end: int = 300_000_000) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT * FROM genes
            WHERE chromosome = ? AND start_pos >= ? AND end_pos <= ?
            ORDER BY start_pos
            """,
            (chromosome, start, end),
        ).fetchall()
        return [self._gene_row_to_dict(row) for row in rows]

    def genes_by_pathway(self, pathway_name: str, limit: int = 50) -> list[dict[str, Any]]:
        pattern = f"%{self._escape_like(pathway_name)}%"
        rows = self.connection.execute(
            """
            SELECT DISTINCT g.* FROM genes g
            WHERE g.pathways_json LIKE ? ESCAPE '\\'
            LIMIT ?
            """,
            (pattern, limit),
        ).fetchall()
        return [self._gene_row_to_dict(row) for row in rows]

    def genes_by_go_term(self, go_term: str, limit: int = 50) -> list[dict[str, Any]]:
        pattern = f"%{self._escape_like(go_term)}%"
        rows = self.connection.execute(
            """
            SELECT DISTINCT g.* FROM genes g
            WHERE g.go_terms_json LIKE ? ESCAPE '\\'
            LIMIT ?
            """,
            (pattern, limit),
        ).fetchall()
        return [self._gene_row_to_dict(row) for row in rows]

    def genes_by_domain(self, domain_name: str, limit: int = 50) -> list[dict[str, Any]]:
        pattern = f"%{self._escape_like(domain_name)}%"
        rows = self.connection.execute(
            """
            SELECT DISTINCT g.* FROM genes g
            WHERE g.domains_json LIKE ? ESCAPE '\\'
            LIMIT ?
            """,
            (pattern, limit),
        ).fetchall()
        return [self._gene_row_to_dict(row) for row in rows]

    def get_domain(self, domain_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM protein_domains WHERE domain_id = ?", (domain_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def search_domains(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        pattern = f"%{self._escape_like(query)}%"
        rows = self.connection.execute(
            """
            SELECT * FROM protein_domains
            WHERE name LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\' OR accession LIKE ? ESCAPE '\\'
            LIMIT ?
            """,
            (pattern, pattern, pattern, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_pathway(self, pathway_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM pathways WHERE pathway_id = ?", (pathway_id,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        try:
            result["gene_ids"] = json.loads(result.pop("gene_ids_json"))
        except (json.JSONDecodeError, TypeError):
            result["gene_ids"] = []
        return result

    def search_pathways(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        pattern = f"%{self._escape_like(query)}%"
        rows = self.connection.execute(
            """
            SELECT * FROM pathways WHERE name LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\'
            LIMIT ?
            """,
            (pattern, pattern, limit),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["gene_ids"] = json.loads(item.pop("gene_ids_json"))
            except (json.JSONDecodeError, TypeError):
                item["gene_ids"] = []
            results.append(item)
        return results

    def add_research_note(
        self,
        note_id: str,
        gene_id: str | None,
        category: str,
        title: str,
        content: str,
        evidence: dict[str, Any] | None = None,
        confidence: float = 0.5,
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO research_notes(note_id, gene_id, category, title, content, evidence_json, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (note_id, gene_id, category, title, content, json.dumps(evidence or {}, ensure_ascii=False), confidence, time.time()),
        )
        self.connection.commit()

    def get_research_notes(self, gene_id: str | None = None, category: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if gene_id:
            clauses.append("gene_id = ?")
            params.append(gene_id)
        if category:
            clauses.append("category = ?")
            params.append(category)
        sql = "SELECT * FROM research_notes"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.connection.execute(sql, params).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["evidence"] = json.loads(item.pop("evidence_json"))
            results.append(item)
        return results

    def gene_count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) as cnt FROM genes").fetchone()
        return int(row["cnt"]) if row else 0

    def domain_count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) as cnt FROM protein_domains").fetchone()
        return int(row["cnt"]) if row else 0

    def pathway_count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) as cnt FROM pathways").fetchone()
        return int(row["cnt"]) if row else 0

    def stats(self) -> dict[str, Any]:
        return {
            "gene_count": self.gene_count(),
            "domain_count": self.domain_count(),
            "pathway_count": self.pathway_count(),
            "gene_types": self._type_counts(),
        }

    def _type_counts(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT gene_type, COUNT(*) as cnt FROM genes GROUP BY gene_type ORDER BY cnt DESC"
        ).fetchall()
        return {row["gene_type"]: int(row["cnt"]) for row in rows}

    def _gene_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        for json_key, attr in [("aliases_json", "aliases"), ("go_terms_json", "go_terms"),
                                ("pathways_json", "pathways"), ("domains_json", "domains"),
                                ("diseases_json", "diseases")]:
            try:
                data[attr] = json.loads(data[json_key])
            except (json.JSONDecodeError, TypeError):
                data[attr] = []
        for key in ("aliases_json", "go_terms_json", "pathways_json", "domains_json", "diseases_json"):
            data.pop(key, None)
        return data

    def close(self) -> None:
        self.connection.close()
