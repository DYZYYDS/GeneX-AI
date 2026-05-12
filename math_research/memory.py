from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(slots=True)
class MemoryEntry:
    category: str
    title: str
    content: str
    tags: list[str] = field(default_factory=list)
    importance: int = 50
    is_high_value: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)


class MemoryStore:
    def __init__(self, database_path: str) -> None:
        self.database_path = str(Path(database_path))
        self.connection = sqlite3.connect(self.database_path)
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS entries (
                entry_id TEXT PRIMARY KEY,
                run_id TEXT,
                created_at REAL NOT NULL,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                importance INTEGER NOT NULL,
                is_high_value INTEGER NOT NULL,
                metadata_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                problem TEXT NOT NULL,
                status TEXT NOT NULL,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                next_checkpoint_at INTEGER NOT NULL,
                next_mini_checkpoint_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                total_tokens INTEGER NOT NULL,
                checkpoint_type TEXT NOT NULL,
                summary_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_entries_run_id ON entries(run_id);
            CREATE INDEX IF NOT EXISTS idx_entries_category ON entries(category);
            CREATE INDEX IF NOT EXISTS idx_entries_high_value ON entries(is_high_value, importance DESC, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_checkpoints_run_id ON checkpoints(run_id, created_at DESC);
            """
        )
        
        # Inject known mathematical boundaries to prevent blind searching
        try:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO entries(entry_id, run_id, created_at, category, title, content, tags_json, importance, is_high_value, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "global-beal-boundary",
                    "global",
                    time.time(),
                    "Theorem",
                    "人类已知比尔猜想验证边界",
                    "人类分布式计算(如 BOINC)已穷举验证底数 A,B,C <= 1000 且指数 <= 100 的所有情况，不存在反例。禁止在此区间内进行盲目的数值搜索。研究必须从更高维度（如模约束、p-adic 估值、代数结构）进行推导，或者针对更高范围设定启发式剪枝搜索。",
                    json.dumps(["beal_conjecture", "boundary", "human_knowledge"], ensure_ascii=False),
                    100,
                    1,
                    json.dumps({"global_knowledge": True}, ensure_ascii=False)
                )
            )
            self.connection.execute(
                """
                INSERT OR IGNORE INTO entries(entry_id, run_id, created_at, category, title, content, tags_json, importance, is_high_value, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "global-proof-strategy",
                    "global",
                    time.time(),
                    "Rule",
                    "定理证明方法论指令",
                    "我们不再进行任何形式的数值扫表（人类的超算远比我们强大）。所有的探索必须通过 proof_tree_manager 建立引理，通过 sympy 工具进行代数推演。我们寻找的是逻辑上的不可能性或者必然性，而不是偶然的数值巧合。",
                    json.dumps(["strategy", "proof_theory", "no_brute_force"], ensure_ascii=False),
                    100,
                    1,
                    json.dumps({"global_knowledge": True}, ensure_ascii=False)
                )
            )
        except Exception:
            pass
            
        self.connection.commit()

    def create_run(
        self,
        problem: str,
        checkpoint_tokens: int,
        mini_checkpoint_tokens: int,
        run_id: str | None = None,
    ) -> str:
        run_id = run_id or str(uuid.uuid4())
        now = time.time()
        self.connection.execute(
            """
            INSERT OR REPLACE INTO runs(run_id, created_at, updated_at, problem, status, total_tokens, next_checkpoint_at, next_mini_checkpoint_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, now, now, problem, "running", 0, checkpoint_tokens, mini_checkpoint_tokens),
        )
        self.connection.commit()
        return run_id

    def get_run(self, run_id: str) -> dict[str, Any]:
        row = self.connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"run_id 不存在: {run_id}")
        return dict(row)

    def update_run_status(self, run_id: str, status: str) -> None:
        self.connection.execute(
            "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
            (status, time.time(), run_id),
        )
        self.connection.commit()

    def add_entry(self, entry: MemoryEntry) -> None:
        self.connection.execute(
            """
            INSERT INTO entries(entry_id, run_id, created_at, category, title, content, tags_json, importance, is_high_value, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.entry_id,
                entry.run_id,
                entry.created_at,
                entry.category,
                entry.title,
                entry.content,
                json.dumps(entry.tags, ensure_ascii=False),
                entry.importance,
                1 if entry.is_high_value else 0,
                json.dumps(entry.metadata, ensure_ascii=False),
            ),
        )
        self.connection.commit()

    def add_entries(self, entries: Iterable[MemoryEntry]) -> None:
        payload = [
            (
                entry.entry_id,
                entry.run_id,
                entry.created_at,
                entry.category,
                entry.title,
                entry.content,
                json.dumps(entry.tags, ensure_ascii=False),
                entry.importance,
                1 if entry.is_high_value else 0,
                json.dumps(entry.metadata, ensure_ascii=False),
            )
            for entry in entries
        ]
        self.connection.executemany(
            """
            INSERT INTO entries(entry_id, run_id, created_at, category, title, content, tags_json, importance, is_high_value, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        self.connection.commit()

    def recent_entries(
        self,
        run_id: str,
        limit: int,
        categories: Sequence[str] | None = None,
        high_value_only: bool = False,
    ) -> list[dict[str, Any]]:
        clauses = ["run_id = ?"]
        params: list[Any] = [run_id]
        if categories:
            clauses.append(f"category IN ({','.join('?' for _ in categories)})")
            params.extend(categories)
        if high_value_only:
            clauses.append("is_high_value = 1")
        sql = f"""
            SELECT * FROM entries
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC, importance DESC
            LIMIT ?
        """
        params.append(limit)
        rows = self.connection.execute(sql, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def search_entries(
        self,
        query: str,
        limit: int = 20,
        run_id: str | None = None,
        categories: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["(title LIKE ? OR content LIKE ?)"]
        params: list[Any] = [f"%{query}%", f"%{query}%"]
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if categories:
            clauses.append(f"category IN ({','.join('?' for _ in categories)})")
            params.extend(categories)
        sql = f"""
            SELECT * FROM entries
            WHERE {' AND '.join(clauses)}
            ORDER BY is_high_value DESC, importance DESC, created_at DESC
            LIMIT ?
        """
        params.append(limit)
        rows = self.connection.execute(sql, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def increment_token_usage(self, run_id: str, token_count: int) -> dict[str, Any]:
        run = self.get_run(run_id)
        total = int(run["total_tokens"]) + int(token_count)
        self.connection.execute(
            """
            UPDATE runs
            SET total_tokens = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (total, time.time(), run_id),
        )
        self.connection.commit()
        updated = self.get_run(run_id)
        return updated

    def record_checkpoint(
        self,
        run_id: str,
        total_tokens: int,
        checkpoint_type: str,
        summary: dict[str, Any],
    ) -> str:
        checkpoint_id = str(uuid.uuid4())
        self.connection.execute(
            """
            INSERT INTO checkpoints(checkpoint_id, run_id, created_at, total_tokens, checkpoint_type, summary_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                checkpoint_id,
                run_id,
                time.time(),
                total_tokens,
                checkpoint_type,
                json.dumps(summary, ensure_ascii=False),
            ),
        )
        self.connection.commit()
        return checkpoint_id

    def latest_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT * FROM checkpoints
            WHERE run_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["summary"] = json.loads(result.pop("summary_json"))
        return result

    def list_checkpoints(self, run_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT * FROM checkpoints
            WHERE run_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (run_id, limit),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["summary"] = json.loads(item.pop("summary_json"))
            results.append(item)
        return results

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT * FROM runs
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def build_checkpoint_summary(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        high_value = self.recent_entries(run_id, limit=20, high_value_only=True)
        recent_failures = self.recent_entries(run_id, limit=10, categories=("FailedPath",))
        frontiers = self.recent_entries(run_id, limit=10, categories=("SearchFrontier",))
        tool_findings = self.recent_entries(run_id, limit=10, categories=("ToolFinding",))
        return {
            "run_id": run_id,
            "problem": run["problem"],
            "status": run["status"],
            "total_tokens": run["total_tokens"],
            "high_value_findings": high_value,
            "recent_failures": recent_failures,
            "frontiers": frontiers,
            "tool_findings": tool_findings,
        }

    def retarget_checkpoint_thresholds(
        self,
        run_id: str,
        next_checkpoint_at: int,
        next_mini_checkpoint_at: int,
    ) -> None:
        self.connection.execute(
            """
            UPDATE runs
            SET next_checkpoint_at = ?, next_mini_checkpoint_at = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (next_checkpoint_at, next_mini_checkpoint_at, time.time(), run_id),
        )
        self.connection.commit()

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["tags"] = json.loads(data.pop("tags_json"))
        data["metadata"] = json.loads(data.pop("metadata_json"))
        data["is_high_value"] = bool(data["is_high_value"])
        return data

    def close(self) -> None:
        self.connection.close()
