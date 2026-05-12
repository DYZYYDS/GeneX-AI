from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any

import psutil


@dataclass(slots=True)
class ResourceSnapshot:
    cpu_percent: float
    logical_cpus: int
    physical_cpus: int
    total_memory_gb: float
    available_memory_gb: float
    gpu_name: str | None = None
    gpu_memory_total_mb: int | None = None
    gpu_memory_free_mb: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "logical_cpus": self.logical_cpus,
            "physical_cpus": self.physical_cpus,
            "total_memory_gb": round(self.total_memory_gb, 2),
            "available_memory_gb": round(self.available_memory_gb, 2),
            "gpu_name": self.gpu_name,
            "gpu_memory_total_mb": self.gpu_memory_total_mb,
            "gpu_memory_free_mb": self.gpu_memory_free_mb,
        }


class ResourceManager:
    def snapshot(self) -> ResourceSnapshot:
        memory = psutil.virtual_memory()
        gpu_name = None
        gpu_total = None
        gpu_free = None
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if completed.returncode == 0 and completed.stdout.strip():
                first = completed.stdout.strip().splitlines()[0]
                parts = [item.strip() for item in first.split(",")]
                if len(parts) >= 3:
                    gpu_name = parts[0]
                    gpu_total = int(parts[1])
                    gpu_free = int(parts[2])
        except Exception:  # noqa: BLE001
            pass
        return ResourceSnapshot(
            cpu_percent=float(psutil.cpu_percent(interval=0.1)),
            logical_cpus=psutil.cpu_count(logical=True) or os.cpu_count() or 1,
            physical_cpus=psutil.cpu_count(logical=False) or max(1, (os.cpu_count() or 1) // 2),
            total_memory_gb=memory.total / (1024**3),
            available_memory_gb=memory.available / (1024**3),
            gpu_name=gpu_name,
            gpu_memory_total_mb=gpu_total,
            gpu_memory_free_mb=gpu_free,
        )

    def recommend_batch_workers(self) -> int:
        snap = self.snapshot()
        by_cpu = max(1, snap.physical_cpus - 1)
        by_mem = max(1, int(snap.available_memory_gb // 6))
        return max(1, min(by_cpu, by_mem))

    def should_use_local_model(self, *, max_cpu_percent: float, min_free_memory_gb: float) -> bool:
        snap = self.snapshot()
        enough_gpu = snap.gpu_memory_free_mb is None or snap.gpu_memory_free_mb >= 4096
        return snap.cpu_percent <= max_cpu_percent and snap.available_memory_gb >= min_free_memory_gb and enough_gpu
