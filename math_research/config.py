from __future__ import annotations

import os
from dataclasses import dataclass, field


def _load_dotenv() -> None:
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k not in os.environ:
                        os.environ[k.strip()] = v.strip()

@dataclass(slots=True)
class DeepSeekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com/chat/completions"
    model: str = "deepseek-v4-pro"
    temperature_reasoning: float = 0.2
    temperature_exploration: float = 0.55
    temperature_summary: float = 0.15
    max_tokens: int = 8192
    timeout_seconds: int = 180
    stream: bool = True

    @classmethod
    def from_env(cls) -> "DeepSeekConfig":
        _load_dotenv()
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("缺少环境变量 DEEPSEEK_API_KEY。")
        return cls(
            api_key=api_key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions"),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            temperature_reasoning=float(os.environ.get("DEEPSEEK_TEMPERATURE_REASONING", "0.2")),
            temperature_exploration=float(os.environ.get("DEEPSEEK_TEMPERATURE_EXPLORATION", "0.55")),
            temperature_summary=float(os.environ.get("DEEPSEEK_TEMPERATURE_SUMMARY", "0.15")),
            max_tokens=int(os.environ.get("DEEPSEEK_MAX_TOKENS", "4096")),
            timeout_seconds=int(os.environ.get("DEEPSEEK_TIMEOUT_SECONDS", "180")),
            stream=os.environ.get("DEEPSEEK_STREAM", "1") != "0",
        )


@dataclass(slots=True)
class ModelBackendConfig:
    name: str
    api_key: str
    base_url: str
    model: str
    temperature_reasoning: float = 0.2
    temperature_exploration: float = 0.55
    temperature_summary: float = 0.15
    max_tokens: int = 4096
    timeout_seconds: int = 180
    stream: bool = True
    enabled: bool = True
    prefer_for_exploration: bool = False
    prefer_for_summary: bool = False
    prefer_for_reasoning: bool = False

    @classmethod
    def from_env(
        cls,
        prefix: str,
        *,
        default_name: str,
        default_base_url: str,
        default_model: str,
        required_api_key: bool,
    ) -> "ModelBackendConfig | None":
        _load_dotenv()
        api_key = os.environ.get(f"{prefix}_API_KEY", "").strip()
        if required_api_key and not api_key:
            return None
        base_url = os.environ.get(f"{prefix}_BASE_URL", default_base_url).strip()
        model = os.environ.get(f"{prefix}_MODEL", default_model).strip()
        enabled = os.environ.get(f"{prefix}_ENABLED", "1") != "0"
        if not enabled:
            return None
        if not api_key and not required_api_key:
            api_key = os.environ.get(f"{prefix}_API_KEY", "local-placeholder").strip() or "local-placeholder"
        return cls(
            name=os.environ.get(f"{prefix}_NAME", default_name).strip() or default_name,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature_reasoning=float(os.environ.get(f"{prefix}_TEMPERATURE_REASONING", "0.2")),
            temperature_exploration=float(os.environ.get(f"{prefix}_TEMPERATURE_EXPLORATION", "0.55")),
            temperature_summary=float(os.environ.get(f"{prefix}_TEMPERATURE_SUMMARY", "0.15")),
            max_tokens=int(os.environ.get(f"{prefix}_MAX_TOKENS", "4096")),
            timeout_seconds=int(os.environ.get(f"{prefix}_TIMEOUT_SECONDS", "180")),
            stream=os.environ.get(f"{prefix}_STREAM", "1") != "0",
            prefer_for_exploration=os.environ.get(f"{prefix}_PREFER_EXPLORATION", "0") == "1",
            prefer_for_summary=os.environ.get(f"{prefix}_PREFER_SUMMARY", "0") == "1",
            prefer_for_reasoning=os.environ.get(f"{prefix}_PREFER_REASONING", "0") == "1",
        )


@dataclass(slots=True)
class RuntimeConfig:
    database_path: str = "research_memory.db"
    archive_limit: int = 50
    high_value_limit: int = 20
    reverse_search_limit: int = 20
    checkpoint_tokens: int = 100_000_000
    mini_checkpoint_tokens: int = 10_000_000
    max_iterations_per_run: int = 20
    stream: bool = True
    stable_system_prompt_version: str = "v1"
    batch_workers: int = 1
    max_cpu_percent_for_local: float = 75.0
    min_free_memory_gb_for_local: float = 8.0
    enable_local_model: bool = True
    auto_pause_on_local_overload: bool = True
    use_sympy_tools: bool = True
    failure_categories: tuple[str, ...] = field(
        default_factory=lambda: (
            "contradiction",
            "non_rigorous_leap",
            "tool_disproved",
            "redundant_with_known_theorem",
            "search_space_explosion",
            "local_obstruction_found",
            "insufficient_generality",
            "parser_error",
            "runtime_error",
        )
    )

    @classmethod
    def from_env(cls, database_path: str = "research_memory.db") -> "RuntimeConfig":
        return cls(
            database_path=database_path,
            archive_limit=int(os.environ.get("RESEARCH_ARCHIVE_LIMIT", "50")),
            high_value_limit=int(os.environ.get("RESEARCH_HIGH_VALUE_LIMIT", "20")),
            reverse_search_limit=int(os.environ.get("RESEARCH_REVERSE_SEARCH_LIMIT", "20")),
            checkpoint_tokens=int(os.environ.get("RESEARCH_CHECKPOINT_TOKENS", "100000000")),
            mini_checkpoint_tokens=int(os.environ.get("RESEARCH_MINI_CHECKPOINT_TOKENS", "10000000")),
            max_iterations_per_run=int(os.environ.get("RESEARCH_MAX_ITERATIONS", "20")),
            stream=os.environ.get("RESEARCH_STREAM", "1") != "0",
            stable_system_prompt_version=os.environ.get("RESEARCH_SYSTEM_PROMPT_VERSION", "v1"),
            batch_workers=int(os.environ.get("RESEARCH_BATCH_WORKERS", "1")),
            max_cpu_percent_for_local=float(os.environ.get("RESEARCH_MAX_CPU_FOR_LOCAL", "75")),
            min_free_memory_gb_for_local=float(os.environ.get("RESEARCH_MIN_FREE_MEMORY_GB", "8")),
            enable_local_model=os.environ.get("RESEARCH_ENABLE_LOCAL_MODEL", "1") != "0",
            auto_pause_on_local_overload=os.environ.get("RESEARCH_AUTO_PAUSE_LOCAL_OVERLOAD", "1") != "0",
            use_sympy_tools=os.environ.get("RESEARCH_USE_SYMPY_TOOLS", "1") != "0",
        )
