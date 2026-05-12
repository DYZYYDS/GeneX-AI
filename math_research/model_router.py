from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .config import DeepSeekConfig, ModelBackendConfig, RuntimeConfig
from .deepseek_client import ChatResult, DeepSeekClient, OpenAICompatibleClient
from .resource_manager import ResourceManager


@dataclass(slots=True)
class RoutedResponse:
    backend_name: str
    task_type: str
    result: ChatResult


class ModelRouter:
    def __init__(
        self,
        *,
        deepseek_config: DeepSeekConfig,
        runtime_config: RuntimeConfig,
        local_backend: ModelBackendConfig | None = None,
        resource_manager: ResourceManager | None = None,
    ) -> None:
        self.runtime_config = runtime_config
        self.deepseek_config = deepseek_config
        self.resource_manager = resource_manager or ResourceManager()
        self.deepseek_client = DeepSeekClient(deepseek_config)
        self.local_backend = local_backend
        self.local_client = OpenAICompatibleClient(local_backend) if local_backend else None

    def backend_status(self) -> dict[str, Any]:
        resource = self.resource_manager.snapshot().to_dict()
        backends: list[dict[str, Any]] = [self.deepseek_client.probe()]
        if self.local_client is not None:
            local_probe = self.local_client.probe()
            local_probe["eligible_now"] = self._should_use_local("exploration")
            backends.append(local_probe)
        return {
            "resource": resource,
            "routing_preview": {
                "reasoning": self._pick_backend_name("reasoning"),
                "exploration": self._pick_backend_name("exploration"),
                "summary": self._pick_backend_name("summary"),
            },
            "backends": backends,
        }

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        task_type: str,
        stream_callback: Callable[[str], None] | None = None,
    ) -> RoutedResponse:
        backend_name, client, temperature = self._pick_backend(task_type)
        try:
            result = client.chat(messages=messages, temperature=temperature, stream_callback=stream_callback)
            return RoutedResponse(backend_name=backend_name, task_type=task_type, result=result)
        except Exception:  # noqa: BLE001
            if backend_name != "deepseek":
                fallback = self.deepseek_client.chat(
                    messages=messages,
                    temperature=self._temperature_for_deepseek(task_type),
                    stream_callback=stream_callback,
                )
                return RoutedResponse(backend_name="deepseek_fallback", task_type=task_type, result=fallback)
            raise

    def _pick_backend(self, task_type: str) -> tuple[str, DeepSeekClient | OpenAICompatibleClient, float]:
        if self._should_use_local(task_type):
            assert self.local_backend is not None
            assert self.local_client is not None
            return self.local_backend.name, self.local_client, self._temperature_for(self.local_backend, task_type)
        return "deepseek", self.deepseek_client, self._temperature_for_deepseek(task_type)

    def _pick_backend_name(self, task_type: str) -> str:
        if self._should_use_local(task_type) and self.local_backend is not None:
            return self.local_backend.name
        return "deepseek"

    def _should_use_local(self, task_type: str) -> bool:
        if not self.runtime_config.enable_local_model or self.local_backend is None or self.local_client is None:
            return False
        if not self.resource_manager.should_use_local_model(
            max_cpu_percent=self.runtime_config.max_cpu_percent_for_local,
            min_free_memory_gb=self.runtime_config.min_free_memory_gb_for_local,
        ):
            return False
        if task_type == "exploration":
            return True
        if task_type == "archive" and self.local_backend.prefer_for_exploration:
            return True
        if task_type == "summary" and self.local_backend.prefer_for_summary:
            return True
        if task_type == "reasoning" and self.local_backend.prefer_for_reasoning:
            return True
        return False

    def _temperature_for_deepseek(self, task_type: str) -> float:
        return self._temperature_for(
            ModelBackendConfig(
                name="deepseek",
                api_key=self.deepseek_config.api_key,
                base_url=self.deepseek_config.base_url,
                model=self.deepseek_config.model,
                temperature_reasoning=self.deepseek_config.temperature_reasoning,
                temperature_exploration=self.deepseek_config.temperature_exploration,
                temperature_summary=self.deepseek_config.temperature_summary,
                max_tokens=self.deepseek_config.max_tokens,
                timeout_seconds=self.deepseek_config.timeout_seconds,
                stream=self.deepseek_config.stream,
            ),
            task_type,
        )

    def _temperature_for(self, config: ModelBackendConfig, task_type: str) -> float:
        if task_type == "summary":
            return config.temperature_summary
        if task_type == "exploration":
            return config.temperature_exploration
        return config.temperature_reasoning
