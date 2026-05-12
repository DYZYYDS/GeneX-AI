from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import DeepSeekConfig, ModelBackendConfig


@dataclass(slots=True)
class ChatResult:
    text: str
    usage: dict[str, Any]
    raw_chunks: list[dict[str, Any]] = field(default_factory=list)


class OpenAICompatibleClient:
    def __init__(self, config: ModelBackendConfig) -> None:
        self.config = config
        self._ssl_context = ssl.create_default_context()

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        stream_callback: Callable[[str], None] | None = None,
    ) -> ChatResult:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.config.max_tokens,
            "stream": self.config.stream,
        }
        if self.config.stream:
            payload["stream_options"] = {"include_usage": True}
        request = urllib.request.Request(
            self.config.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(
            request,
            timeout=self.config.timeout_seconds,
            context=self._ssl_context,
        ) as response:
            if not self.config.stream:
                raw = json.loads(response.read().decode("utf-8"))
                choice = raw["choices"][0]["message"]["content"]
                usage = raw.get("usage", {})
                return ChatResult(text=choice, usage=usage, raw_chunks=[raw])
            return self._consume_stream(response, stream_callback=stream_callback)

    def probe(self) -> dict[str, Any]:
        models_url = self._models_url()
        request = urllib.request.Request(
            models_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=min(self.config.timeout_seconds, 20),
                context=self._ssl_context,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return {
                    "ok": True,
                    "backend": self.config.name,
                    "models_url": models_url,
                    "model_count": len(payload.get("data", [])),
                }
        except urllib.error.URLError as exc:
            return {
                "ok": False,
                "backend": self.config.name,
                "models_url": models_url,
                "error": str(exc),
            }

    def _models_url(self) -> str:
        if self.config.base_url.endswith("/chat/completions"):
            return self.config.base_url[: -len("/chat/completions")] + "/models"
        return self.config.base_url.rstrip("/") + "/models"

    def _consume_stream(
        self,
        response: Any,
        stream_callback: Callable[[str], None] | None = None,
    ) -> ChatResult:
        fragments: list[str] = []
        raw_chunks: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            raw_chunks.append(chunk)
            if "usage" in chunk and chunk["usage"]:
                usage = chunk["usage"]
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            piece = delta.get("content", "")
            if piece:
                fragments.append(piece)
                if stream_callback:
                    stream_callback(piece)
        return ChatResult(text="".join(fragments), usage=usage, raw_chunks=raw_chunks)


class DeepSeekClient(OpenAICompatibleClient):
    def __init__(self, config: DeepSeekConfig) -> None:
        base_url = config.base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url + "/chat/completions"
        elif not base_url.endswith("/chat/completions"):
            base_url = f"{base_url}/v1/chat/completions"

        backend = ModelBackendConfig(
            name="deepseek",
            api_key=config.api_key,
            base_url=base_url,
            model=config.model,
            temperature_reasoning=config.temperature_reasoning,
            temperature_exploration=config.temperature_exploration,
            temperature_summary=config.temperature_summary,
            max_tokens=config.max_tokens,
            timeout_seconds=config.timeout_seconds,
            stream=config.stream,
            prefer_for_reasoning=True,
            prefer_for_summary=True,
        )
        super().__init__(backend)
