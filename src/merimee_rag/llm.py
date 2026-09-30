"""Client Ollama minimal (API HTTP locale) avec cache disque.

Le cache (JSONL, clé = hash(modèle, messages, options)) rend les évaluations
reprenables : si une exécution de 2 h plante à 80 %, on relance sans tout refaire.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any, Protocol


class LLM(Protocol):
    model: str

    def chat(self, messages: list[dict], json_mode: bool = False, **opts: Any) -> str: ...


class DiskCache:
    def __init__(self, path: Path | None) -> None:
        self.path, self.data, self.lock = path, {}, threading.Lock()
        if path and path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.data[rec["k"]] = rec["v"]
                    except (json.JSONDecodeError, KeyError):
                        continue  # ligne tronquée par un crash : on l'ignore

    @staticmethod
    def key(*parts: Any) -> str:
        return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def get(self, k: str) -> str | None:
        return self.data.get(k)

    def put(self, k: str, v: str) -> None:
        with self.lock:
            self.data[k] = v
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")


class OllamaLLM:
    def __init__(self, model: str, url: str = "http://localhost:11434", temperature: float = 0.0,
                 cache: DiskCache | None = None, num_ctx: int = 8192, timeout: int = 600) -> None:
        self.model, self.url, self.temperature = model, url.rstrip("/"), temperature
        self.cache, self.num_ctx, self.timeout = cache, num_ctx, timeout
        self.last_from_cache = False  # utile pour ne pas afficher une fausse latence

    def chat(self, messages: list[dict], json_mode: bool = False, **opts: Any) -> str:
        options = {"temperature": self.temperature, "num_ctx": self.num_ctx, "seed": 42, **opts}
        k = DiskCache.key(self.model, messages, options, json_mode)
        if self.cache and (hit := self.cache.get(k)) is not None:
            self.last_from_cache = True
            return hit
        self.last_from_cache = False
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False, "options": options}
        if json_mode:
            payload["format"] = "json"
        out = self._post_with_retry(payload)["content"]
        if self.cache:
            self.cache.put(k, out)
        return out

    def chat_message(self, messages: list[dict], tools: list[dict] | None = None, **opts: Any) -> dict:
        """Appel avec outils (tool calling). Renvoie le message complet de l'assistant :
        {"role": "assistant", "content": ..., "tool_calls": [{"function": {"name", "arguments"}}]}."""
        options = {"temperature": self.temperature, "num_ctx": self.num_ctx, "seed": 42, **opts}
        k = DiskCache.key("tools", self.model, messages, options, tools)
        if self.cache and (hit := self.cache.get(k)) is not None:
            self.last_from_cache = True
            return json.loads(hit)
        self.last_from_cache = False
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False, "options": options}
        if tools:
            payload["tools"] = tools
        msg = self._post_with_retry(payload)
        msg = {"role": "assistant", "content": msg.get("content") or "", "tool_calls": msg.get("tool_calls") or []}
        if self.cache:
            self.cache.put(k, json.dumps(msg, ensure_ascii=False))
        return msg

    def _post_with_retry(self, payload: dict, attempts: int = 5) -> dict:
        """Réessaie sur erreur serveur (500, runner Ollama qui plante en changeant de modèle…)
        ou connexion coupée : 5 s, 10 s, 20 s, 40 s d'attente entre les essais."""
        import time

        import requests

        for i in range(attempts):
            try:
                r = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
                if r.status_code == 404:
                    raise RuntimeError(f"Modèle Ollama '{self.model}' absent. Lance : ollama pull {self.model}")
                if r.status_code < 500:
                    r.raise_for_status()
                    return r.json()["message"]
                err = f"HTTP {r.status_code} : {r.text[:200]}"
            except (requests.ConnectionError, requests.Timeout) as e:
                err = f"{type(e).__name__}"
            if i < attempts - 1:
                wait = 5 * 2 ** i
                print(f"\n  ⚠ Ollama ({self.model}) : {err} — nouvel essai dans {wait} s", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"Ollama ({self.model}) ne répond plus après {attempts} essais : {err}")


def parse_json(text: str) -> Any:
    """Parse tolérant : les petits modèles entourent parfois le JSON de texte ou de ```."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}|\[.*\]", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None
