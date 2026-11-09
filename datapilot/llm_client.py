"""llm_client.py — Client LLM robuste en Python pur (bibliothèque requests)."""
from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import requests

logger = logging.getLogger("llm_client")


class LLMError(Exception):
    """Erreur définitive du client LLM (après épuisement des tentatives)."""


class LLMClient:
    RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

    def __init__(
        self,
        model,
        base_url="http://localhost:11434",
        temperature=0.1,
        timeout=120,
        max_retries=3,
        backoff_factor=1.0,
        num_ctx=8192,
        log_file=None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.url = self.base_url + "/api/chat"
        # TODO : mémoriser les autres paramètres dans des attributs
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.num_ctx = num_ctx
        self.log_file = Path(log_file) if log_file else None
        self.session = requests.Session()  # réutilise la connexion entre les appels

    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }  # TODO : construire le corps de la requête (voir le contrat)
        if json_mode:
            payload["format"] = "json"

        start = time.perf_counter()
        data = self._post_with_retry(payload)
        duration = time.perf_counter() - start
        # TODO : extraire le texte ; si la structure est inattendue, lever LLMError
        try:
            answer = data["message"]["content"]
            if not isinstance(answer, str):
                raise TypeError("le contenu du message n'est pas du texte")
        except (KeyError, TypeError) as exc:
            raise LLMError("Réponse Ollama inattendue : texte du message absent ou invalide") from exc

        # TODO : appeler self._log_exchange(...) puis renvoyer le texte
        self._log_exchange(messages, answer, duration, data)
        return answer

    def _post_with_retry(self, payload: dict) -> dict:
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(self.url, json=payload, timeout=self.timeout)
                if response.status_code in self.RETRYABLE_STATUS:
                    # TODO 1 : code réessayable -> lever requests.HTTPError(..., response=response)
                    raise requests.HTTPError(
                        f"HTTP {response.status_code}", response=response
                    )
                if response.status_code >= 400:
                    # TODO 2 : autre code >= 400 -> lever LLMError avec le code dans le message
                    raise LLMError(f"Erreur HTTP {response.status_code} du serveur Ollama")
                try:
                    return response.json()
                except ValueError as exc:
                    raise LLMError("Réponse JSON invalide reçue du serveur Ollama") from exc
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                last_error = exc
            # TODO 3 : si ce n'est pas le dernier essai, journaliser un avertissement
            #          puis attendre self.backoff_factor * 2 ** (attempt - 1) secondes
            if attempt < self.max_retries:
                logger.warning(
                    "Échec de l'appel Ollama (%s), nouvelle tentative %s/%s",
                    last_error,
                    attempt + 1,
                    self.max_retries,
                )
                delay = self.backoff_factor * 2 ** (attempt - 1)
                jitter = random.uniform(0, self.backoff_factor)
                time.sleep(delay + jitter)
        # TODO 4 : message qui contient l'erreur ET « ollama serve »
        raise LLMError(
            f"Échec de l'appel à Ollama : {last_error}. "
            "Vérifiez qu'Ollama est lancé (ollama serve)."
        ) from last_error

    def _log_exchange(self, messages, answer, duration, raw):
        # TODO : construire un dictionnaire (timestamp, model, duration_s,
        #        prompt_tokens, completion_tokens, messages, answer) ;
        #        si self.log_file : créer le dossier parent si besoin, ouvrir le
        #        fichier en mode "a" (ajout) et encodage utf-8, écrire json.dumps(...,
        #        ensure_ascii=False) + "\n"
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": self.model,
            "duration_s": duration,
            "prompt_tokens": raw.get("prompt_eval_count"),
            "completion_tokens": raw.get("eval_count"),
            "messages": messages,
            "answer": answer,
        }
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.log_file.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def is_available(self) -> bool:
        # TODO : GET sur /api/tags avec un timeout court ; True si r.ok, False en cas
        # d'exception
        try:
            response = self.session.get(
                self.base_url + "/api/tags",
                timeout=min(self.timeout, 5),
            )
            return response.ok
        except requests.RequestException:
            return False


class Conversation:
    def __init__(self, system_prompt: str, max_turns: int = 6):
        if max_turns < 0:
            raise ValueError("max_turns doit être positif ou nul")
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.history: list[dict[str, str]] = []

    def add_user(self, content: str) -> None:
        limit = 2 * self.max_turns
        if limit == 0:
            self.history.clear()
            return
        if len(self.history) >= limit:
            del self.history[:2]
        self.history.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        limit = 2 * self.max_turns
        if limit == 0:
            self.history.clear()
            return
        self.history.append({"role": "assistant", "content": content})
        if len(self.history) > limit:
            del self.history[: len(self.history) - limit]

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            *(
                {"role": message["role"], "content": message["content"]}
                for message in self.history[-2 * self.max_turns :]
            ),
        ]

    def estimated_tokens(self) -> int:
        return sum(len(message["content"]) for message in self.messages()) // 4

    def reset(self) -> None:
        self.history.clear()
