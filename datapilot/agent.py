"""agent.py — Routing et boucle de l'agent DataPilot."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from llm_client import Conversation, LLMError

logger = logging.getLogger("agent")
FINAL = "final_answer"


@dataclass
class Decision:
    action: str
    input: object
    thought: str = ""
    params: list = field(default_factory=list)


class ParseError(ValueError):
    """La réponse du modèle ne contient pas une décision JSON exploitable."""


def parse_llm_output(raw: str) -> Decision:
    if not isinstance(raw, str):
        raise ParseError("La réponse du modèle doit être du texte.")

    text = re.sub(
        r"```(?:json)?\s*|\s*```",
        "",
        raw.strip(),
        flags=re.IGNORECASE,
    )
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ParseError("Aucun objet JSON trouvé dans la réponse du modèle.")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ParseError(f"JSON invalide dans la réponse du modèle : {exc}") from exc

    if not isinstance(data, dict):
        raise ParseError("La réponse JSON doit être un objet.")
    if "action" not in data:
        raise ParseError("La clé 'action' est absente de la réponse JSON.")

    action = data["action"]
    if not isinstance(action, str):
        raise ParseError("La valeur de 'action' doit être du texte.")

    return Decision(
        action=action.strip().lower(),
        input=data.get("input", data.get("action_input", "")),
        thought=str(data.get("thought", "")),
        params=data.get("params", []),
    )


@dataclass
class RouteResult:
    kind: str
    content: str
    tool: str | None = None
    tool_input: object = None
    thought: str = ""


def route_query(llm_output: str, tools: dict) -> RouteResult:
    try:
        decision = parse_llm_output(llm_output)
    except ParseError as exc:
        return RouteResult(
            kind="error",
            content=(
                f"Réponse JSON illisible ({exc}). "
                "Répondez uniquement avec un objet JSON valide contenant "
                "les champs action et input."
            ),
        )

    if decision.action == FINAL:
        return RouteResult(
            kind="final",
            content=str(decision.input),
            thought=decision.thought,
        )

    if decision.action not in tools:
        valid_tools = ", ".join(sorted(tools)) or "(aucun)"
        return RouteResult(
            kind="error",
            content=f"Outil inconnu « {decision.action} ». Outils valides : {valid_tools}.",
            thought=decision.thought,
        )

    tool = tools[decision.action]
    tool_input = str(decision.input)
    if decision.action == "query_database" and decision.params:
        content = tool.run(tool_input, decision.params)
    else:
        content = tool.run(tool_input)

    return RouteResult(
        kind="tool",
        content=str(content),
        tool=decision.action,
        tool_input=decision.input,
        thought=decision.thought,
    )
