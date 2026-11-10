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


@dataclass
class AgentResponse:
    answer: str
    steps: list[RouteResult] = field(default_factory=list)


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


class Agent:
    def __init__(
        self,
        llm,
        tools: dict,
        system_prompt: str,
        max_steps: int = 5,
        max_history_turns: int = 6,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.conversation = Conversation(
            system_prompt,
            max_turns=max_history_turns,
        )

    def ask(self, question: str) -> AgentResponse:
        self.conversation.add_user(question)
        scratchpad = self.conversation.messages()
        steps: list[RouteResult] = []
        answer = None

        try:
            for _ in range(self.max_steps):
                llm_output = self.llm.chat(scratchpad, json_mode=True)
                result = route_query(llm_output, self.tools)
                steps.append(result)
                if result.kind == "final":
                    answer = result.content
                    break

                scratchpad.append({"role": "assistant", "content": llm_output})
                if result.kind == "error":
                    observation = f"ERREUR DE FORMAT :\n{result.content}"
                else:
                    observation = (
                        f"OBSERVATION ({result.tool}) :\n{result.content}"
                    )
                scratchpad.append({"role": "user", "content": observation})

            if answer is None:
                scratchpad.append(
                    {
                        "role": "user",
                        "content": (
                            "La limite d'étapes est atteinte. Donnez maintenant "
                            "une réponse finale avec l'action final_answer, sans "
                            "appeler d'autre outil."
                        ),
                    }
                )
                llm_output = self.llm.chat(scratchpad, json_mode=True)
                result = route_query(llm_output, self.tools)
                steps.append(result)
                if result.kind == "final":
                    answer = result.content
                else:
                    answer = (
                        "Désolé, je n'ai pas pu produire une réponse finale "
                        "dans le nombre d'étapes autorisé."
                    )
        except LLMError as exc:
            logger.warning("Le service LLM est indisponible : %s", exc)
            answer = f"Le service LLM est indisponible. {exc}"

        self.conversation.add_assistant(answer)
        return AgentResponse(answer=answer, steps=steps)

    def reset(self) -> None:
        self.conversation.reset()


def build_agent():
    import config
    from llm_client import LLMClient
    from prompts import build_system_prompt
    from tools import DocSearchTool, SQLTool, load_corpus, load_embedder

    llm = LLMClient(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_URL,
        temperature=config.LLM_TEMPERATURE,
        timeout=config.LLM_TIMEOUT,
        num_ctx=config.LLM_NUM_CTX,
        log_file=config.LOG_DIR / "llm_exchanges.jsonl",
    )
    sql_tool = SQLTool(config.DB_PATH, max_rows=config.SQL_MAX_ROWS)
    chunks = load_corpus(
        config.DOCS_DIR,
        chunk_size=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP,
    )
    doc_tool = DocSearchTool(
        chunks,
        load_embedder(config.EMBEDDING_MODEL),
        top_k=config.TOP_K,
    )
    tools = {
        sql_tool.name: sql_tool,
        doc_tool.name: doc_tool,
    }
    system_prompt = build_system_prompt(sql_tool.get_schema())
    agent = Agent(
        llm,
        tools,
        system_prompt,
        max_steps=config.MAX_STEPS,
        max_history_turns=config.MAX_HISTORY_TURNS,
    )
    return agent, llm


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent, _llm = build_agent()
    print("DataPilot prêt. Tapez « quit » pour quitter ou « reset » pour vider la mémoire.")
    while True:
        try:
            question = input("\nVous : ").strip()
        except EOFError:
            break
        if question.lower() in {"quit", "exit"}:
            break
        if question.lower() == "reset":
            agent.reset()
            print("Mémoire vidée.")
            continue
        if not question:
            continue

        response = agent.ask(question)
        for index, step in enumerate(response.steps, start=1):
            step_input = (
                step.tool_input if step.tool_input is not None else step.content
            )
            tool_input = str(step_input).replace("\n", " ")
            print(
                f"Étape {index} — {step.kind}"
                f"{f' ({step.tool})' if step.tool else ''}: "
                f"{tool_input[:120]}"
            )
        print(f"DataPilot : {response.answer}")
