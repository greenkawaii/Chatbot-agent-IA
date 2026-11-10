"""Interface Gradio de DataPilot (couche présentation uniquement)."""
from __future__ import annotations

import logging

import gradio as gr

TITLE = "DataPilot — l'assistant data analyst de ShopNova"
DESCRIPTION = (
    "Interrogez les données commerciales de ShopNova ou ses documents métier "
    "(ventes, clients, retours et indicateurs)."
)
EXAMPLES = [
    "Quel est le chiffre d'affaires net de 2025 ?",
    "Comment est défini le taux de retour ?",
]
TOOL_ICONS = {
    "query_database": "🗄 Requête SQL",
    "search_docs": "📚 Recherche documentaire",
}


def message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item["text"]
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return str(content)


def steps_to_messages(steps) -> list[dict]:
    messages = []
    for step in steps:
        if step.kind == "tool":
            title = f"🔧 {TOOL_ICONS.get(step.tool, step.tool or 'Outil')}"
            content = (
                f"**Entrée :**\n```text\n{step.tool_input or ''}\n```\n\n"
                f"**Résultat :**\n{message_text(step.content)[:1500]}"
            )
        elif step.kind == "error":
            title = "⚠ Erreur"
            content = message_text(step.content)[:1500]
        else:
            continue

        messages.append(
            {
                "role": "assistant",
                "content": content,
                "metadata": {"title": title, "status": "done"},
            }
        )
    return messages


def create_ui(agent) -> gr.Blocks:
    def add_user_message(message: str, history: list):
        if not message.strip():
            return "", history
        return "", history + [{"role": "user", "content": message}]

    def bot_response(history: list):
        if not history or history[-1].get("role") != "user":
            return history

        question = message_text(history[-1].get("content"))
        response = agent.ask(question)
        return history + steps_to_messages(response.steps) + [
            {"role": "assistant", "content": response.answer}
        ]

    def reset_conversation():
        agent.reset()
        return [], ""

    with gr.Blocks(title="DataPilot") as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)
        chatbot = gr.Chatbot(
            label="Conversation",
            height=520,
            placeholder="Posez une question ou cliquez sur un exemple 👇",
        )
        with gr.Row():
            question = gr.Textbox(
                show_label=False,
                scale=8,
                placeholder="Votre question...",
            )
            send_btn = gr.Button("Envoyer", variant="primary", scale=1)
            reset_btn = gr.Button("🔄 Réinitialiser", scale=1)
        gr.Examples(
            examples=EXAMPLES,
            inputs=question,
            label="Exemples de questions",
        )

        for trigger in (question.submit, send_btn.click):
            trigger(
                add_user_message,
                [question, chatbot],
                [question, chatbot],
                queue=False,
            ).then(
                bot_response,
                chatbot,
                chatbot,
                show_progress="full",
            )

        reset_btn.click(
            reset_conversation,
            outputs=[chatbot, question],
            queue=False,
        )
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from agent import build_agent

    agent, llm = build_agent()
    if not llm.is_available():
        print("⚠ Ollama ne répond pas : lancez l'application Ollama ou 'ollama serve'.")
    create_ui(agent).launch(theme=gr.themes.Soft())
