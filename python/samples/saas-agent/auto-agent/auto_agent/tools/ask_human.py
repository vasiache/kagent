"""AskHumanTool — Human-in-the-Loop via LangGraph interrupt.

Pauses the graph and waits for human input.
Use before ANY create / update / delete action.

How it works:
1. Agent calls ask_human(question, options)
2. LangGraph interrupt() suspends the graph — state is checkpointed
3. TG Bot receives the interrupt payload → sends inline keyboard to user
4. User taps button → TG Bot calls agent.resume(answer)
5. Graph continues with the human's answer
"""
from __future__ import annotations

from langchain_core.tools import tool
from langgraph.types import interrupt


@tool
def ask_human(question: str, options: list[str] | None = None) -> str:
    """Ask the user a question and wait for their answer before continuing.

    ALWAYS call this before any action that creates, updates, or deletes data.
    The conversation will pause until the user responds.

    Args:
        question: clear description of what you are about to do and why
        options:  list of suggested responses (e.g. ["Да", "Нет"]).
                  If None, user can type a free-form answer.
    """
    payload: dict = {"question": question}
    if options:
        payload["options"] = options

    # interrupt() suspends the LangGraph graph execution.
    # The return value here is what the human typed / selected.
    answer = interrupt(payload)
    return str(answer)
