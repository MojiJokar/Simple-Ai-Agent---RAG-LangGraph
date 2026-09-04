
"""
Chat Interface — Gradio

The path taken by the agent is shown alongside each answer.
This is the same "observability" concept from Station Twelve:
without seeing the steps, debugging is impossible.

The conversation history is also passed to the agent so that
the second question can understand what the first question was about.

Run:
    python app.py
"""

import gradio as gr
from agent import ask

CSS = """
.rtl textarea, .rtl input { direction: rtl; text-align: right; }
.message { direction: rtl; text-align: right; }
footer { display: none !important; }
"""

EXAMPLES = [
    "What is chunking?",
    "What is a vector database and why is it needed?",
    "What does context window mean?",
    "What is prompt injection?",
    "What is LangGraph?",
    "What is today's dollar exchange rate?",
]


def _tabdil(tarikhche):
    """
    Gradio provides conversation history in its own format.
    We only need the role and content.
    """
    out = []

    if not tarikhche:
        return out

    for m in tarikhche:
        if isinstance(m, dict) and m.get("content"):
            out.append({
                "role": m.get("role", "user"),
                "content": str(m["content"])
            })

        elif isinstance(m, (list, tuple)) and len(m) == 2:
            if m[0]:
                out.append({
                    "role": "user",
                    "content": str(m[0])
                })

            if m[1]:
                out.append({
                    "role": "assistant",
                    "content": str(m[1])
                })

    return out


def pasokh(payam, tarikhche):
    try:
        javab, masir = ask(payam, _tabdil(tarikhche))

    except Exception as e:
        return f"Error: {type(e).__name__} — {e}"

    khat = "\n".join(
        f"{i + 1}. {s}"
        for i, s in enumerate(masir)
    )

    return (
        f"{javab}\n\n"
        f"---\n"
        f"**Agent Path:**\n\n"
        f"{khat}"
    )


with gr.Blocks(title="Moji Assistant") as demo:

    gr.Markdown(
        "## Moji Assistant\n"
        "Ask me anything that was covered in the channel's videos."
    )

    gr.ChatInterface(
        fn=pasokh,
        examples=EXAMPLES,
        textbox=gr.Textbox(
            placeholder="Type your question...",
            elem_classes="rtl",
        ),
    )


if __name__ == "__main__":
    demo.launch(css=CSS, inbrowser=True)

