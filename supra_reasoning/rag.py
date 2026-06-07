from __future__ import annotations

from .knowledge import KnowledgeTree, RetrievalHit

RAG_INSTRUCTION = (
    "Use the foundation knowledge below when it clearly applies. "
    "Speak as the priest: weave facts into warm counsel addressed directly to the person. "
    "When natural, ask one follow-up question about their topic to keep the conversation alive. "
    "Never use the words 'user' or 'listener'. Do not describe the question—just answer it. "
    "Do not quote headings, cite sources, or say 'according to the knowledge base'. "
    "Never mention that you looked anything up. If nothing fits, answer from your own wisdom."
)


def build_knowledge_context(hits: list[RetrievalHit]) -> str:
    if not hits:
        return ""

    blocks = [RAG_INSTRUCTION, ""]
    for hit in hits:
        topic = " > ".join(hit.topic_path) if hit.topic_path else "General"
        blocks.append(f"Topic: {topic}")
        blocks.append(f"Question: {hit.entry.question}")
        blocks.append(f"Answer: {hit.entry.answer}")
        blocks.append("")
    return "\n".join(blocks).strip()


def retrieve_knowledge(
    tree: KnowledgeTree,
    question: str,
    *,
    top_k: int = 4,
) -> tuple[str, list[RetrievalHit]]:
    hits = tree.retrieve(question, top_k=top_k)
    return build_knowledge_context(hits), hits
