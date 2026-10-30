"""The RAG pipeline: understand the question, retrieve/compute context, then generate a grounded answer."""

from . import config, llm
from .retrieve import build_context, explain
from .store import get_pokedex

SYSTEM_PROMPT = """You are PokéRAG, a Pokémon expert assistant for stats, regions, lore, team building and counters.

Rules:
- Answer ONLY from the CONTEXT provided with each question. If the context does not contain the answer, say you don't know rather than guessing.
- Sections marked ANALYSIS or RANKING were computed exactly by a type-chart engine or database query. Trust their numbers, rankings and recommendations; your job is to explain them clearly, not to recompute them.
- For a RANKING, entry #1 is the answer to "which is the fastest/strongest/...". Keep the ranking's order; note ties.
- When recommending counters or teammates, give the reason for each pick (type advantage, resistances, speed).
- Cite sources in square brackets after the facts they support, using the bracketed names in the context, e.g. [Garchomp] or [Intimidate (ability)].
- Be concise. Prefer short bullet lists for recommendations. Don't repeat the whole context."""


def history_pokemon(history):
    dex = get_pokedex()
    for msg in reversed(history or []):
        if msg.get("role") == "user":
            names = [n for n, _ in dex.find_in_text(msg.get("content", ""))]
            if names:
                return names
    return None


def compact_analysis(ctx):
    """The parts of the analysis the UI renders as cards/tables."""
    a = ctx.analysis
    if not a:
        return None
    if ctx.analysis_kind == "counters":
        return {k: a[k] for k in ("per_opponent", "team_wide", "best_attacking_types", "my_team_plan", "no_safe_answer") if k in a}
    an = a["analysis"]
    return {"team": an["team"], "shared_weaknesses": an["shared_weaknesses"], "coverage_gaps": an["coverage_gaps"],
            "unresisted_types": an["unresisted_types"], "suggestions": a.get("suggestions", []),
            "built": a.get("built")}


def answer(question, history=None):
    """Yield events: meta (sources + analysis), token*, then done (or error)."""
    dex = get_pokedex()
    ctx = build_context(question, history_pokemon(history))
    context = ctx.render()
    yield {
        "type": "meta",
        "understood": explain(ctx),
        "sources": [dex.brief(dex.pokemon[n]) for n in ctx.pokemon[:10]],
        "other_sources": ctx.other_sources,
        "analysis_kind": ctx.analysis_kind,
        "analysis": compact_analysis(ctx),
        "llm": llm.describe(),
        "context": context,
    }
    if config.LLM_PROVIDER == "none":
        yield {"type": "token", "text": "LLM generation is turned off. Here is the retrieved context:\n\n" + context}
        yield {"type": "done"}
        return

    messages = [m for m in (history or [])[-4:] if m.get("role") in ("user", "assistant") and m.get("content")]
    messages.append({"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"})
    try:
        for text in llm.stream(SYSTEM_PROMPT, messages):
            yield {"type": "token", "text": text}
    except llm.LLMError as e:
        yield {"type": "error", "message": str(e), "context": context}
    yield {"type": "done"}


def answer_text(question, history=None):
    """Blocking helper for the CLI and eval: returns (answer, meta)."""
    meta, parts = None, []
    for ev in answer(question, history):
        if ev["type"] == "meta":
            meta = ev
        elif ev["type"] == "token":
            parts.append(ev["text"])
        elif ev["type"] == "error":
            parts.append(f"[LLM error: {ev['message']}]")
    return "".join(parts), meta
