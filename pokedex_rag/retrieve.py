"""Hybrid retrieval: exact name matches + structured analysis + BM25 + vector search (fused with RRF)."""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from . import strategy
from .config import RERANK_MODEL, TOP_K
from .index import chunks, search
from .router import parse
from .store import STAT_LABELS, get_pokedex, norm, pretty

MAX_CONTEXT_CHARS = 14000
STOPWORDS = set("""a an the is are was were be of to in on for with and or what which who whom how why when where
does do did can could should would i me my it its this that these those pokemon pokémon type types best good
about tell vs versus from into than then them they their there please give show list""".split())
# Move/ability names that are also everyday words in questions.
NAME_STOPLIST = {"counter", "struggle", "rest", "protect", "swift", "charge", "simple", "pressure", "stall",
                 "unaware", "illusion", "run away", "pickup", "teleport", "endure", "growth", "harvest"}
RRF_K = 60


def tokenize(text):
    return [t for t in norm(text).split() if t not in STOPWORDS]


class BM25:
    def __init__(self, docs, k1=1.4, b=0.75):
        self.ids = [d[0] for d in docs]
        self.texts = {}
        for _, text, meta in docs:
            self.texts.setdefault((meta["kind"], meta["ref"]), []).append(text)
        self.meta = [d[2] for d in docs]
        self.tf = [Counter(tokenize(d[1])) for d in docs]
        self.len = [sum(tf.values()) for tf in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(t for tf in self.tf for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def search(self, query, k=8):
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if s:
                scores.append((s, i))
        scores.sort(reverse=True)
        out, seen = [], set()
        for s, i in scores:
            key = (self.meta[i]["kind"], self.meta[i]["ref"])
            if key not in seen:
                seen.add(key)
                out.append(key)
            if len(out) == k:
                break
        return out


@lru_cache(maxsize=1)
def bm25():
    return BM25(list(chunks(get_pokedex())))


@lru_cache(maxsize=1)
def reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(RERANK_MODEL)


def hybrid_search(query, k=TOP_K, mode="rerank", candidates=20):
    """mode: 'bm25', 'vector', 'hybrid' (reciprocal rank fusion of both),
    or 'rerank' (hybrid candidates re-scored by a cross-encoder)."""
    n = candidates if mode == "rerank" else k * 2
    vec = [(kind, ref) for kind, ref, _, _ in search(query, k=n)] if mode != "bm25" else []
    lex = bm25().search(query, k=n) if mode != "vector" else []
    scores = {}
    for ranked in (vec, lex):
        for rank, key in enumerate(ranked):
            scores[key] = scores.get(key, 0) + 1 / (RRF_K + rank)
    fused = [key for key, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    if mode != "rerank":
        return fused[:k]
    # Score every chunk of each candidate; a candidate's score is its best chunk.
    pairs = [(key, text) for key in fused[:candidates] for text in bm25().texts.get(key, [])]
    rs = list(reranker().rerank(query, [t for _, t in pairs]))
    best = {}
    for (key, _), score in zip(pairs, rs):
        best[key] = max(best.get(key, float("-inf")), score)
    return sorted(best, key=best.get, reverse=True)[:k]


@dataclass
class Context:
    query: object
    blocks: list = field(default_factory=list)       # (heading, text)
    pokemon: list = field(default_factory=list)      # names, for UI source cards
    other_sources: list = field(default_factory=list)
    analysis: dict | None = None
    analysis_kind: str | None = None

    def add_pokemon(self, name):
        if name not in self.pokemon:
            self.pokemon.append(name)

    def render(self):
        out, used = [], 0
        for head, text in self.blocks:
            piece = f"## {head}\n{text}\n"
            if used + len(piece) > MAX_CONTEXT_CHARS:
                break
            out.append(piece)
            used += len(piece)
        return "\n".join(out)


def mentioned_names(table, low):
    hits = []
    for key, item in table.items():
        n = norm(item["display"])
        if len(n) > 2 and n not in NAME_STOPLIST and f" {n} " in low:
            hits.append(key)
    return hits


def fmt_team_analysis(dex, a):
    lines = ["Team: " + ", ".join(f"{p['display']} ({'/'.join(t.title() for t in p['types'])})" for p in a["team"])]
    if a["shared_weaknesses"]:
        for w in a["shared_weaknesses"]:
            lines.append(f"- Shared weakness to {w['type'].title()}: {', '.join(w['weak'])} are weak; "
                         f"{'resisted by ' + ', '.join(w['resist']) if w['resist'] else 'nobody resists it'}.")
    else:
        lines.append("- No type hits two or more members super-effectively without a resist on the team.")
    if a["unresisted_types"]:
        lines.append("- Types nobody on the team resists: " + ", ".join(t.title() for t in a["unresisted_types"]))
    if a["coverage_gaps"]:
        lines.append("- Offensive gaps (no super-effective move vs): " + ", ".join(t.title() for t in a["coverage_gaps"]))
    else:
        lines.append("- Offense: the team can hit every type super-effectively with learnable moves.")
    if a["stab_only_gaps"]:
        lines.append("- Types only hit super-effectively by coverage moves (no STAB): "
                     + ", ".join(t.title() for t in a["stab_only_gaps"]))
    lines.append(f"- Physical attackers: {', '.join(a['physical']) or 'none'}; special: {', '.join(a['special']) or 'none'}; "
                 f"average Speed {a['avg_speed']}.")
    lines += [f"- Note: {n}" for n in a["notes"]]
    return "\n".join(lines)


def fmt_counters(c):
    lines = []
    opp_types = ", ".join(t.title() for t in c["opponent_stab_types"])
    lines.append(f"Opposing Pokémon: {', '.join(o['display'] for o in c['opponents'])} (their STAB types: {opp_types}).")
    if c["best_attacking_types"]:
        lines.append("Best attacking types against them: " + "; ".join(
            f"{x['type'].title()} (super effective on {', '.join(x['super_effective_on'])}"
            f"{'; resisted by ' + ', '.join(x['resisted_by']) if x['resisted_by'] else ''})"
            for x in c["best_attacking_types"][:4]))
    for opp, rows in c["per_opponent"].items():
        lines.append(f"Top counters to {opp}:")
        lines += [f"  {i}. {r['display']} — {r['reason']}" for i, r in enumerate(rows[:4], 1)]
    if len(c["opponents"]) > 1:
        lines.append("Pokémon that perform well against the whole group: " + "; ".join(
            f"{r['display']}{' (safely beats ' + ', '.join(r['checks']) + ')' if r['checks'] else ''}"
            for r in c["team_wide"][:5]))
    if "my_team_plan" in c:
        lines.append("Using the asker's own team:")
        for opp, rows in c["my_team_plan"].items():
            lines.append(f"  vs {opp}: " + "; ".join(f"{r['pokemon']} ({r['reason']})" for r in rows))
        if c["no_safe_answer"]:
            lines.append("  The asker's team has no safe super-effective answer to: " + ", ".join(c["no_safe_answer"]))
    return "\n".join(lines)


def build_context(question, history_pokemon=None, k=TOP_K):
    dex = get_pokedex()
    q = parse(dex, question, history_pokemon)
    ctx = Context(query=q)
    low = " " + norm(question) + " "
    filters = {"region": q.region, "allow_legendary": q.allow_legendary, "allow_mega": q.allow_mega}

    if q.intent == "counter":
        c = strategy.counters(dex, q.opponents, my_team=q.my_team or None, **filters)
        ctx.analysis, ctx.analysis_kind = c, "counters"
        ctx.blocks.append(("ANALYSIS (computed by the type-chart engine)", fmt_counters(c)))
        for n in q.opponents + q.my_team:
            ctx.add_pokemon(n)
        for rows in c["per_opponent"].values():
            ctx.add_pokemon(rows[0]["name"])
        for r in c["team_wide"][:2]:
            ctx.add_pokemon(r["name"])

    elif q.intent == "team":
        team_types = q.types[:1] if q.types and "team" in low else None
        pool = {**filters, "types": team_types}
        if q.my_team and len(q.my_team) >= 6:
            a = strategy.analyze_team(dex, q.my_team[:6])
            sugg = strategy.suggest_members(dex, q.my_team[:6], **pool)
            text = fmt_team_analysis(dex, a) + "\nPossible replacements that would patch weaknesses: " + "; ".join(
                f"{s['display']} ({s['reason']})" for s in sugg)
            ctx.analysis = {"analysis": a, "suggestions": sugg}
        else:
            b = strategy.build_team(dex, q.my_team, **pool)
            a = b["analysis"]
            alts = strategy.suggest_members(dex, q.my_team, n=5, **pool) if q.my_team else []
            text = "Suggested full team: " + ", ".join(dex.pokemon[n]["display"] for n in b["team"]) + "\n"
            text += "\n".join(f"- Added {s['display']}: {s['reason']}" for s in b["added"]) + "\n"
            text += fmt_team_analysis(dex, a)
            if alts:
                text += "\nOther good options for the next slot: " + "; ".join(f"{s['display']} ({s['reason']})" for s in alts)
            ctx.analysis = {"analysis": a, "built": b["team"], "suggestions": alts}
            for n in b["team"]:
                ctx.add_pokemon(n)
        ctx.analysis_kind = "team"
        ctx.blocks.append(("ANALYSIS (computed by the type-chart engine)", text))
        for n in q.my_team:
            ctx.add_pokemon(n)

    elif q.intent == "ranking":
        if q.about_moves:
            moves = dex.rank_moves(q.types[0] if q.types else None)
            text = "\n".join(f"{i}. {m['display']} — {m['type'].title()} {m['damage_class']}, power {m['power']}, "
                             f"accuracy {m['accuracy'] or '—'}" for i, m in enumerate(moves, 1))
            ctx.blocks.append(("RANKING (exact, from the database)", text))
        else:
            label = {"bst": "Base Stat Total", "bulk": "HP+Def+SpD"}.get(q.stat, STAT_LABELS.get(q.stat, q.stat))
            ranked = dex.rank(q.stat, descending=q.descending, region=q.region, types=q.types[:2] or None,
                              allow_legendary=not re.search(r"\b(no|non|without|exclude|not) ?legend", low),
                              allow_mega=q.allow_mega)
            scope = " ".join(x for x in [q.region and q.region.title(), "/".join(t.title() for t in q.types)] if x)
            text = f"{'Highest' if q.descending else 'Lowest'} {label}{' among ' + scope + ' Pokémon' if scope else ''}:\n"
            text += "\n".join(f"{i}. {p['display']} — {label} {v} ({'/'.join(t.title() for t in p['types'])})"
                              for i, (p, v) in enumerate(ranked, 1))
            ctx.blocks.append(("RANKING (exact, from the database)", text))
            for p, _ in ranked[:4]:
                ctx.add_pokemon(p["name"])

    for n in q.pokemon[:6]:
        ctx.add_pokemon(n)

    for t in q.types[:2]:
        ctx.blocks.append((f"{t.title()} type", dex.type_card(t)))
        ctx.other_sources.append(f"{t.title()} type")
    if q.region:
        ctx.blocks.append((f"{q.region.title()} region", dex.region_card(q.region)))
        ctx.other_sources.append(f"{q.region.title()} region")
    for a in mentioned_names(dex.abilities, low)[:3]:
        ctx.blocks.append((f"{dex.abilities[a]['display']} (ability)", dex.ability_card(a)))
        ctx.other_sources.append(f"{dex.abilities[a]['display']} (ability)")
    for m in mentioned_names(dex.moves, low)[:3]:
        ctx.blocks.append((f"{dex.moves[m]['display']} (move)", dex.move_card(m)))
        ctx.other_sources.append(f"{dex.moves[m]['display']} (move)")

    # Free-text search fills the rest; fewer hits when exact data already answers the question.
    # Team/counter/ranking answers come from exact data, so search would only add noise there.
    n_search = 0 if q.intent != "lookup" else (k if not q.pokemon else 3)
    for kind, ref in (hybrid_search(question, k=n_search) if n_search else []):
        if norm(ref) in NAME_STOPLIST:
            continue
        if kind == "pokemon":
            ctx.add_pokemon(ref)
        elif kind == "ability" and ref in dex.abilities:
            ctx.blocks.append((f"{dex.abilities[ref]['display']} (ability)", dex.ability_card(ref)))
            ctx.other_sources.append(f"{dex.abilities[ref]['display']} (ability)")
        elif kind == "move" and ref in dex.moves:
            ctx.blocks.append((f"{dex.moves[ref]['display']} (move)", dex.move_card(ref)))
            ctx.other_sources.append(f"{dex.moves[ref]['display']} (move)")
        elif kind == "type" and f"{ref.title()} type" not in ctx.other_sources:
            ctx.blocks.append((f"{ref.title()} type", dex.type_card(ref)))
            ctx.other_sources.append(f"{ref.title()} type")
        elif kind == "region" and f"{ref.title()} region" not in ctx.other_sources:
            ctx.blocks.append((f"{ref.title()} region", dex.region_card(ref)))
            ctx.other_sources.append(f"{ref.title()} region")

    # Pokémon cards go after analysis/rankings so the budget keeps the most important parts.
    for n in ctx.pokemon[:10]:
        ctx.blocks.append((dex.pokemon[n]["display"], dex.card(n)))
    ctx.other_sources = list(dict.fromkeys(ctx.other_sources))
    return ctx


def explain(ctx):
    """Debug view of how a query was understood."""
    q = ctx.query
    return {"intent": q.intent, "pokemon": q.pokemon, "my_team": q.my_team, "opponents": q.opponents,
            "types": q.types, "region": q.region, "stat": q.stat, "descending": q.descending,
            "allow_legendary": q.allow_legendary, "sources": [pretty(n) for n in ctx.pokemon] + ctx.other_sources}
