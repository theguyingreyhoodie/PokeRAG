"""Rule-based query understanding: which Pokémon, types, regions, stats and intent a question is about.

Rules instead of an LLM call keep this fast, free and predictable, and they work the same
with a 3B local model as with a large hosted one.
"""

import re
from dataclasses import dataclass, field

from .store import REGIONS, norm
from .typechart import TYPES

GEN_WORDS = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9}
GEN_REGION = {1: "kanto", 2: "johto", 3: "hoenn", 4: "sinnoh", 5: "unova", 6: "kalos", 7: "alola", 8: "galar", 9: "paldea"}

# Order matters: more specific phrases first.
STAT_PHRASES = [
    ("special attack", "spa"), ("sp atk", "spa"), ("spatk", "spa"), ("special defense", "spd"), ("sp def", "spd"),
    ("bulkiest", "bulk"), ("tankiest", "bulk"), ("bulk", "bulk"),
    ("fastest", "spe"), ("slowest", "spe"), ("speed", "spe"), ("quickest", "spe"),
    ("hp", "hp"), ("health", "hp"), ("attack", "atk"), ("defense", "def"), ("defence", "def"),
    ("base stat total", "bst"), ("bst", "bst"), ("strongest", "bst"), ("best stats", "bst"),
    ("highest stats", "bst"), ("weakest", "bst"), ("total stats", "bst"),
]
SUPERLATIVE = re.compile(r"\b(fastest|slowest|quickest|strongest|weakest|bulkiest|tankiest|highest|lowest|most|least|best|top|biggest|largest|rank)\b")
ASCENDING = re.compile(r"\b(slowest|weakest|lowest|least|worst)\b")

COUNTER = re.compile(r"\b(counter|counters|countering|beat|beats|beating|against|deal with|check|checks|handle|weak to|wall|walls|struggl\w*|lose to|losing to|answer to|stop)\b")
TEAM = re.compile(r"\b(team|teams|teammate|teammates|partner|partners|build|synergy|core|lineup|line up|party|squad|sixth|6th|fill|add to|rate my)\b")
MY_TEAM = re.compile(r"\b(my team|my party|my lineup|i have|i'm using|i am using|i use|my squad)\b")
MOVE_Q = re.compile(r"\bmoves?\b")


@dataclass
class Query:
    text: str
    pokemon: list = field(default_factory=list)      # resolved Pokémon names in order
    types: list = field(default_factory=list)
    region: str | None = None
    stat: str | None = None
    descending: bool = True
    intent: str = "lookup"                           # lookup | ranking | team | counter
    my_team: list = field(default_factory=list)
    opponents: list = field(default_factory=list)
    allow_legendary: bool = False
    allow_mega: bool = False
    about_moves: bool = False


def parse(dex, text, history_pokemon=None):
    q = Query(text=text)
    low = " " + norm(text) + " "
    found = dex.find_in_text(text)
    q.pokemon = list(dict.fromkeys(n for n, _ in found))

    q.types = [t for t in TYPES if re.search(rf"\b{t}\b", low)]
    for r in REGIONS:
        if re.search(rf"\b{r}(n|an)?\b", low):
            q.region = r
            break
    if not q.region:
        m = re.search(r"\b(?:gen|generation) ?(\d|i{1,3}|iv|v|vi{0,3}|ix)\b", low)
        if m:
            g = m.group(1)
            q.region = GEN_REGION.get(int(g) if g.isdigit() else GEN_WORDS.get(g, 0))

    legend_no = re.search(r"\b(no|non|without|exclude|excluding|not) ?(legend|legendaries|legendary|ubers?)", low)
    q.allow_legendary = bool(re.search(r"\b(legend|legendary|legendaries|ubers?|mythical)\b", low)) and not legend_no
    q.allow_mega = bool(re.search(r"\bmegas?\b", low))
    q.about_moves = bool(MOVE_Q.search(low))

    is_counter = bool(COUNTER.search(low))
    is_team = bool(TEAM.search(low))
    if is_counter and q.pokemon:
        q.intent = "counter"
        my = MY_TEAM.search(low)
        kw = COUNTER.search(low)
        cut = len(low[:COUNTER.search(low).start()].split())
        before = list(dict.fromkeys(n for n, pos in found if pos < cut))
        after = [n for n in dict.fromkeys(n for n, pos in found if pos >= cut) if n not in before]
        if MY_TEAM.search(low) and before and after:
            # "My team is A, B, C. How do I beat X?": names before the counter keyword are mine.
            q.my_team, q.opponents = before, after
        elif MY_TEAM.search(low) and is_team and not before:
            # "Check my team: A, B, C" is a team review, not a counter request.
            q.intent, q.my_team = "team", q.pokemon
        else:
            q.opponents = q.pokemon
    elif is_team:
        q.intent = "team"
        q.my_team = q.pokemon
    elif SUPERLATIVE.search(low) and not q.pokemon:
        q.intent = "ranking"
        for phrase, stat in STAT_PHRASES:
            if re.search(rf"\b{phrase}\b", low):
                q.stat = stat
                break
        q.stat = q.stat or "bst"
        q.descending = not ASCENDING.search(low)

    # Follow-ups like "what about its weaknesses?" inherit the previous turn's Pokémon.
    if not q.pokemon and history_pokemon and q.intent == "lookup" and not q.types and not q.region:
        q.pokemon = list(history_pokemon)
    return q
