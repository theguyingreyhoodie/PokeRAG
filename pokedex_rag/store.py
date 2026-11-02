"""In-memory Pokédex: the dataset, name lookup, structured filters, and text cards for the LLM."""

import difflib
import json
import operator
import re
import unicodedata
from functools import lru_cache

from .config import DATASET_PATH
from .typechart import TYPES, TypeChart

STAT_LABELS = {"hp": "HP", "atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD", "spe": "Spe"}
REGIONS = ["kanto", "johto", "hoenn", "sinnoh", "unova", "kalos", "alola", "galar", "hisui", "paldea"]
FORM_ADJ = {"alola": "alolan", "galar": "galarian", "hisui": "hisuian", "paldea": "paldean"}
OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
# Move/ability names that are also everyday words; never treated as a mention.
NAME_STOPLIST = {"counter", "struggle", "rest", "protect", "swift", "charge", "simple", "pressure", "stall",
                 "unaware", "illusion", "run away", "pickup", "teleport", "endure", "growth", "harvest"}


def norm(text):
    """Lowercase, strip accents/punctuation, turn hyphens into spaces: 'Farfetch’d' -> 'farfetchd'."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower().replace("♀", " f").replace("♂", " m")
    text = re.sub(r"[.'’:]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def pretty(name):
    return name.replace("-", " ").title()


def mentioned_names(table, low):
    """Keys of moves/abilities whose display name appears in normalized text `low` (padded with spaces)."""
    hits = []
    for key, item in table.items():
        n = norm(item["display"])
        if len(n) > 2 and n not in NAME_STOPLIST and f" {n} " in low:
            hits.append(key)
    return hits


class Pokedex:
    def __init__(self, path=DATASET_PATH):
        data = json.loads(path.read_text())
        self.pokemon = data["pokemon"]
        self.abilities = data["abilities"]
        self.moves = data["moves"]
        self.types = data["types"]
        self.regions = data["regions"]
        self.evolutions = data.get("evolutions", [])  # older datasets: rerun `ingest` or `python -m pokedex_rag.build`
        # Evolution items as they'd be written in a question: "moon stone" -> "moon-stone".
        self.evo_items = {norm(m[k]): m[k] for e in self.evolutions for m in e["methods"]
                          for k in ("item", "held_item") if m.get(k)}
        self.chart = TypeChart(self.types)
        self.alias = self._build_aliases()
        self.max_alias_len = max(len(a.split()) for a in self.alias)

    # ---------- names ----------

    def _build_aliases(self):
        alias = {}

        def add(key, name, override=False):
            key = norm(key)
            if key and (override or key not in alias):
                alias[key] = name

        for p in self.pokemon.values():
            if p["is_default"]:
                add(p["display"], p["name"], override=True)
                add(p["species"], p["name"], override=True)
            add(p["display"], p["name"])
            add(p["name"], p["name"])
            base = self.pokemon_by_species(p["species"])
            base_name = base["display"] if base else p["species"]
            if p["form_region"]:
                add(f"{FORM_ADJ[p['form_region']]} {base_name}", p["name"])
            elif not p["is_default"] and not p["is_mega"]:
                # "Wash Rotom", "Therian Landorus", "Bloodmoon Ursaluna"
                add(f"{p['name'][len(p['species']):]} {base_name}", p["name"])
        add("nidoran", "nidoran-m")
        return alias

    def pokemon_by_species(self, species):
        for p in self.pokemon.values():
            if p["species"] == species and p["is_default"]:
                return p
        return None

    def find_spans(self, text):
        """All Pokémon mentioned in free text, in order, as (name, start_token, end_token). Longest match wins."""
        tokens = norm(text).split()
        found, i = [], 0
        while i < len(tokens):
            for n in range(min(self.max_alias_len, len(tokens) - i), 0, -1):
                key = " ".join(tokens[i:i + n])
                if key in self.alias and (n > 1 or key not in COMMON_WORDS):
                    found.append((self.alias[key], i, i + n))
                    i += n
                    break
            else:
                i += 1
        return found

    def find_in_text(self, text):
        """All Pokémon mentioned in free text, in order, as (name, token_position)."""
        return [(name, start) for name, start, _ in self.find_spans(text)]

    def resolve(self, text):
        """Best-effort single-name lookup with typo tolerance (for team-builder inputs)."""
        key = norm(text)
        if key in self.alias:
            return self.alias[key]
        close = difflib.get_close_matches(key, self.alias.keys(), n=1, cutoff=0.75)
        return self.alias[close[0]] if close else None

    def search_names(self, prefix, limit=8):
        key = norm(prefix)
        hits = [p for p in self.pokemon.values() if norm(p["display"]).startswith(key)]
        hits += [p for p in self.pokemon.values() if key in norm(p["display"]) and p not in hits]
        return hits[:limit]

    # ---------- structured queries ----------

    def pool(self, region=None, types=None, allow_legendary=False, allow_mega=False,
             fully_evolved=None, min_bst=0):
        out = []
        for p in self.pokemon.values():
            if p["is_mega"] and not allow_mega:
                continue
            if not allow_legendary and (p["is_legendary"] or p["is_mythical"]):
                continue
            if region and region not in p["dex_regions"] and p["origin_region"] != region:
                continue
            if types and not all(t in p["types"] for t in types):
                continue
            if fully_evolved is not None and p["fully_evolved"] != fully_evolved:
                continue
            if p["bst"] < min_bst:
                continue
            out.append(p)
        return out

    @staticmethod
    def value(p, field):
        """A numeric field: 'bst', 'bulk' (HP+Def+SpD), 'weight_kg', 'height_m', or a stat key ('hp', 'atk', ...)."""
        if field in ("bst", "weight_kg", "height_m"):
            return p[field]
        if field == "bulk":
            return p["stats"]["hp"] + p["stats"]["def"] + p["stats"]["spd"]
        return p["stats"][field]

    def rank(self, stat, n=10, descending=True, **filters):
        """Top Pokémon by a stat ('bst', 'hp', 'atk', ..., or 'bulk' = HP+Def+SpD)."""
        filters.setdefault("allow_legendary", True)
        filters.setdefault("allow_mega", False)
        ranked = sorted(self.pool(**filters), key=lambda p: self.value(p, stat), reverse=descending)
        return [(p, self.value(p, stat)) for p in ranked[:n]]

    def filter_pokemon(self, move=None, ability=None, numeric=(), **pool_filters):
        """Pokémon matching every condition. `numeric` holds (field, op, value) like ("weight_kg", "<", 50)."""
        out = []
        for p in self.pool(**pool_filters):
            if move and move not in p["moves"]:
                continue
            if ability and not any(a["name"] == ability for a in p["abilities"]):
                continue
            if all(OPS[op](self.value(p, f), v) for f, op, v in numeric):
                out.append(p)
        return out

    def rank_moves(self, type_=None, n=10, damage_class=None):
        moves = [m for m in self.moves.values() if m["power"] and (not type_ or m["type"] == type_)
                 and (not damage_class or m["damage_class"] == damage_class)]
        return sorted(moves, key=lambda m: (m["power"], m["accuracy"] or 100), reverse=True)[:n]

    def evolution_steps(self, species):
        """Steps into and out of a species: (evolves_from, evolves_into)."""
        return ([e for e in self.evolutions if e["to"] == species],
                [e for e in self.evolutions if e["from"] == species])

    def evolutions_by(self, item=None, trade=False, friendship=False):
        """Evolution steps that use a given item (used or held), trading, or friendship."""
        def ok(m):
            return ((not item or item in (m.get("item"), m.get("held_item")))
                    and (not trade or m.get("trade")) and (not friendship or m.get("friendship")))
        return [e for e in self.evolutions if any(ok(m) for m in e["methods"])]

    # ---------- text cards (what the LLM sees) ----------

    @lru_cache(maxsize=4096)
    def card(self, name):
        p = self.pokemon[name]
        s = p["stats"]
        lines = [f"[{p['display']}] #{p['dex_id']} {p['display']} — {p['genus']}"]
        lines.append(f"Type: {'/'.join(t.title() for t in p['types'])}")
        region = f"Region: {p['origin_region'].title() if p['origin_region'] else 'Unknown'} (Generation {p['generation']})"
        if p["dex_regions"]:
            region += f"; appears in the regional Pokédex of {', '.join(r.title() for r in p['dex_regions'])}"
        lines.append(region)
        lines.append("Base stats: " + " / ".join(f"{STAT_LABELS[k]} {v}" for k, v in s.items()) + f" (Total {p['bst']})")
        lines.append("Abilities: " + "; ".join(
            f"{a['display']}{' (hidden)' if a['hidden'] else ''} — {a['effect']}" for a in p["abilities"]))
        lines.append(f"Evolution: {p['evolution']}{'' if p['fully_evolved'] else ' (not fully evolved)'}")
        d = p["defense"]
        parts = []
        if "4x weak" in d:
            parts.append("4x weak to " + ", ".join(t.title() for t in d["4x weak"]))
        if "2x weak" in d:
            parts.append("2x weak to " + ", ".join(t.title() for t in d["2x weak"]))
        res = d.get("resists (0.25x)", []) + d.get("resists (0.5x)", [])
        if res:
            parts.append("resists " + ", ".join(t.title() for t in res))
        if "immune" in d:
            parts.append("immune to " + ", ".join(t.title() for t in d["immune"]))
        lines.append("Type matchups: " + "; ".join(parts))
        if p.get("ability_effects"):
            lines.append("Matchups above include its ability: " + ", ".join(
                f"{a} affects {t.title()} attacks" for t, a in p["ability_effects"].items()))
        tags = [t for t, flag in (("Legendary", p["is_legendary"]), ("Mythical", p["is_mythical"]),
                                  ("Baby", p["is_baby"]), ("Mega/Primal form", p["is_mega"])) if flag]
        if tags:
            lines.append("Category: " + ", ".join(tags))
        stab = [f"{t.title()}: {pretty(m)}" for t, m in p["best_moves"].items() if t in p["types"]]
        cov = [f"{t.title()}: {pretty(m)}" for t, m in p["best_moves"].items() if t not in p["types"]]
        if stab:
            lines.append("Best same-type (STAB) moves: " + ", ".join(stab))
        if cov:
            lines.append("Coverage moves it can learn: " + ", ".join(cov[:10]))
        lines.append(f"Height {p['height_m']} m, weight {p['weight_kg']} kg")
        if p["flavor"]:
            lines.append("Pokédex entries: " + " ".join(f'"{f}"' for f in p["flavor"][:3]))
        return "\n".join(lines)

    def ability_card(self, name):
        a = self.abilities[name]
        holders = a["pokemon"]
        more = f" and {len(holders) - 25} more" if len(holders) > 25 else ""
        return f"[{a['display']} (ability)] {a['display']}: {a['effect']} Pokémon with this ability: {', '.join(holders[:25])}{more}."

    def move_card(self, name):
        m = self.moves[name]
        return (f"[{m['display']} (move)] {m['display']} — {m['type'].title()}-type {m['damage_class']} move. "
                f"Power {m['power'] or '—'}, accuracy {m['accuracy'] or '—'}, PP {m['pp']}, priority {m['priority']}. "
                f"Effect: {m['effect']} Learned by {m['learned_by']} Pokémon.")

    def type_card(self, t):
        atk = self.chart.mult[t]
        se = [d.title() for d in TYPES if atk[d] == 2]
        nve = [d.title() for d in TYPES if atk[d] == 0.5]
        immune = [d.title() for d in TYPES if atk[d] == 0]
        weak = [a.title() for a in TYPES if self.chart.mult[a][t] == 2]
        res = [a.title() for a in TYPES if self.chart.mult[a][t] == 0.5]
        imm = [a.title() for a in TYPES if self.chart.mult[a][t] == 0]
        return (f"[{t.title()} type] Attacking: super effective against {', '.join(se) or 'nothing'}; "
                f"not very effective against {', '.join(nve) or 'nothing'}"
                f"{'; no effect on ' + ', '.join(immune) if immune else ''}. "
                f"Defending: weak to {', '.join(weak) or 'nothing'}; resists {', '.join(res) or 'nothing'}"
                f"{'; immune to ' + ', '.join(imm) if imm else ''}.")

    def region_card(self, r):
        g = self.regions[r]
        return (f"[{g['display']} region] {g['display']} — introduced in Generation {g['generation']}. "
                f"Games: {', '.join(g['games'])}. Regional Pokédexes: {', '.join(g['pokedexes']) or 'none'} "
                f"({g['dex_size']} species in total). {g['native_count']} Pokémon/forms originate here. "
                f"Starters: {', '.join(g['starters']) or 'n/a'}. "
                f"Legendaries: {', '.join(g['legendaries']) or 'none'}. Mythicals: {', '.join(g['mythicals']) or 'none'}. "
                f"Regional forms: {', '.join(g['regional_forms']) or 'none'}. {g['location_count']} locations.")

    def brief(self, p):
        """Small JSON-friendly summary for the UI."""
        return {"name": p["name"], "display": p["display"], "types": p["types"], "sprite": p["sprite"],
                "artwork": p["artwork"], "bst": p["bst"], "stats": p["stats"], "dex_id": p["dex_id"]}


# Single-word Pokémon names that are also everyday words; only matched when part of a longer alias.
COMMON_WORDS = {"type", "mega", "iron", "great", "roaring", "walking", "flutter", "gouging", "raging"}


@lru_cache(maxsize=1)
def get_pokedex():
    return Pokedex()
