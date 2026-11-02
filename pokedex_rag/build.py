"""Turn the cached PokéAPI resources into one clean dataset (data/dataset.json)."""

import json
import re

from .config import DATASET_PATH, RAW_DIR
from .typechart import TYPES, TypeChart, ability_mods

STAT_KEYS = {"hp": "hp", "attack": "atk", "defense": "def", "special-attack": "spa",
             "special-defense": "spd", "speed": "spe"}

STARTERS = {
    "kanto": ["bulbasaur", "charmander", "squirtle"],
    "johto": ["chikorita", "cyndaquil", "totodile"],
    "hoenn": ["treecko", "torchic", "mudkip"],
    "sinnoh": ["turtwig", "chimchar", "piplup"],
    "unova": ["snivy", "tepig", "oshawott"],
    "kalos": ["chespin", "fennekin", "froakie"],
    "alola": ["rowlet", "litten", "popplio"],
    "galar": ["grookey", "scorbunny", "sobble"],
    "hisui": ["rowlet", "cyndaquil", "oshawott"],
    "paldea": ["sprigatito", "fuecoco", "quaxly"],
}

FORM_REGIONS = {"alola": ("Alolan", 7), "galar": ("Galarian", 8), "hisui": ("Hisuian", 8), "paldea": ("Paldean", 9)}

# Moves that are poor picks for "best move of this type" (recharge, self-KO, charge turns, gimmicks).
BAD_MOVES = {
    "explosion", "self-destruct", "misty-explosion", "hyper-beam", "giga-impact", "blast-burn",
    "hydro-cannon", "frenzy-plant", "rock-wrecker", "roar-of-time", "prismatic-laser", "eternabeam",
    "meteor-assault", "focus-punch", "sky-attack", "dream-eater", "belch", "last-resort",
    "synchronoise", "mind-blown", "steel-beam", "shell-trap", "head-smash", "light-of-ruin",
    "solar-beam", "solar-blade", "razor-wind", "skull-bash", "dig", "fly", "dive", "bounce",
    "phantom-force", "shadow-force", "freeze-shock", "ice-burn", "geomance", "future-sight",
    "doom-desire", "spit-up", "natural-gift", "fling", "trump-card", "wring-out", "crush-grip",
    "hidden-power", "struggle", "chloroblast", "burn-up", "double-shock", "meteor-beam",
}

GEN_NUM = {"generation-" + r: i for i, r in enumerate(
    ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"], start=1)}


def load(kind):
    return [json.loads(p.read_text()) for p in sorted((RAW_DIR / kind).glob("*.json"))]


def pretty(name):
    return name.replace("-", " ").title()


def form_info(pokemon_name, species_name):
    """Return (form_region, is_mega, display_suffix) for a non-default form name."""
    rest = pokemon_name[len(species_name):].strip("-")
    parts = rest.split("-") if rest else []
    region = next((p for p in parts if p in FORM_REGIONS), None)
    is_mega = "mega" in parts or "primal" in parts
    return region, is_mega, [p for p in parts if p not in FORM_REGIONS and p not in ("mega", "primal")]


def display_name(p, species):
    if p["name"] == next(v["name"] for v in species["varieties"] if v["is_default"]):
        return species["display"]
    region, is_mega, extra = form_info(p["name"], species["name"])
    base = species["display"]
    if region:
        base = f"{FORM_REGIONS[region][0]} {base}"
    if is_mega:
        base = f"{'Primal' if 'primal' in p['name'] else 'Mega'} {base}"
    if extra:
        if is_mega and len(extra) == 1 and len(extra[0]) == 1:  # Mega Charizard X
            return f"{base} {extra[0].upper()}"
        return f"{base} ({' '.join(extra).title()})"
    return base


def evo_condition(d):
    bits = []
    if d.get("min_level"):
        bits.append(f"level {d['min_level']}")
    if d.get("item"):
        bits.append(f"use {pretty(d['item'])}")
    if d.get("trigger") == "trade":
        bits.append("trade")
    if d.get("held_item"):
        bits.append(f"holding {pretty(d['held_item'])}")
    if d.get("min_happiness"):
        bits.append("high friendship")
    if d.get("known_move"):
        bits.append(f"knowing {pretty(d['known_move'])}")
    if d.get("time_of_day"):
        bits.append(f"at {d['time_of_day']}")
    if d.get("location"):
        bits.append(f"at {pretty(d['location'])}")
    if not bits and d.get("trigger") and d["trigger"] != "level-up":
        bits.append(pretty(d["trigger"]).lower())
    return ", ".join(bits) or "level up"


def evolution_edges(chain):
    """Every (from_species, to_species, details) step in a chain, parents before children."""
    edges = []

    def walk(node):
        for child in node["evolves_to"]:
            edges.append((node["species"], child["species"], child["details"]))
            walk(child)

    walk(chain)
    return edges


def evo_method(d):
    """One way to evolve, as structured fields (only those that apply) plus readable text."""
    out = {k: d[k] for k in ("min_level", "item", "held_item", "known_move", "time_of_day", "location") if d.get(k)}
    if d.get("trigger") == "trade":
        out["trade"] = True
    if d.get("min_happiness"):
        out["friendship"] = True
    out["text"] = evo_condition(d)
    return out


def evolution_text(chain, names):
    """Human-readable evolution line. Linear chains use arrows; branching ones list each step."""
    edges = [(a, b, "; or ".join(dict.fromkeys(evo_condition(d) for d in details)) or "special")
             for a, b, details in evolution_edges(chain)]
    if not edges:
        return "Does not evolve."
    branching = len({a for a, _, _ in edges}) < len(edges)
    if not branching:
        out = names.get(edges[0][0], pretty(edges[0][0]))
        for _, b, c in edges:
            out += f" → {names.get(b, pretty(b))} ({c})"
        return out
    return "; ".join(f"{names.get(a, pretty(a))} → {names.get(b, pretty(b))} ({c})" for a, b, c in edges)


def chain_members(chain):
    out, stack = [], [chain]
    while stack:
        n = stack.pop()
        out.append((n["species"], not n["evolves_to"]))
        stack.extend(n["evolves_to"])
    return out


def unique_flavor(entries, limit=4):
    seen, out = set(), []
    for e in reversed(entries):  # newest games first
        key = re.sub(r"[^a-z]", "", e["text"].lower())[:60]
        if key not in seen:
            seen.add(key)
            out.append(e["text"])
        if len(out) == limit:
            break
    return out


def best_moves(learnset, moves, physical):
    """Strongest sensible damaging move of each type this Pokémon can learn."""
    best = {}
    for name in learnset:
        m = moves.get(name)
        if not m or not m["power"] or m["type"] not in TYPES or name in BAD_MOVES or m["damage_class"] == "status":
            continue
        score = m["power"] * ((m["accuracy"] or 100) / 100)
        if (m["damage_class"] == "physical") != physical:
            score *= 0.75
        if m["type"] not in best or score > best[m["type"]][1]:
            best[m["type"]] = (name, score)
    return {t: n for t, (n, _) in best.items()}


def build():
    types = {t["name"]: t for t in load("type")}
    chart = TypeChart(types)
    species = {s["name"]: s for s in load("pokemon-species")}
    pokemon_raw = load("pokemon")
    abilities = {a["name"]: a for a in load("ability")}
    moves = {m["name"]: m for m in load("move")}
    chains = {c["id"]: c["chain"] for c in load("evolution-chain")}
    generations = {g["name"]: g for g in load("generation")}
    regions = {r["name"]: r for r in load("region")}
    pokedexes = load("pokedex")

    species_names = {s: d["display"] for s, d in species.items()}

    # Which regional Pokédexes include each species.
    dex_regions = {}
    for dex in pokedexes:
        if dex["region"] and dex["main_series"]:
            for s in dex["species"]:
                dex_regions.setdefault(s, set()).add(dex["region"])

    fully_evolved, evo_text, evo_line, evolutions = {}, {}, {}, []
    for sp in species.values():
        cid = int(sp["evolution_chain"].rstrip("/").split("/")[-1]) if sp["evolution_chain"] else None
        chain = chains.get(cid)
        if not chain:
            fully_evolved[sp["name"]], evo_text[sp["name"]], evo_line[sp["name"]] = True, "Does not evolve.", []
            continue
        members = chain_members(chain)
        for m, final in members:
            fully_evolved[m] = final
        text = evolution_text(chain, species_names)
        for a, b, details in evolution_edges(chain):
            methods = list({m["text"]: m for m in map(evo_method, details)}.values())
            evolutions.append({"from": a, "to": b, "from_display": species_names.get(a, pretty(a)),
                               "to_display": species_names.get(b, pretty(b)),
                               "methods": methods or [{"text": "special"}]})
        for m, _ in members:
            evo_text[m] = text
            evo_line[m] = [n for n, _ in members]

    out_pokemon = {}
    for p in pokemon_raw:
        sp = species[p["species"]]
        is_default = p["name"] == next(v["name"] for v in sp["varieties"] if v["is_default"])
        form_region, is_mega, _ = (None, False, None) if is_default else form_info(p["name"], sp["name"])
        gen = GEN_NUM.get(sp["generation"], 0)
        origin = generations[sp["generation"]]["main_region"] if sp["generation"] in generations else None
        regions_in = sorted(dex_regions.get(sp["name"], set()))
        if form_region:
            gen, origin, regions_in = FORM_REGIONS[form_region][1], form_region, [form_region]
        stats = {STAT_KEYS[k]: v for k, v in p["stats"].items() if k in STAT_KEYS}
        physical = stats["atk"] >= stats["spa"]
        bm = best_moves(p["moves"], moves, physical)
        out_pokemon[p["name"]] = {
            "name": p["name"],
            "display": display_name(p, sp),
            "species": sp["name"],
            "dex_id": sp["id"],
            "genus": sp["genus"],
            "types": p["types"],
            "stats": stats,
            "bst": sum(stats.values()),
            "abilities": [{
                "name": a["name"],
                "display": abilities.get(a["name"], {}).get("display", pretty(a["name"])),
                "hidden": a["hidden"],
                "effect": abilities.get(a["name"], {}).get("effect", ""),
            } for a in p["abilities"]],
            "height_m": p["height"] / 10,
            "weight_kg": p["weight"] / 10,
            "generation": gen,
            "origin_region": origin,
            "dex_regions": regions_in,
            "is_default": is_default,
            "is_legendary": sp["is_legendary"],
            "is_mythical": sp["is_mythical"],
            "is_baby": sp["is_baby"],
            "is_mega": is_mega,
            "form_region": form_region,
            "fully_evolved": fully_evolved.get(sp["name"], True),
            "evolution": evo_text.get(sp["name"], "Does not evolve."),
            "evolution_line": evo_line.get(sp["name"], []),
            "defense": chart.summarize_defense(p["types"], ability_mods(p["abilities"])),
            "ability_effects": {t: pretty(a) for t, (_, a) in ability_mods(p["abilities"]).items()},
            "best_moves": bm,
            "moves": p["moves"],
            # Regional forms share the species' Pokédex text, which describes the original form.
            "flavor": [] if form_region else unique_flavor(sp["flavor"]),
            "habitat": sp["habitat"],
            "color": sp["color"],
            "egg_groups": sp["egg_groups"],
            "capture_rate": sp["capture_rate"],
            "sprite": p["sprite"],
            "artwork": p["artwork"],
        }

    ability_holders = {}
    for p in out_pokemon.values():
        if p["is_default"]:
            for a in p["abilities"]:
                ability_holders.setdefault(a["name"], []).append(p["display"])
    out_abilities = {
        n: {**a, "pokemon": ability_holders.get(n, [])}
        for n, a in abilities.items() if a["main_series"] and ability_holders.get(n)
    }

    learners = {}
    for p in out_pokemon.values():
        if p["is_default"]:
            for m in p["moves"]:
                learners[m] = learners.get(m, 0) + 1
    out_moves = {n: {**m, "learned_by": learners[n]} for n, m in moves.items()
                 if learners.get(n) and m["type"] in TYPES}

    out_regions = {}
    for r in regions.values():
        dexed = set()
        for dex in pokedexes:
            if dex["region"] == r["name"] and dex["main_series"]:
                dexed.update(dex["species"])
        native = [p for p in out_pokemon.values() if p["origin_region"] == r["name"] and not p["is_mega"]]
        out_regions[r["name"]] = {
            "name": r["name"],
            "display": r["name"].title(),
            "generation": GEN_NUM.get(r["main_generation"] or "", None),
            "games": [pretty(v) for v in r["version_groups"]],
            "pokedexes": [pretty(p) for p in r["pokedexes"]],
            "dex_size": len(dexed),
            "native_count": len(native),
            "starters": [species_names.get(s, s) for s in STARTERS.get(r["name"], [])],
            "legendaries": [p["display"] for p in native if p["is_legendary"]],
            "mythicals": [p["display"] for p in native if p["is_mythical"]],
            "regional_forms": [p["display"] for p in native if p["form_region"]],
            "location_count": r["location_count"],
        }

    dataset = {
        "pokemon": out_pokemon,
        "abilities": out_abilities,
        "moves": out_moves,
        "types": types,
        "regions": out_regions,
        # Each chain is reachable from every member species; keep one copy of each step.
        "evolutions": list({(e["from"], e["to"]): e for e in evolutions}.values()),
    }
    DATASET_PATH.write_text(json.dumps(dataset))
    print(f"Built dataset: {len(out_pokemon)} Pokémon, {len(out_abilities)} abilities, "
          f"{len(out_moves)} moves, {len(out_regions)} regions -> {DATASET_PATH}")
    return dataset


if __name__ == "__main__":
    build()
