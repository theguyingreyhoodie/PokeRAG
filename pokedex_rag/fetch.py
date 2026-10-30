"""Download everything we need from PokéAPI, slim it down, and cache it on disk.

PokéAPI asks clients to cache responses, so every resource is fetched once and stored
under data/raw/<kind>/<name>.json. Re-running only fetches what is missing.
"""

import asyncio
import json
import re

import httpx

from .config import POKEAPI_BASE, RAW_DIR

CONCURRENCY = 10

# Alternate forms are kept when they matter for battling or regions (regional variants, megas,
# Rotom appliances, Therian forms, ...). Cosmetic, Gigantamax, totem and story-only forms are skipped.
FORM_EXCLUDE = (
    "totem", "-cap", "gmax", "starter", "-cosplay", "pikachu-", "minior-", "squawkabilly-", "tatsugiri-",
    "-build", "-mode", "power-construct", "eternamax", "battle-bond", "-busted", "-noice", "-hangry",
    "cramorant-", "-school", "castform-", "-blade", "maushold-", "dudunsparce-", "-own-tempo",
    "gimmighoul-", "-small", "-large", "-super", "-female", "basculin-", "magearna-original", "zarude-",
    "terapagos-stellar",
)

MAIN_TYPES = {
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground",
    "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy",
}


def _en(entries, key):
    """First English value of `key` in a list of localized entries."""
    for e in entries or []:
        if e.get("language", {}).get("name") == "en":
            return e.get(key)
    return None


def _clean(text):
    return re.sub(r"\s+", " ", (text or "").replace("­", "").replace("\f", " ")).strip()


def slim_species(d):
    flavor = []
    for e in d["flavor_text_entries"]:
        if e["language"]["name"] == "en":
            flavor.append({"text": _clean(e["flavor_text"]), "version": e["version"]["name"]})
    return {
        "id": d["id"],
        "name": d["name"],
        "display": _en(d["names"], "name") or d["name"].title(),
        "genus": _en(d["genera"], "genus") or "",
        "generation": d["generation"]["name"],
        "is_legendary": d["is_legendary"],
        "is_mythical": d["is_mythical"],
        "is_baby": d["is_baby"],
        "evolves_from": (d.get("evolves_from_species") or {}).get("name"),
        "evolution_chain": d["evolution_chain"]["url"] if d.get("evolution_chain") else None,
        "habitat": (d.get("habitat") or {}).get("name"),
        "color": (d.get("color") or {}).get("name"),
        "egg_groups": [g["name"] for g in d["egg_groups"]],
        "capture_rate": d["capture_rate"],
        "flavor": flavor,
        "varieties": [{"name": v["pokemon"]["name"], "is_default": v["is_default"]} for v in d["varieties"]],
    }


def slim_pokemon(d):
    sprites = d.get("sprites") or {}
    art = ((sprites.get("other") or {}).get("official-artwork") or {}).get("front_default")
    return {
        "id": d["id"],
        "name": d["name"],
        "species": d["species"]["name"],
        "types": [t["type"]["name"] for t in sorted(d["types"], key=lambda t: t["slot"])],
        "stats": {s["stat"]["name"]: s["base_stat"] for s in d["stats"]},
        "abilities": [{"name": a["ability"]["name"], "hidden": a["is_hidden"]} for a in d["abilities"]],
        "height": d["height"],
        "weight": d["weight"],
        "moves": sorted({m["move"]["name"] for m in d["moves"]}),
        "sprite": sprites.get("front_default"),
        "artwork": art,
    }


def slim_evolution(d):
    def walk(node):
        details = []
        for e in node.get("evolution_details") or []:
            details.append({
                "trigger": (e.get("trigger") or {}).get("name"),
                "min_level": e.get("min_level"),
                "item": (e.get("item") or {}).get("name"),
                "held_item": (e.get("held_item") or {}).get("name"),
                "min_happiness": e.get("min_happiness"),
                "time_of_day": e.get("time_of_day") or None,
                "known_move": (e.get("known_move") or {}).get("name"),
                "location": (e.get("location") or {}).get("name"),
            })
        return {
            "species": node["species"]["name"],
            "details": details,
            "evolves_to": [walk(n) for n in node["evolves_to"]],
        }

    return {"id": d["id"], "chain": walk(d["chain"])}


def slim_ability(d):
    effect = _en(d["effect_entries"], "short_effect")
    if not effect:  # newer abilities often only have flavor text
        effect = _en(d["flavor_text_entries"], "flavor_text")
    return {
        "name": d["name"],
        "display": _en(d["names"], "name") or d["name"].replace("-", " ").title(),
        "effect": _clean(effect),
        "generation": d["generation"]["name"],
        "main_series": d["is_main_series"],
    }


def slim_move(d):
    effect = _en(d["effect_entries"], "short_effect")
    if not effect:
        effect = _en(d["flavor_text_entries"], "flavor_text")
    effect = _clean(effect)
    if d.get("effect_chance") is not None:
        effect = effect.replace("$effect_chance", str(d["effect_chance"]))
    return {
        "name": d["name"],
        "display": _en(d["names"], "name") or d["name"].replace("-", " ").title(),
        "type": d["type"]["name"],
        "power": d["power"],
        "accuracy": d["accuracy"],
        "pp": d["pp"],
        "priority": d["priority"],
        "damage_class": (d.get("damage_class") or {}).get("name"),
        "effect": effect,
        "generation": d["generation"]["name"],
    }


def slim_type(d):
    rel = d["damage_relations"]
    return {
        "name": d["name"],
        "double_damage_to": [t["name"] for t in rel["double_damage_to"]],
        "half_damage_to": [t["name"] for t in rel["half_damage_to"]],
        "no_damage_to": [t["name"] for t in rel["no_damage_to"]],
    }


def slim_generation(d):
    return {"name": d["name"], "id": d["id"], "main_region": d["main_region"]["name"],
            "version_groups": [v["name"] for v in d["version_groups"]]}


def slim_region(d):
    return {
        "name": d["name"],
        "main_generation": (d.get("main_generation") or {}).get("name"),
        "version_groups": [v["name"] for v in d["version_groups"]],
        "pokedexes": [p["name"] for p in d["pokedexes"]],
        "location_count": len(d["locations"]),
    }


def slim_pokedex(d):
    return {
        "name": d["name"],
        "region": (d.get("region") or {}).get("name"),
        "main_series": d["is_main_series"],
        "species": [e["pokemon_species"]["name"] for e in d["pokemon_entries"]],
    }


class Fetcher:
    def __init__(self, client):
        self.client = client
        self.sem = asyncio.Semaphore(CONCURRENCY)
        self.fetched = 0

    async def get_json(self, url):
        async with self.sem:
            for attempt in range(4):
                try:
                    r = await self.client.get(url)
                    r.raise_for_status()
                    return r.json()
                except (httpx.HTTPError, ValueError):
                    if attempt == 3:
                        raise
                    await asyncio.sleep(2 ** attempt)

    async def resource(self, kind, name, slim):
        """Return a slimmed resource, from cache if possible."""
        path = RAW_DIR / kind / f"{name}.json"
        if path.exists():
            return json.loads(path.read_text())
        url = name if name.startswith("http") else f"{POKEAPI_BASE}/{kind}/{name}"
        data = slim(await self.get_json(url))
        key = str(data.get("name", data.get("id")))
        path = RAW_DIR / kind / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        self.fetched += 1
        if self.fetched % 250 == 0:
            print(f"  fetched {self.fetched} resources...")
        return data

    async def names(self, kind):
        data = await self.get_json(f"{POKEAPI_BASE}/{kind}?limit=5000")
        return [r["name"] for r in data["results"]]

    async def many(self, kind, names, slim):
        return await asyncio.gather(*(self.resource(kind, n, slim) for n in names))


def keep_form(name):
    return not any(s in name for s in FORM_EXCLUDE)


async def fetch_all():
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": "pokedex-rag (portfolio project)"}) as client:
        f = Fetcher(client)
        print("Fetching species...")
        species = await f.many("pokemon-species", await f.names("pokemon-species"), slim_species)

        pokemon_names = []
        for s in species:
            for v in s["varieties"]:
                if v["is_default"] or keep_form(v["name"]):
                    pokemon_names.append(v["name"])
        print(f"Fetching {len(pokemon_names)} Pokémon (default forms + regional forms + megas)...")
        await f.many("pokemon", pokemon_names, slim_pokemon)

        chain_urls = sorted({s["evolution_chain"] for s in species if s["evolution_chain"]})
        print(f"Fetching {len(chain_urls)} evolution chains...")
        # Chains are cached by id, so filter out cached ones before fetching by URL.
        missing = [u for u in chain_urls
                   if not (RAW_DIR / "evolution-chain" / f"{u.rstrip('/').split('/')[-1]}.json").exists()]
        await f.many("evolution-chain", missing, slim_evolution)

        print("Fetching abilities, moves, types, generations, regions, pokedexes...")
        await f.many("ability", await f.names("ability"), slim_ability)
        await f.many("move", await f.names("move"), slim_move)
        await f.many("type", [t for t in await f.names("type") if t in MAIN_TYPES], slim_type)
        await f.many("generation", await f.names("generation"), slim_generation)
        await f.many("region", await f.names("region"), slim_region)
        await f.many("pokedex", await f.names("pokedex"), slim_pokedex)
        print(f"Done. {f.fetched} new resources downloaded.")


def main():
    asyncio.run(fetch_all())


if __name__ == "__main__":
    main()
