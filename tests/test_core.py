"""Tests for the deterministic parts: type chart, name matching, query routing and strategy."""

import pytest

from pokedex_rag import strategy
from pokedex_rag.router import parse
from pokedex_rag.store import get_pokedex


@pytest.fixture(scope="module")
def dex():
    return get_pokedex()


# ---------- type chart ----------

@pytest.mark.parametrize("atk,defn,mult", [
    ("fire", ["grass"], 2), ("water", ["fire"], 2), ("ground", ["flying"], 0), ("ghost", ["normal"], 0),
    ("ice", ["dragon", "ground"], 4), ("rock", ["fire", "flying"], 4), ("fighting", ["dark", "rock"], 4),
    ("electric", ["water", "flying"], 4), ("grass", ["fire", "flying"], 0.25), ("dragon", ["fairy"], 0),
])
def test_type_effectiveness(dex, atk, defn, mult):
    assert dex.chart.eff(atk, defn) == mult


def test_charizard_matchups(dex):
    d = dex.pokemon["charizard"]["defense"]
    assert d["4x weak"] == ["rock"]
    assert "ground" in d["immune"]


# ---------- names ----------

@pytest.mark.parametrize("text,name", [
    ("mega charizard x", "charizard-mega-x"), ("Alolan Ninetales", "ninetales-alola"), ("Mr. Mime", "mr-mime"),
    ("farfetch'd", "farfetchd"), ("Ho-Oh", "ho-oh"), ("Iron Valiant", "iron-valiant"), ("Type: Null", "type-null"),
    ("Wash Rotom", "rotom-wash"), ("rotom-wash", "rotom-wash"), ("Porygon-Z", "porygon-z"),
])
def test_find_names(dex, text, name):
    assert dex.find_in_text(f"tell me about {text} please")[0][0] == name


def test_typo_tolerant_resolve(dex):
    assert dex.resolve("garchmop") == "garchomp"
    assert dex.resolve("totally not a pokemon") is None


def test_iron_alone_is_not_a_pokemon(dex):
    assert dex.find_in_text("which steel types have iron in their name") == []


# ---------- routing ----------

def test_route_counter(dex):
    q = parse(dex, "How do I counter Garchomp and Dragonite?")
    assert q.intent == "counter" and q.opponents == ["garchomp", "dragonite"]


def test_route_counter_with_my_team(dex):
    q = parse(dex, "My team is Charizard, Gyarados and Pikachu. How do I beat Gengar?")
    assert q.intent == "counter"
    assert q.my_team == ["charizard", "gyarados", "pikachu"] and q.opponents == ["gengar"]


def test_route_team_review_is_not_counter(dex):
    q = parse(dex, "Check my team: Charizard, Venusaur, Blastoise")
    assert q.intent == "team" and len(q.my_team) == 3


def test_route_team_build_with_region(dex):
    q = parse(dex, "Build me a Paldea team around Garchomp")
    assert q.intent == "team" and q.region == "paldea" and q.my_team == ["garchomp"]


def test_route_ranking(dex):
    q = parse(dex, "Who is the slowest non-legendary Steel type?")
    assert q.intent == "ranking" and q.stat == "spe" and not q.descending and q.types == ["steel"]


def test_route_generation_maps_to_region(dex):
    assert parse(dex, "best gen 4 pokemon by stats").region == "sinnoh"


# ---------- strategy ----------

def test_rank_fastest_kanto(dex):
    top, speed = dex.rank("spe", region="kanto")[0]
    assert top["name"] == "electrode" and speed == 150


def test_counters_to_garchomp_use_ice(dex):
    c = strategy.counters(dex, ["garchomp"])
    best = c["per_opponent"]["Garchomp"]
    assert all("Ice" in r["reason"] or "Fairy" in r["reason"] or "Dragon" in r["reason"] for r in best[:3])


def test_team_analysis_flags_shared_weakness(dex):
    # Three Pokémon that all take super-effective damage from Electric, with no Electric resist.
    a = strategy.analyze_team(dex, ["gyarados", "pelipper", "skarmory"])
    assert "electric" in [w["type"] for w in a["shared_weaknesses"]]


def test_suggestions_patch_shared_weakness(dex):
    sugg = strategy.suggest_members(dex, ["gyarados", "pelipper", "skarmory"], n=3)
    for s in sugg:
        assert dex.chart.eff("electric", s["types"]) < 1, s["display"]


def test_build_team_respects_region_and_size(dex):
    b = strategy.build_team(dex, ["garchomp"], region="paldea")
    assert len(b["team"]) == 6
    for n in b["team"][1:]:
        p = dex.pokemon[n]
        assert "paldea" in p["dex_regions"] or p["origin_region"] == "paldea"


def test_no_duplicate_species_in_suggestions(dex):
    sugg = strategy.suggest_members(dex, ["charizard"], n=6)
    species = [dex.pokemon[s["name"]]["species"] for s in sugg]
    assert len(species) == len(set(species)) and "charizard" not in species


def test_levitate_makes_rotom_wash_ground_immune(dex):
    p = dex.pokemon["rotom-wash"]
    assert dex.chart.eff_vs("ground", p) == 0
    assert "ground" in p["defense"]["immune"]
    best = strategy.counters(dex, ["rotom-wash"])["per_opponent"]["Rotom (Wash)"]
    assert not any("(Ground)" in r["reason"] for r in best)


def test_optional_ability_is_not_assumed(dex):
    # Bronzong (Steel/Psychic) can have Levitate or Heatproof, so Ground keeps its 2x on Steel.
    assert dex.chart.eff_vs("ground", dex.pokemon["bronzong"]) == 2
