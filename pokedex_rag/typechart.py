"""Type effectiveness, built from PokéAPI's damage relations."""

TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground",
    "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy",
]


# Abilities that change how much damage a type does to the holder.
ABILITY_MODS = {
    "levitate": {"ground": 0}, "earth-eater": {"ground": 0},
    "flash-fire": {"fire": 0}, "well-baked-body": {"fire": 0},
    "water-absorb": {"water": 0}, "storm-drain": {"water": 0}, "dry-skin": {"water": 0, "fire": 1.25},
    "volt-absorb": {"electric": 0}, "lightning-rod": {"electric": 0}, "motor-drive": {"electric": 0},
    "sap-sipper": {"grass": 0},
    "thick-fat": {"fire": 0.5, "ice": 0.5}, "heatproof": {"fire": 0.5}, "water-bubble": {"fire": 0.5},
    "purifying-salt": {"ghost": 0.5},
}


def ability_mods(abilities):
    """Type modifiers the Pokémon is guaranteed to have: every non-hidden ability option gives it.

    Rotom-Wash (only Levitate) is treated as Ground-immune; Bronzong (Levitate or Heatproof) is not.
    Returns {type: (multiplier, ability_name)}.
    """
    options = [a["name"] for a in abilities if not a["hidden"]] or [a["name"] for a in abilities]
    mods = {}
    for t in TYPES:
        vals = [ABILITY_MODS.get(a, {}).get(t) for a in options]
        if vals and all(v is not None for v in vals) and len(set(vals)) == 1:
            mods[t] = (vals[0], options[0])
    return mods


class TypeChart:
    def __init__(self, type_data):
        """type_data: {type: {"double_damage_to": [...], "half_damage_to": [...], "no_damage_to": [...]}}"""
        self.mult = {a: {d: 1.0 for d in TYPES} for a in TYPES}
        for atk, rel in type_data.items():
            for d in rel["double_damage_to"]:
                self.mult[atk][d] = 2.0
            for d in rel["half_damage_to"]:
                self.mult[atk][d] = 0.5
            for d in rel["no_damage_to"]:
                self.mult[atk][d] = 0.0

    def eff(self, atk, def_types):
        m = 1.0
        for t in def_types:
            m *= self.mult[atk][t]
        return m

    def eff_vs(self, atk, pokemon):
        """Effectiveness against a specific Pokémon, including guaranteed ability immunities/resists."""
        m = self.eff(atk, pokemon["types"])
        if "_mods" not in pokemon:  # cache on the record; this runs in tight loops
            pokemon["_mods"] = ability_mods(pokemon.get("abilities") or [])
        mod = pokemon["_mods"].get(atk)
        return m * mod[0] if mod else m

    def defensive_profile(self, def_types, mods=None):
        """Damage multiplier taken from each attacking type."""
        mods = mods or {}
        return {a: self.eff(a, def_types) * (mods[a][0] if a in mods else 1) for a in TYPES}

    def summarize_defense(self, def_types, mods=None):
        prof = self.defensive_profile(def_types, mods)
        groups = {"4x weak": [], "2x weak": [], "resists (0.5x)": [], "resists (0.25x)": [], "immune": []}
        for a, m in prof.items():
            key = {4.0: "4x weak", 2.0: "2x weak", 0.5: "resists (0.5x)", 0.25: "resists (0.25x)", 0.0: "immune"}.get(m)
            if key:
                groups[key].append(a)
        return {k: v for k, v in groups.items() if v}
