"""Deterministic battle math: team weaknesses, coverage, counters, and team suggestions.

The LLM never does this arithmetic. These functions compute it, and their output is
handed to the LLM as context to explain.
"""

from .store import pretty
from .typechart import TYPES

# Abilities that badly handicap a Pokémon whose every ability option is one of these (e.g. Slaking).
CRIPPLING_ABILITIES = {"truant", "slow-start", "defeatist"}


def crippled(p):
    return all(a["name"] in CRIPPLING_ABILITIES for a in p["abilities"])


def offense(dex, attacker, defender):
    """Best hit an attacker can land on a defender (a Pokémon, or {"types": [...]})."""
    best = (0.0, None, None, 0.0)
    moves = attacker["best_moves"] or {t: None for t in attacker["types"]}
    for mtype, move in moves.items():
        tm = dex.chart.eff_vs(mtype, defender)
        m = tm * (1.5 if mtype in attacker["types"] else 1.0)
        if m > best[0]:
            best = (m, move, mtype, tm)
    return best


def threat(dex, attacker, defender):
    """How hard `attacker` hits `defender`.

    Returns (weighted, stab_mult, stab_type, cov_mult, cov_move): STAB counts fully,
    coverage moves at 2/3 weight since the attacker may not be running them.
    """
    stab, stab_t, cov, cov_m = 0.0, None, 0.0, None
    for mtype, move in (attacker["best_moves"] or {t: None for t in attacker["types"]}).items():
        tm = dex.chart.eff_vs(mtype, defender)
        if mtype in attacker["types"]:
            if tm > stab or stab_t is None:
                stab, stab_t = tm, mtype
        elif tm > cov:
            cov, cov_m = tm, move
    return max(stab, cov * 2 / 3), stab, stab_t, cov, cov_m


def matchup(dex, cand, opp):
    """Score how well `cand` handles `opp` (higher is better) with a short explanation."""
    o_mult, move, mtype, tm = offense(dex, cand, opp)
    thr, stab, stab_t, cov, cov_m = threat(dex, opp, cand)
    s, os_ = cand["stats"], opp["stats"]
    attack = max(s["atk"], s["spa"]) / 100
    bulk = (s["hp"] + s["def"] + s["spd"]) / 300
    faster = s["spe"] > os_["spe"]
    score = (o_mult / 1.5) * attack * bulk * (1.15 if faster else 1.0) / (0.5 + thr)
    if crippled(cand):
        score *= 0.5
    why = []
    if move:
        why.append(f"{pretty(move)} ({mtype.title()}) does {tm:g}x to {opp['display']}")
    opp_stab = "/".join(t.title() for t in opp["types"])
    if stab == 0:
        why.append(f"immune to its {opp_stab} STAB")
    elif stab < 1:
        why.append(f"resists its {opp_stab} STAB")
    elif stab >= 2:
        why.append(f"but is weak to its {stab_t.title()} STAB")
    if cov >= 2 and cov_m:
        why.append(f"watch out for {pretty(cov_m)} ({cov:g}x)")
    why.append("outspeeds it" if faster else "slower than it")
    return {"score": round(score, 3), "hits_se": tm >= 2, "safe": thr <= 1, "reason": "; ".join(why)}


def analyze_team(dex, names):
    team = [dex.pokemon[n] for n in names]
    defense = {}
    for atk in TYPES:
        row = {p["display"]: dex.chart.eff_vs(atk, p) for p in team}
        weak = [k for k, v in row.items() if v > 1]
        resist = [k for k, v in row.items() if v < 1]
        defense[atk] = {"weak": weak, "resist": resist, "multipliers": row}

    shared_weak = [t for t, r in defense.items() if len(r["weak"]) >= 2 and len(r["weak"]) > len(r["resist"])]
    unresisted = [t for t, r in defense.items() if not r["resist"]]

    coverage, gaps = {}, []
    for dt in TYPES:
        hits = []
        for p in team:
            for mtype, move in p["best_moves"].items():
                if dex.chart.eff(mtype, [dt]) >= 2:
                    hits.append({"pokemon": p["display"], "move": pretty(move), "stab": mtype in p["types"]})
        stab_hits = [h for h in hits if h["stab"]]
        coverage[dt] = (stab_hits or hits)[:3]
        if not hits:
            gaps.append(dt)
    stab_gaps = [t for t in TYPES if not any(h["stab"] for h in coverage[t]) and t not in gaps]

    phys = [p["display"] for p in team if p["stats"]["atk"] >= p["stats"]["spa"]]
    spec = [p["display"] for p in team if p["stats"]["atk"] < p["stats"]["spa"]]
    notes = []
    for p in team:
        if not p["fully_evolved"]:
            notes.append(f"{p['display']} is not fully evolved.")
        if p["is_legendary"] or p["is_mythical"]:
            notes.append(f"{p['display']} is legendary/mythical (banned in many formats).")
    if len(team) >= 4 and (len(phys) <= 1 or len(spec) <= 1):
        notes.append(f"Attack split is lopsided: {len(phys)} physical vs {len(spec)} special attackers.")

    return {
        "team": [dex.brief(p) for p in team],
        "shared_weaknesses": [{"type": t, "weak": defense[t]["weak"], "resist": defense[t]["resist"]} for t in shared_weak],
        "unresisted_types": unresisted,
        "coverage_gaps": gaps,
        "stab_only_gaps": stab_gaps,
        "coverage": coverage,
        "defense": defense,
        "physical": phys,
        "special": spec,
        "avg_speed": round(sum(p["stats"]["spe"] for p in team) / len(team)) if team else 0,
        "notes": notes,
    }


def _same_species(a, b):
    return a["species"] == b["species"]


def suggest_members(dex, names, n=5, **pool_filters):
    """Candidates that patch the team's shared weaknesses and coverage gaps."""
    team = [dex.pokemon[x] for x in names]
    analysis = analyze_team(dex, names)
    weak_counts = {t: len(analysis["defense"][t]["weak"]) for t in TYPES}
    shared = {w["type"] for w in analysis["shared_weaknesses"]}
    gaps = set(analysis["coverage_gaps"])
    team_types = {t for p in team for t in p["types"]}

    pool_filters.setdefault("fully_evolved", True)
    pool_filters.setdefault("min_bst", 420)
    out = []
    for c in dex.pool(**pool_filters):
        if any(_same_species(c, p) for p in team):
            continue
        score, why = (c["bst"] - 500) / 50, []
        for t in TYPES:
            m = dex.chart.eff_vs(t, c)
            if t in shared:
                if m == 0:
                    score += 3 * weak_counts[t]
                    why.append(f"immune to {t.title()}")
                elif m < 1:
                    score += 2 * weak_counts[t]
                    why.append(f"resists {t.title()}")
                elif m > 1:
                    score -= 2 * weak_counts[t]
            elif m > 1 and weak_counts[t] >= 1:
                score -= 1.5 * weak_counts[t]  # stacks an existing weakness
        if crippled(c):
            score -= 4
        # Reward resisting types nobody on the team resists yet.
        new_resists = [t for t in analysis["unresisted_types"] if dex.chart.eff_vs(t, c) < 1]
        score += 0.75 * len(new_resists)
        if new_resists and not why:
            why.append("adds a resistance to " + ", ".join(t.title() for t in new_resists[:4]))
        covered = [t for t in gaps if offense(dex, c, {"types": [t]})[3] >= 2]
        if covered:
            score += 1.5 * len(covered)
            why.append("adds super-effective coverage vs " + ", ".join(t.title() for t in covered))
        overlap = team_types & set(c["types"])
        score -= 1.5 * len(overlap)
        out.append({**dex.brief(c), "score": round(score, 2),
                    "reason": "; ".join(why) or "solid stats with no new shared weaknesses"})
    out.sort(key=lambda x: x["score"], reverse=True)

    picked, seen_species = [], set()
    for c in out:
        sp = dex.pokemon[c["name"]]["species"]
        if sp not in seen_species:
            picked.append(c)
            seen_species.add(sp)
        if len(picked) == n:
            break
    return picked


def build_team(dex, core, size=6, **pool_filters):
    """Greedily complete a team: repeatedly add the best suggestion for the current team."""
    team = list(core)
    if not team:
        seeds = [p for p in dex.pool(fully_evolved=True, **pool_filters) if not crippled(p)]
        if not seeds:
            return {"team": [], "added": []}
        team = [max(seeds, key=lambda p: p["bst"])["name"]]
    added = []
    while len(team) < size:
        sugg = suggest_members(dex, team, n=1, **pool_filters)
        if not sugg:
            break
        team.append(sugg[0]["name"])
        added.append(sugg[0])
    return {"team": team, "added": added, "analysis": analyze_team(dex, team)}


def counters(dex, opponents, my_team=None, n=5, **pool_filters):
    """Best answers to each opposing Pokémon, plus Pokémon that handle the whole team."""
    opps = [dex.pokemon[x] for x in opponents]
    pool_filters.setdefault("fully_evolved", True)
    pool_filters.setdefault("min_bst", 420)
    pool = [c for c in dex.pool(**pool_filters) if not any(_same_species(c, o) for o in opps)]

    per_opp, totals = {}, {}
    for o in opps:
        rows = []
        for c in pool:
            m = matchup(dex, c, o)
            rows.append((c, m))
            t = totals.setdefault(c["name"], {"sum": 0.0, "checks": []})
            t["sum"] += min(m["score"], 3.0)  # cap so one lopsided matchup doesn't dominate
            if m["hits_se"] and m["safe"]:
                t["checks"].append(o["display"])
        rows.sort(key=lambda r: (r[1]["hits_se"], r[1]["score"]), reverse=True)
        best, seen = [], set()
        for c, m in rows:
            if c["species"] not in seen:
                best.append({**dex.brief(c), **m})
                seen.add(c["species"])
            if len(best) == n:
                break
        per_opp[o["display"]] = best

    overall = sorted(pool, key=lambda c: totals[c["name"]]["sum"], reverse=True)
    team_wide, seen = [], set()
    for c in overall:
        if c["species"] not in seen:
            team_wide.append({**dex.brief(c), "checks": totals[c["name"]]["checks"],
                              "score": round(totals[c["name"]]["sum"], 2)})
            seen.add(c["species"])
        if len(team_wide) == n:
            break

    # Attacking types that hit several opponents super-effectively.
    type_pressure = []
    for t in TYPES:
        hit = [o["display"] for o in opps if dex.chart.eff_vs(t, o) >= 2]
        walled = [o["display"] for o in opps if dex.chart.eff_vs(t, o) < 1]
        if hit:
            type_pressure.append({"type": t, "super_effective_on": hit, "resisted_by": walled})
    type_pressure.sort(key=lambda x: (len(x["super_effective_on"]) - len(x["resisted_by"])), reverse=True)

    result = {
        "opponents": [dex.brief(o) for o in opps],
        "per_opponent": per_opp,
        "team_wide": team_wide,
        "best_attacking_types": type_pressure[:6],
        "opponent_stab_types": sorted({t for o in opps for t in o["types"]}),
    }

    if my_team:
        mine = [dex.pokemon[x] for x in my_team]
        plan, uncovered = {}, []
        for o in opps:
            rows = sorted(((p, matchup(dex, p, o)) for p in mine), key=lambda r: r[1]["score"], reverse=True)
            plan[o["display"]] = [{"pokemon": p["display"], **m} for p, m in rows[:2]]
            if not any(m["hits_se"] and m["safe"] for _, m in rows):
                uncovered.append(o["display"])
        result["my_team_plan"] = plan
        result["no_safe_answer"] = uncovered
    return result
