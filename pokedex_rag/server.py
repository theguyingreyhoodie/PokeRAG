"""FastAPI app: streaming chat endpoint, team tools, and the static web UI."""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import llm, strategy
from .config import ROOT
from .rag import answer
from .store import get_pokedex

WEB = ROOT / "web"


@asynccontextmanager
async def lifespan(app):
    get_pokedex()  # load the dataset once at startup
    yield


app = FastAPI(title="PokéRAG", lifespan=lifespan)


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[dict] = []


class TeamBody(BaseModel):
    team: list[str] = Field(default=[], max_length=6)
    opponents: list[str] = Field(default=[], max_length=6)
    region: str | None = None
    allow_legendary: bool = False
    allow_mega: bool = False


def resolve_all(names):
    dex = get_pokedex()
    out, unknown = [], []
    for n in names:
        if not n.strip():
            continue
        r = dex.resolve(n)
        (out if r else unknown).append(r or n)
    if unknown:
        raise HTTPException(400, f"Unknown Pokémon: {', '.join(unknown)}")
    return out


@app.get("/api/health")
def health():
    dex = get_pokedex()
    return {"ok": True, "pokemon": len(dex.pokemon), **llm.describe()}


@app.get("/api/pokemon")
def search_pokemon(q: str = ""):
    dex = get_pokedex()
    return [dex.brief(p) for p in dex.search_names(q)] if q else []


@app.get("/api/pokemon/{name}")
def get_pokemon(name: str):
    dex = get_pokedex()
    key = dex.resolve(name)
    if not key:
        raise HTTPException(404, "Not found")
    p = dex.pokemon[key]
    return {k: v for k, v in p.items() if k != "moves"} | {"card": dex.card(key)}


@app.post("/api/ask")
def ask(body: AskBody):
    def events():
        for ev in answer(body.question, body.history):
            yield json.dumps(ev) + "\n"
    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/api/team/analyze")
def team_analyze(body: TeamBody):
    dex = get_pokedex()
    team = resolve_all(body.team)
    if not team:
        raise HTTPException(400, "Add at least one Pokémon.")
    filters = {"region": body.region, "allow_legendary": body.allow_legendary, "allow_mega": body.allow_mega}
    analysis = strategy.analyze_team(dex, team)
    suggestions = strategy.suggest_members(dex, team, n=6, **filters) if len(team) < 6 else []
    return {"analysis": analysis, "suggestions": suggestions}


@app.post("/api/team/build")
def team_build(body: TeamBody):
    dex = get_pokedex()
    filters = {"region": body.region, "allow_legendary": body.allow_legendary, "allow_mega": body.allow_mega}
    b = strategy.build_team(dex, resolve_all(body.team), **filters)
    return {"team": [dex.brief(dex.pokemon[n]) for n in b["team"]], "added": b["added"], "analysis": b.get("analysis")}


@app.post("/api/counters")
def counters(body: TeamBody):
    dex = get_pokedex()
    opps = resolve_all(body.opponents)
    if not opps:
        raise HTTPException(400, "Add at least one opposing Pokémon.")
    filters = {"region": body.region, "allow_legendary": body.allow_legendary, "allow_mega": body.allow_mega}
    return strategy.counters(dex, opps, my_team=resolve_all(body.team) or None, **filters)


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")
