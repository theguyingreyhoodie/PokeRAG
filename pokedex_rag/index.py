"""Build and query the vector index.

Small-to-big retrieval: each Pokémon is embedded as two short chunks (a stats/profile chunk
and a lore chunk built from Pokédex entries). Search runs over the chunks, but what reaches
the LLM is the Pokémon's full card. Abilities, moves, types and regions get one chunk each.
"""

from functools import lru_cache

import chromadb

from .config import CHROMA_DIR, COLLECTION, EMBED_MODEL
from .store import get_pokedex


@lru_cache(maxsize=1)
def embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(EMBED_MODEL)


def chunks(dex):
    for p in dex.pokemon.values():
        types = "/".join(t.title() for t in p["types"])
        abil = ", ".join(f"{a['display']} ({a['effect']})" for a in p["abilities"])
        tags = " ".join(t for t, f in (("Legendary.", p["is_legendary"]), ("Mythical.", p["is_mythical"]),
                                       ("Mega Evolution.", p["is_mega"]), ("Baby Pokémon.", p["is_baby"])) if f)
        region = (p["origin_region"] or "").title()
        meta = {"kind": "pokemon", "ref": p["name"], "type1": p["types"][0],
                "type2": p["types"][1] if len(p["types"]) > 1 else "", "region": p["origin_region"] or ""}
        yield (f"pokemon:{p['name']}:profile",
               f"{p['display']}, the {p['genus']}. {types} type from the {region} region. {tags} "
               f"Abilities: {abil}. Evolution: {p['evolution']}", meta)
        if p["flavor"]:
            lore = " ".join(p["flavor"])
            extra = ", ".join(x for x in (p["habitat"] and f"lives in {p['habitat']} areas",
                                          p["color"] and f"{p['color']} colored") if x)
            yield (f"pokemon:{p['name']}:lore", f"{p['display']}, the {p['genus']} ({extra}). {lore}", meta)
    for a in dex.abilities.values():
        yield (f"ability:{a['name']}", f"{a['display']} is an ability. {a['effect']}",
               {"kind": "ability", "ref": a["name"], "type1": "", "type2": "", "region": ""})
    for m in dex.moves.values():
        power = f" with {m['power']} power" if m["power"] else ""
        yield (f"move:{m['name']}",
               f"{m['display']} is a {m['type'].title()}-type {m['damage_class']} move{power}. {m['effect']}",
               {"kind": "move", "ref": m["name"], "type1": m["type"], "type2": "", "region": ""})
    for t in dex.types:
        yield (f"type:{t}", dex.type_card(t), {"kind": "type", "ref": t, "type1": t, "type2": "", "region": ""})
    for r in dex.regions:
        yield (f"region:{r}", dex.region_card(r), {"kind": "region", "ref": r, "type1": "", "type2": "", "region": r})


def client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def build_index(batch=256):
    dex = get_pokedex()
    items = list(chunks(dex))
    c = client()
    if COLLECTION in [col.name for col in c.list_collections()]:
        c.delete_collection(COLLECTION)
    col = c.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    emb = embedder()
    print(f"Embedding {len(items)} chunks with {EMBED_MODEL}...")
    for i in range(0, len(items), batch):
        part = items[i:i + batch]
        vecs = [v.tolist() for v in emb.embed([t for _, t, _ in part])]
        col.add(ids=[x[0] for x in part], documents=[x[1] for x in part],
                metadatas=[x[2] for x in part], embeddings=vecs)
    print(f"Indexed {col.count()} chunks -> {CHROMA_DIR}")


@lru_cache(maxsize=1)
def collection():
    return client().get_collection(COLLECTION)


def search(query, k=8, where=None):
    """Semantic search; returns [(kind, ref, distance, chunk_text)] deduplicated by (kind, ref)."""
    vec = next(iter(embedder().query_embed(query))).tolist()
    res = collection().query(query_embeddings=[vec], n_results=k * 2, where=where)
    out, seen = [], set()
    for meta, dist, doc in zip(res["metadatas"][0], res["distances"][0], res["documents"][0]):
        key = (meta["kind"], meta["ref"])
        if key not in seen:
            seen.add(key)
            out.append((meta["kind"], meta["ref"], dist, doc))
        if len(out) == k:
            break
    return out


if __name__ == "__main__":
    build_index()
