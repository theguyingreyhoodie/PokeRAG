"""Evaluate the RAG system.

Retrieval eval (default, no LLM needed): for each question in retrieval.jsonl, does the right
Pokémon/ability/move/region come back? Compares BM25, vector and hybrid search with Hit@1,
Hit@5 and MRR. The questions paraphrase Pokédex entries and never name the answer.

End-to-end eval (--e2e): asks the full pipeline questions from e2e.jsonl and checks the
generated answer contains the expected facts. Needs the configured LLM to be running.

    python -m eval.run_eval
    python -m eval.run_eval --e2e
"""

import argparse
import json
import time
from pathlib import Path

from pokedex_rag.retrieve import hybrid_search

HERE = Path(__file__).parent


def load(name):
    return [json.loads(line) for line in (HERE / name).read_text().splitlines() if line.strip()]


def retrieval_eval(k=5):
    rows = load("retrieval.jsonl")
    results = {}
    for mode in ("bm25", "vector", "hybrid", "rerank"):
        hit1 = hitk = rr = 0.0
        misses = []
        for row in rows:
            got = [f"{kind}:{ref}" for kind, ref in hybrid_search(row["question"], k=k, mode=mode)]
            ranks = [got.index(e) + 1 for e in row["expected"] if e in got]
            best = min(ranks) if ranks else None
            hit1 += best == 1
            hitk += best is not None
            rr += 1 / best if best else 0
            if best is None:
                misses.append((row["question"], got[:3]))
        n = len(rows)
        results[mode] = {"hit@1": hit1 / n, f"hit@{k}": hitk / n, "mrr": rr / n, "misses": misses}

    print(f"\nRetrieval eval: {len(rows)} questions\n")
    print(f"{'mode':<8} {'Hit@1':>7} {f'Hit@{k}':>7} {'MRR':>7}")
    for mode, r in results.items():
        print(f"{mode:<8} {r['hit@1']:>7.1%} {r[f'hit@{k}']:>7.1%} {r['mrr']:>7.3f}")
    if results["rerank"]["misses"]:
        print("\nMisses (rerank):")
        for q, got in results["rerank"]["misses"]:
            print(f"  - {q}\n      got: {', '.join(got)}")
    return results


def e2e_eval():
    from pokedex_rag.rag import answer_text

    rows = load("e2e.jsonl")
    passed = 0
    print(f"\nEnd-to-end eval: {len(rows)} questions\n")
    for row in rows:
        t = time.time()
        text, _ = answer_text(row["question"])
        low = text.lower()
        ok = all(any(alt.lower() in low for alt in group.split("|")) for group in row["must_include"])
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] ({time.time() - t:4.1f}s) {row['question']}")
        if not ok:
            print(f"        expected {row['must_include']}; got: {text[:200]!r}")
    print(f"\n{passed}/{len(rows)} passed ({passed / len(rows):.0%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", action="store_true", help="also run the end-to-end answer eval (needs an LLM)")
    args = ap.parse_args()
    retrieval_eval()
    if args.e2e:
        e2e_eval()
