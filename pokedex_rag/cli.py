"""Command line entry point.

    python -m pokedex_rag.cli ingest          # download PokéAPI data, build dataset + vector index
    python -m pokedex_rag.cli ask "question"  # one-off question (add --debug to see the context)
    python -m pokedex_rag.cli chat            # interactive chat in the terminal
    python -m pokedex_rag.cli serve           # web UI at http://localhost:8000
"""

import argparse
import sys


def cmd_ingest(_):
    from .build import build
    from .fetch import main as fetch
    from .index import build_index
    fetch()
    build()
    build_index()


def print_answer(question, history, debug):
    from .rag import answer

    for ev in answer(question, history):
        if ev["type"] == "meta":
            u = ev["understood"]
            print(f"\033[2m[{u['intent']}] sources: {', '.join(u['sources'][:8])}\033[0m")
            if debug:
                print("\033[2m" + ev["context"] + "\033[0m\n")
        elif ev["type"] == "token":
            yield ev["text"]
            sys.stdout.write(ev["text"])
            sys.stdout.flush()
        elif ev["type"] == "error":
            print(f"\n[LLM error] {ev['message']}")
    print()


def cmd_ask(args):
    list(print_answer(args.question, None, args.debug))


def cmd_chat(args):
    history = []
    print("PokéRAG chat. Ctrl-C or empty line to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        text = "".join(print_answer(q, history, args.debug))
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": text}]


def cmd_serve(args):
    import uvicorn
    uvicorn.run("pokedex_rag.server:app", host=args.host, port=args.port)


def main():
    ap = argparse.ArgumentParser(prog="pokedex_rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").set_defaults(fn=cmd_ingest)
    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--debug", action="store_true")
    p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("chat")
    p.add_argument("--debug", action="store_true")
    p.set_defaults(fn=cmd_chat)
    p = sub.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
