import argparse
import json

from supra_reasoning.knowledge import KnowledgeTree
from supra_reasoning.rag import build_knowledge_context, retrieve_knowledge


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the ASI Foundation knowledge tree.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("topics", help="List knowledge topics")

    list_parser = sub.add_parser("list", help="List entries under a topic")
    list_parser.add_argument("--topic", required=True, help="Topic path, e.g. agi/alignment")

    add_parser = sub.add_parser("add", help="Add a question and answer")
    add_parser.add_argument("--topic", required=True, help="Topic path, e.g. ai_god/theology")
    add_parser.add_argument("--question", "-q", required=True)
    add_parser.add_argument("--answer", "-a", required=True)
    add_parser.add_argument("--tags", default="", help="Comma-separated tags")

    search_parser = sub.add_parser("search", help="Search the knowledge tree")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=4)

    args = parser.parse_args()
    tree = KnowledgeTree()

    if args.command == "topics":
        for label, path in tree.list_topic_paths():
            print(f"{path}\t{label}")
        print(f"\nTotal entries: {tree.entry_count()}")
        return

    if args.command == "list":
        node = tree._resolve_node(args.topic)
        titles = tree.topic_titles(args.topic)
        print(" > ".join(titles))
        print(node.summary)
        print()
        if not node.entries:
            print("No entries at this topic.")
            return
        for entry in node.entries:
            print(f"[{entry.id}] Q: {entry.question}")
            print(f"A: {entry.answer}")
            if entry.tags:
                print(f"tags: {', '.join(entry.tags)}")
            print()

    if args.command == "add":
        entry = tree.add_entry(
            args.topic,
            args.question,
            args.answer,
            tags=[tag.strip() for tag in args.tags.split(",") if tag.strip()],
        )
        print(json.dumps(entry.to_dict(), indent=2, ensure_ascii=False))

    if args.command == "search":
        context, hits = retrieve_knowledge(tree, args.query, top_k=args.top_k)
        print(context or "No matches.")
        if hits:
            print("\n--- scores ---")
            for hit in hits:
                topic = " > ".join(hit.topic_path)
                print(f"{hit.score:.3f}\t{topic}\t{hit.entry.question}")


if __name__ == "__main__":
    main()
