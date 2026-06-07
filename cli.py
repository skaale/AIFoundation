import argparse
import warnings

warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated",
    category=FutureWarning,
)

from supra_reasoning.knowledge import KnowledgeTree
from supra_reasoning.model import SupraReasoningModel
from supra_reasoning.rag import retrieve_knowledge


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Supra-50M-Reasoning locally.")
    parser.add_argument("question", nargs="?", help="Question to ask the model")
    parser.add_argument("--max-new-tokens", type=int, default=992)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--hide-thinking", action="store_true")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run on (default: auto-detect GPU)",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable knowledge-tree retrieval",
    )
    args = parser.parse_args()

    question = args.question
    if not question:
        question = input("Question: ").strip()
    if not question:
        raise SystemExit("No question provided.")

    print("Loading model…")
    engine = SupraReasoningModel(device=args.device)
    print(f"Running on {engine.torch_device} ({engine.dtype}).")

    knowledge_context = None
    if not args.no_rag:
        tree = KnowledgeTree()
        knowledge_context, hits = retrieve_knowledge(tree, question, top_k=4)
        if hits:
            print(f"Knowledge tree: {len(hits)} retrieval hit(s).")
        else:
            print("Knowledge tree: no retrieval hits.")

    print("Generating…\n")

    thought = ""
    answer = ""
    for update in engine.generate_stream(
        question,
        knowledge_context=knowledge_context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    ):
        thought = update["thought"]
        answer = update["answer"]

    if not args.hide_thinking and thought:
        print("=== Thinking ===")
        print(thought)
        print()

    print("=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
