"""Command-line inference utility for the next-word prediction model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from next_word_prediction.generate import generate_text


def _post_process(text: str) -> str:
    """Simple cleanup: collapse whitespace, capitalize first char, ensure trailing period."""
    cleaned = " ".join(text.strip().split())
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
        if cleaned[-1].isalnum():
            cleaned += "."
    return cleaned


def run_inference(workspace: Path, seed_text: str, max_words: int, top_k: int, output_path: Path | None) -> None:
    result = generate_text(workspace, seed_text=seed_text, max_words=max_words, top_k=top_k)
    processed_text = _post_process(result.generated_text)

    print("Seed:", seed_text)
    print("Generated:")
    print(processed_text)
    print()
    print(f"Top-{top_k} options for the first step:")
    if result.steps:
        for token, prob in result.steps[0]["top_k"]:
            print(f"  {token:>12} : {prob:.4f}")

    if output_path:
        payload = {
            "seed_text": seed_text,
            "generated_text": processed_text,
            "num_generated_tokens": len(processed_text.split()),
            "steps": result.steps,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved full details to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."), help="Workspace containing artifacts/")
    parser.add_argument("--seed", required=True, help="Seed text for generation.")
    parser.add_argument("--max-words", type=int, default=30, help="Number of new tokens to generate.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions to track per step.")
    parser.add_argument("--output", type=Path, help="Optional path to save JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(
        workspace=args.workspace,
        seed_text=args.seed,
        max_words=args.max_words,
        top_k=args.top_k,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
