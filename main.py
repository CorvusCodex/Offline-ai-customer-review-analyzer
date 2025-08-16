#!/usr/bin/env python3
"""
Customer Review Analyzer (offline)
Usage:
  python main.py --file reviews.txt
  python main.py --input "Great product! ..."
Input format: one review per line or full text.
"""
import argparse, requests, os, sys

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = "llama3.2:4b"
TIMEOUT = 300

def run_llama(prompt):
    r = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json().get("response","").strip()

def build_prompt(reviews_text):
    return (
        "You are an analyst that processes customer reviews.\n"
        "Input: multiple reviews (one per line).\n"
        "For each review output:\n"
        "Review: <original>\nSentiment: Positive|Negative|Neutral\nTop keywords: <comma separated 3 keywords>\nOne-line summary: <short>\n\n"
        f"REVIEWS:\n{reviews_text}\n\nRespond for each review in the format above."
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", "-f")
    p.add_argument("--input", "-i")
    args = p.parse_args()
    content = args.input or ""
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as fh:
                content = (content + "\n" if content else "") + fh.read()
        except Exception as e:
            print("Error:", e, file=sys.stderr); sys.exit(1)
    if not content.strip():
        print("Provide --input or --file", file=sys.stderr); sys.exit(1)
    prompt = build_prompt(content)
    print(run_llama(prompt))

if __name__ == "__main__":
    main()
