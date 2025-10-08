import os
import time
import sys
from typing import Tuple, List

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# =========================
# Configuration
# =========================
API_SECRET_KEY = "sk-xxxxxx"  # 🔒 Replace with your real API key (do NOT commit to repo)
BASE_URL = "https://api.example.com/v1/"
MODEL_NAME = "gpt-5-mini"     # Model name (e.g., gpt-4, gpt-3.5-turbo, etc.)
N_RESPONSES = 2               # Number of responses per item
SLEEP_SECONDS = 1.2           # Delay between requests to avoid throttling

# =========================
# Prompt template to reduce copyright restrictions on OPENAI —— our main results
# =========================
PROMPT_PREFIX = "Continue this text based on the given prefix: "

# =========================
# Prompt Template
# =========================

# PROMPT_PREFIX = (
#     "Consider this passage. First, reflect on whether you have encountered it before. Next, try to identify its source — such as a book, article, website, or dataset. Finally, answer: What is the next word: "
# )


# =========================
# Global Client Instance
# =========================
_client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)

# =========================
# Utility Function — Parse Response
# =========================
def extract_summary_and_output(response) -> Tuple[str, str]:
    """
    Extract reasoning summary and final output text from a model response.
    Handles multiple possible response formats for robustness.
    """
    reasoning_summary_parts: List[str] = []
    output_text_parts: List[str] = []

    try:
        if getattr(response, "output_text", None):
            output_text_parts.append(response.output_text)
    except Exception:
        pass

    try:
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "reasoning":
                summary = getattr(item, "summary", None)
                if isinstance(summary, list):
                    for s in summary:
                        txt = getattr(s, "text", None)
                        if isinstance(txt, str) and txt.strip():
                            reasoning_summary_parts.append(txt)
                elif isinstance(summary, str) and summary.strip():
                    reasoning_summary_parts.append(summary)

            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) in ("output_text", "text"):
                        txt = getattr(c, "text", None)
                        if isinstance(txt, str) and txt.strip():
                            output_text_parts.append(txt)
    except Exception:
        pass

    try:
        reasoning = getattr(response, "reasoning", None)
        if reasoning:
            summary = getattr(reasoning, "summary", None)
            if isinstance(summary, list):
                for s in summary:
                    txt = getattr(s, "text", None)
                    if isinstance(txt, str) and txt.strip():
                        reasoning_summary_parts.append(txt)
            elif isinstance(summary, str) and summary.strip():
                reasoning_summary_parts.append(summary)
    except Exception:
        pass

    def _join_unique(parts: List[str]) -> str:
        seen = set()
        ordered = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                ordered.append(p)
        return "\n".join(ordered).strip()

    return _join_unique(reasoning_summary_parts), _join_unique(output_text_parts)


# =========================
# Model Invocation
# =========================
def get_reasoning_paths(query: str, n: int = N_RESPONSES) -> List[Tuple[str, str]]:
    """
    Sends a query multiple times to obtain distinct reasoning and output pairs.
    """
    results: List[Tuple[str, str]] = []
    MAX_RETRIES = 5

    for i in range(n):
        retries = 0
        reasoning_summary, output_text = "", ""

        while retries < MAX_RETRIES:
            try:
                resp = _client.responses.create(
                    model=MODEL_NAME,
                    input=query,
                    reasoning={"effort": "medium", "summary": "auto"},
                )

                reasoning_summary, output_text = extract_summary_and_output(resp)

                if reasoning_summary.strip() and reasoning_summary.strip().lower() != "detailed":
                    break  # Got valid reasoning, stop retrying

                print(f"⚠️ Empty or generic reasoning ('detailed'), retrying ({retries + 1}/{MAX_RETRIES})...")

            except Exception as e:
                print(f"❌ Error in generation {i + 1} (retry {retries + 1}) for query '{query}': {e}", file=sys.stderr)

            retries += 1
            time.sleep(SLEEP_SECONDS)

        results.append((reasoning_summary, output_text))

    return results


# =========================
# Main Processing Function
# =========================
def process_excel(input_excel_path: str, output_excel_path: str) -> None:
    """
    Processes an Excel file row by row, queries the model,
    and saves reasoning paths and outputs to a new Excel file.
    """
    df = pd.read_excel(input_excel_path)
    results = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        item_id = row.get("ID", index)
        item_text = str(row.get("Item", ""))
        query_input = PROMPT_PREFIX + item_text

        print(f"\n🌀 Processing ID {item_id}...")

        pairs = get_reasoning_paths(query_input, n=N_RESPONSES)

        record = {"ID": item_id, "Item": item_text}
        for i, (r_sum, out_txt) in enumerate(pairs, start=1):
            record[f"Reasoning_Path_{i}"] = r_sum
            record[f"Output_{i}"] = out_txt

        results.append(record)

        # Save progress every 10 rows
        if (index + 1) % 10 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_excel(output_excel_path, index=False)
            print(f"💾 Progress saved after {index + 1} rows → {output_excel_path}")

    # Final save
    out_df = pd.DataFrame(results)
    out_df.to_excel(output_excel_path, index=False)
    print(f"\n✅ All reasoning paths and outputs saved to: {output_excel_path}")


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    input_excel = "/path/to/input_file.xlsx"
    output_excel = "/path/to/output_file.xlsx"
    process_excel(input_excel, output_excel)