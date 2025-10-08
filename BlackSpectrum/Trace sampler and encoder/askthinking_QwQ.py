import os
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

# ====================================
# Configuration
# ====================================
API_SECRET_KEY = "sk-xxxxxx"  # 🔒 Replace with your actual API key
BASE_URL = "https://api.example.com/v1"
MODEL_NAME = "Qwen/QwQ-32B"   # Example: "deepseek-ai/DeepSeek-R1"
N_RESPONSES = 3               # Number of responses per item
SLEEP_SECONDS = 1.2           # Delay between API calls
BATCH_SIZE = 50               # Save results every N items
MAX_RETRIES_PER_RESPONSE = 10 # Retry limit per generation

# ====================================
# Utility: Clean invalid Excel characters
# ====================================
def safe_clean(value):
    """Remove characters incompatible with Excel."""
    try:
        if isinstance(value, str):
            value = ILLEGAL_CHARACTERS_RE.sub("", value)
            value = value.replace("\u2028", "").replace("\u2029", "")
            return value.strip()
        return value
    except Exception:
        return ""

# ====================================
# Generate multiple reasoning paths from model
# ====================================
def get_reasoning_paths(query, n=N_RESPONSES):
    """Generate N reasoning paths for a given query."""
    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
    results = []
    attempt_counts = [0] * n
    i = 0

    while i < n:
        if attempt_counts[i] >= MAX_RETRIES_PER_RESPONSE:
            print(f"Response {i+1} reached max retries, skipping.")
            results.append(("", ""))
            i += 1
            continue

        try:
            print(f"🚀 Generating response {i+1} from model...")
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": (
                            "Consider this passage. First, reflect on whether you have encountered it before. "
                            "Next, try to identify its source — such as a book, article, website, or dataset. "
                            f"Finally, answer: What is the next word: {query}"
                        ),
                    },
                ],
                stream=False,
            )

            choice = response.choices[0].message
            reasoning = getattr(choice, "reasoning_content", "").strip()
            output = choice.content.strip()

            if reasoning:
                results.append((reasoning, output))
                i += 1
            else:
                attempt_counts[i] += 1
                print(f"Response {i+1} missing reasoning; retrying ({attempt_counts[i]} attempt(s))")

        except Exception as e:
            attempt_counts[i] += 1
            print(f"Error (Response {i+1}, Attempt {attempt_counts[i]}): {e}")

        time.sleep(SLEEP_SECONDS)

    return results

# ====================================
# Main processing function
# ====================================
def process_excel(input_excel_path, output_excel_path):
    """Read input Excel, call model, and save outputs with reasoning paths."""
    try:
        df_input = pd.read_excel(input_excel_path)
    except Exception as e:
        print(f"Failed to read input file: {e}")
        return

    print(f"Loaded input data: {len(df_input)} rows")
    all_results = []
    new_batch = []

    for index, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Processing"):
        try:
            item_text = str(row.get("Item", "")).strip()
            item_id = str(row.get("ID", index))

            if not item_text:
                print(f"⚠️ Row {item_id} is empty, skipping.")
                continue

            print(f"\n🔍 Processing ID={item_id} | Preview: {item_text[:40]}")

            responses = get_reasoning_paths(item_text, n=N_RESPONSES)

            result_entry = {"ID": item_id, "Item": item_text}
            for j, (reasoning, output) in enumerate(responses, start=1):
                result_entry[f"Reasoning_Path_{j}"] = reasoning
                result_entry[f"Output_{j}"] = output

            cleaned_result = {k: safe_clean(v) for k, v in result_entry.items()}

            # Test write to detect Excel encoding issues early
            try:
                _ = pd.DataFrame([cleaned_result]).to_excel("/dev/null", engine="openpyxl")
            except Exception as e:
                print(f"⚠️ ID={item_id} contains invalid characters, skipped. Error: {e}")
                continue

            all_results.append(cleaned_result)
            new_batch.append(cleaned_result)

            # Save batch
            if len(new_batch) >= BATCH_SIZE:
                try:
                    df_batch = pd.DataFrame(all_results).applymap(safe_clean)
                    df_batch.to_excel(output_excel_path, index=False, engine="openpyxl")
                    print(f"Saved {len(all_results)} rows so far.")
                    new_batch = []
                except Exception as e:
                    print(f"Batch save failed: {e}")

        except Exception as e:
            print(f"⚠️ Skipping ID={item_id} due to processing error: {e}")
            continue

    # Final save
    if all_results:
        try:
            df_final = pd.DataFrame(all_results).applymap(safe_clean)
            df_final.to_excel(output_excel_path, index=False, engine="openpyxl")
            print(f"Final save complete: {len(all_results)} rows processed → {output_excel_path}")
        except Exception as e:
            print(f"Final save failed: {e}")

# ====================================
# Script entry point
# ====================================
if __name__ == "__main__":
    input_excel = "/path/to/input_file.xlsx"
    output_excel = "/path/to/output_file.xlsx"
    process_excel(input_excel, output_excel)