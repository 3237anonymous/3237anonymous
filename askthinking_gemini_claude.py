import os
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# === Configuration ===
API_SECRET_KEY = "sk-xxxxxx"  # 🔒 Replace with your actual API key
BASE_URL = "https://api.example.com/v1/"
MODEL_NAME = "claude-sonnet-4-thinking" # or "gemini-2.5-flash-preview-05-20-thinking"
N_RESPONSES = 3
SLEEP_SECONDS = 1.2


# === Generate multiple reasoning paths and outputs ===
def get_reasoning_paths(query, n=N_RESPONSES):
    """
    Sends a query to the model multiple times to obtain different reasoning paths and outputs.
    """
    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
    results = []

    attempt = 0
    while len(results) < n:
        attempt += 1
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": (
                            f"Consider this passage. "
                            f"First, reflect on whether you have encountered it before. "
                            f"Next, try to identify its source — such as a book, article, website, or dataset. "
                            f"Finally, answer: What is the next word: {query}"
                        ),
                    },
                ],
            )

            reasoning_path = getattr(response.choices[0].message, "reasoning_content", "").strip()
            output = response.choices[0].message.content.strip()

            if reasoning_path and output:
                print(f"\n--- Generation {len(results) + 1} (Attempt {attempt}) ---")
                results.append((reasoning_path, output))
            else:
                print(f"⚠️ Incomplete result in attempt {attempt}, retrying...")

        except Exception as e:
            print(f"❌ Error in attempt {attempt} for query '{query}': {e}")

        time.sleep(SLEEP_SECONDS)

    return results


# === Main processing function ===
def process_excel(input_path, output_path):
    """
    Processes an Excel file, sends each row's content to the model,
    and stores reasoning and output results.
    """
    df = pd.read_excel(input_path)
    results = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        item_id = row.get("ID", index)
        item_text = str(row.get("Item", ""))

        print(f"\n🌀 Processing ID {item_id}...")

        responses = get_reasoning_paths(item_text, n=N_RESPONSES)

        result_entry = {
            "ID": item_id,
            "Item": item_text,
        }

        for i, (reasoning, output) in enumerate(responses, start=1):
            result_entry[f"Reasoning_Path_{i}"] = reasoning
            result_entry[f"Output_{i}"] = output

        results.append(result_entry)

        # === Save progress every 10 items ===
        if (index + 1) % 10 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_excel(output_path, index=False)
            print(f"💾 Saved first {index + 1} entries to {output_path}")

    # === Final save ===
    result_df = pd.DataFrame(results)
    result_df.to_excel(output_path, index=False)
    print(f"\n✅ All reasoning paths and outputs saved to: {output_path}")


# === Entry point ===
if __name__ == "__main__":
    input_excel = "/path/to/input_file.xlsx"
    output_excel = "/path/to/output_file.xlsx"
    process_excel(input_excel, output_excel)