from openai import OpenAI
import pandas as pd
import time
from tqdm import tqdm

# ====================================
# Configuration
# ====================================
API_SECRET_KEY = "sk-xxxxxx"  # 🔒 Replace with your real API key
BASE_URL = "https://api.example.com/v1/"
MODEL_NAME = "gpt-3.5-turbo"  # Example model name

# Initialize OpenAI client
client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)

# ====================================
# Load input data
# ====================================
input_path = "/path/to/input_file.xlsx"
print(f"📂 Loading input file: {input_path}")
df = pd.read_excel(input_path)
texts = df["Reasoning_path_1"].astype(str)

# ====================================
# Prepare storage
# ====================================
uncertainty_scores = []

# ====================================
# Prompt builder
# ====================================
def build_prompt(reasoning_text: str):
    """
    Build the evaluation prompt for uncertainty scoring.
    """
    return [
        {
            "role": "system",
            "content": "You are an assistant that evaluates the uncertainty expressed in reasoning text.",
        },
        {
            "role": "user",
            "content": (
                "Given the following reasoning path, rate how uncertain or speculative "
                "the language sounds on a scale from 0 (very certain and factual) "
                "to 1 (very uncertain and speculative).\n\n"
                "Respond only with a number between 0 and 1.\n\n"
                f'Text:\n"""{reasoning_text}"""'
            ),
        },
    ]

# ====================================
# Main processing loop
# ====================================
for text in tqdm(texts, desc="Scoring via Proxy LLM"):
    try:
        messages = build_prompt(text)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0
        )

        reply = response.choices[0].message.content.strip()
        print(f"Model reply: {reply}")

        try:
            score = float(reply)
            score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
        except Exception:
            print(f"⚠️ Could not parse model reply: {reply}")
            score = None

    except Exception as e:
        print(f"❌ Request failed: {e}")
        score = None

    uncertainty_scores.append(score)
    time.sleep(1.0)  # Prevent rate limit issues

# ====================================
# Save results
# ====================================
df["LLM_Judge_Uncertainty"] = uncertainty_scores

output_path = "/path/to/output_file.xlsx"
df.to_excel(output_path, index=False)
print(f"✅ Processing complete. Results saved to: {output_path}")