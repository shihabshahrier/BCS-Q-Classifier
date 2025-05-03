import os
import json
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any
from tqdm import tqdm  # For progress bar

load_dotenv()

# ── CONFIG ──────────────────────────────────────────────────────────────────────
INPUT_CSV = "bcs_10_to_43.csv"
OUTPUT_CSV = "bcs_with_categories.csv"
CHUNK_SIZE = 25  
MAX_RETRIES = 3  
RETRY_DELAY = 2  

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("NEBIUS_API_KEY")
)

# Using DeepSeek V3 for JSON classification
MODEL = "deepseek-ai/DeepSeek-V3-0324"

# Categories for classification
CATEGORIES = [
    "Bengali",
    "English",
    "Bangladesh Affairs",
    "International Affairs",
    "General Science & Tech",
    "Computer & IT",
    "Math Reasoning",
    "Mental Ability", 
    "Ethics & Good Governance",
    "Geography"
]

# Enhanced system prompt with examples and explicit instructions
SYSTEM_PROMPT = f"""
You are a precise JSON-only classifier. For each BCS preliminary MCQ, assign exactly one category from this list:
{', '.join(CATEGORIES)}

Your response must be a valid JSON array of objects with this exact structure:
[
  {{"row": 0, "category": "CATEGORY_NAME"}},
  {{"row": 1, "category": "CATEGORY_NAME"}},
  ...
]

- "row" must be the index (starting from 0) of the question in the provided list
- "category" must be exactly one of the allowed categories
- Do not include any explanations or additional text outside the JSON array
- Ensure your output is valid, parseable JSON

Examples:
1. A question about Bangladesh history -> "Bangladesh Affairs"
2. A question about grammar or vocabulary -> "English"
3. A question about mathematical problem solving -> "Math Reasoning"
""".strip()


def classify_questions(questions: List[str]) -> List[Dict[str, Any]]:
    """
    Classify a batch of questions using the LLM API.
    
    Args:
        questions: List of question strings to classify
        
    Returns:
        List of classification objects with row and category
    """
    questions_block = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=0))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": questions_block}
    ]
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=1024,  # Increased for longer responses
                temperature=0.0,  # Keep deterministic for classification
                response_format={"type": "json_object"}  # Request JSON format if supported
            )
            
            raw = resp.choices[0].message.content.strip()
            
            # Handle potential JSON array vs JSON object wrapping
            try:
                # Try parsing as direct JSON array first
                classification = json.loads(raw)
                if not isinstance(classification, list):
                    # If it's a JSON object with a results field, extract it
                    if isinstance(classification, dict) and "results" in classification:
                        classification = classification["results"]
                    else:
                        raise ValueError(f"Unexpected JSON structure: {raw[:100]}...")
                return classification
                
            except json.JSONDecodeError:
                # Try to extract JSON from text (in case model adds explanations)
                import re
                json_match = re.search(r'\[\s*{.*}\s*\]', raw, re.DOTALL)
                if json_match:
                    classification = json.loads(json_match.group(0))
                    return classification
                raise  # Re-raise if extraction failed
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[WARN] Classification error (attempt {attempt}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES:
                print(f"  -> Waiting {RETRY_DELAY}s before retry...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  -> All retries failed. Raw response: {raw[:200]}...")
                return []  # Return empty list after all retries fail
                
        except Exception as e:
            print(f"[ERROR] Unexpected error: {str(e)}")
            if attempt < MAX_RETRIES:
                print(f"  -> Waiting {RETRY_DELAY}s before retry...")
                time.sleep(RETRY_DELAY)
            else:
                return []


def main():
    print(f"🔄 Starting BCS question classification process...")
    
    # Load and prepare data
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"📊 Loaded {len(df)} questions from {INPUT_CSV}")
    except Exception as e:
        print(f"❌ Failed to load input CSV: {str(e)}")
        return
    
    # Add or reset category column
    df["category"] = ""
    
    # Calculate total chunks for progress tracking
    total_chunks = (len(df) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    # Process in chunks with progress bar
    with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
        for start in range(0, len(df), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(df))
            chunk = df.iloc[start:end]
            
            # Get classifications for this chunk
            classifications = classify_questions(chunk["question"].tolist())
            
            # Apply categories to dataframe
            for item in classifications:
                try:
                    row_idx = start + int(item["row"])
                    if row_idx < len(df):
                        category = item["category"]
                        # Validate category is in allowed list
                        if category in CATEGORIES:
                            df.at[row_idx, "category"] = category
                        else:
                            print(f"[WARN] Invalid category '{category}' for row {row_idx}")
                except (KeyError, ValueError, IndexError) as e:
                    print(f"[ERROR] Issue with classification item: {item} - {str(e)}")
            
            # Save intermediate results every 5 chunks
            if (start // CHUNK_SIZE) % 5 == 0 and start > 0:
                df.to_csv(f"{OUTPUT_CSV}.partial", index=False, encoding="utf-8")
                print(f"💾 Saved intermediate results ({start}/{len(df)} processed)")
            
            pbar.update(1)
    
    # Count categorized items
    categorized = df[df["category"] != ""].shape[0]
    categorization_rate = (categorized / len(df)) * 100
    
    # Generate summary of categories
    category_counts = df["category"].value_counts().to_dict()
    
    # Save final results
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    
    # Print summary
    print("\n" + "="*50)
    print(f"✅ Classification complete!")
    print(f"📈 Stats: {categorized}/{len(df)} items classified ({categorization_rate:.1f}%)")
    print(f"📊 Category distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        if cat:  # Skip empty category
            print(f"   - {cat}: {count} ({(count/len(df))*100:.1f}%)")
    print(f"💾 Results saved to {OUTPUT_CSV}")
    print("="*50)


if __name__ == "__main__":
    main()