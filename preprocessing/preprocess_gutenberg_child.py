import os
import re
import unicodedata
from tqdm import tqdm

# This program could cost hours to finish the preprocessing
# Preprocessing function 
def preprocess_text(text):
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)  # Normalize Unicode characters
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove all punctuation/special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Remove Gutenberg-specific metadata and noise
def clean_gutenberg_text(text):
    text = re.sub(r'^.*?\*\*\* START OF(.*?)\*\*\*', '', text, flags=re.S | re.I)
    text = re.sub(r'\*\*\* END OF(.*?)$', '', text, flags=re.S | re.I)
    text = re.sub(r'\[illustration[^\]]*\]', '', text, flags=re.I)
    text = re.sub(r'^(?:[A-Z][A-Z\s,\.\'-]{4,}|copyright.*|_.*?_)\s*$', '', text, flags=re.M | re.I)
    text = re.sub(r'contents\s*\(.*?\)\s*.*?(?=\n[a-zA-Z])', '', text, flags=re.S | re.I)
    text = re.sub(r'\bpage\s*\d+\b', '', text, flags=re.I)
    return text

# File paths
input_file = 'text_data/train_100M/gutenberg.train'
output_dir = 'preprocess'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'gutenberg_preprocessed.train')

seen_lines = set()

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in tqdm(infile, desc="Processing Gutenberg"):
        line = line.strip().lower()

        # Combined filter conditions for speed
        if (
            line.startswith('=') or
            re.search(r'(http[s]?://|www\.)\S+', line) or
            re.match(r'^(pg|page|p\.?)\s?\d{3,6}$', line) or
            re.match(r'^(chapter\s+)?[ivxlc]+\s+.+\s+\d+$', line) or
            re.match(r'^chapter\s+((\d+)|([ivxlc]+)|(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve))\b.*$', line) or
            re.match(r'^[a-z\s]+(\d+\s*){1,}$', line)
        ):
            continue

        # Clean and normalize
        cleaned = preprocess_text(clean_gutenberg_text(line))

        # Skip empty
        if not cleaned:
            continue

        # Efficient duplicate removal logic (keep only longest variation)
        word_count = len(cleaned.split())
        if 3 <= word_count <= 20:
            is_sub = any(cleaned in s or s in cleaned for s in seen_lines)
            to_remove = {s for s in seen_lines if s in cleaned or cleaned in s}
        else:
            is_sub = False
            to_remove = set()

        if not is_sub:
            seen_lines.difference_update(to_remove)
            outfile.write(cleaned + '\n')
            seen_lines.add(cleaned)

print(f"✅ Finished preprocessing: {input_file} → {output_file}")
