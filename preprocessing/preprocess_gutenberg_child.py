import os
import re
import unicodedata
from tqdm import tqdm

# Preprocessing function 
def preprocess_text(text):
    # Normalize text and preserve basic punctuation
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)  # Normalize Unicode characters
    # Keep basic punctuation: . , ! ? ' and spaces
    text = re.sub(r'[^a-z0-9\s.,!?\']', '', text)  # Modified regex
    text = re.sub(r'\s+', ' ', text).strip()  # Collapse multiple whitespaces
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
data_folder = os.getenv('DATA_FOLDER')
input_file = f'data/{data_folder}/gutenberg.train' if data_folder != 'dev' else f'data/{data_folder}/gutenberg.dev'
output_dir = f'data/{data_folder}_cleaned'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'gutenberg.train') if data_folder != 'dev' else os.path.join(output_dir, 'gutenberg.dev')

seen_lines = set()

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    for line in tqdm(infile, desc="Processing Gutenberg"):
        line = line.strip().lower()

        # Skip metadata lines
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

        # Write valid lines directly without duplicate checking
        if cleaned:
            outfile.write(cleaned + '\n')

print(f"✅ Finished preprocessing: {input_file} → {output_file}")