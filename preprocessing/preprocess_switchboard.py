import re
import unicodedata
import os
from tqdm import tqdm

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data_folder = os.getenv('DATA_FOLDER')
# File paths
input_file = f'data/{data_folder}/switchboard.train'
output_file = f'data/{data_folder}_cleaned/switchboard_preprocessed.train'
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Read, clean, and write
with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    for line in tqdm(infile, desc="Processing Simple Wikipedia"):
        stripped = line.strip().lower()

        # Skip empty lines and title markers
        if not stripped or re.match(r"^=+\s*.*?\s*=+$", stripped):
            continue
        
        # Filter out URLs
        if re.search(r'(http[s]?://|www\.)\S+', stripped):
            continue

        cleaned = preprocess_text(stripped)
        
        if cleaned:
            outfile.write(cleaned + '\n')

print(f"✅ Finished preprocessing. Saved to {output_file}")
