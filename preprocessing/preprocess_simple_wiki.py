import os
import re
import unicodedata
from tqdm import tqdm


# Preprocessing Functions
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = unicodedata.normalize("NFKC", text)  # Normalize unicode characters
    # Preserve basic punctuation: . , ! ? '
    text = re.sub(r'[^a-z0-9\s.,!?\']', '', text)  # Modified regex
    text = re.sub(r'\s+', ' ', text).strip()  # Collapse whitespace
    return text


data_folder = os.getenv('DATA_FOLDER')
# Input/Output Paths
input_file = f'data/{data_folder}/simple_wiki.train'
output_file = f'data/{data_folder}_cleaned/simple_wiki_preprocessed.train'
os.makedirs(os.path.dirname(output_file), exist_ok=True)


# Processing
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