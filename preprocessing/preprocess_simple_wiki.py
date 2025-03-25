import os
import re
import unicodedata
from tqdm import tqdm


# Preprocessing Functions
def preprocess_text(text):
    text = text.lower()  # lowercase
    text = unicodedata.normalize("NFKC", text)  # normalize unicode
    text = re.sub(r'[\(\[\{][^)\]\}]*[\)\]\}]', '', text) # Remove text inside (), [], {}
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation/special characters
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text


# Input/Output Paths
input_file = 'text_data/train_100M/simple_wiki.train'
output_file = 'preprocess/simple_wiki_preprocessed.train'
os.makedirs(os.path.dirname(output_file), exist_ok=True)


# Processing
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in tqdm(infile, desc="Processing Simple Wikipedia"):
        stripped = line.strip().lower()

        # Skip empty lines
        if not stripped:
            continue

        # Skip title lines like: = = = Article Title = = =
        if re.match(r"^=+\s*.*?\s*=+$", stripped):
            continue

        # Remove URLs
        if re.search(r'(http[s]?://|www\.)\S+', stripped):
            continue

        cleaned = preprocess_text(stripped)

        # Skip after final cleaning if still empty
        if not cleaned:
            continue

        outfile.write(cleaned + '\n')

print(f"✅ Finished preprocessing. Saved to {output_file}")
