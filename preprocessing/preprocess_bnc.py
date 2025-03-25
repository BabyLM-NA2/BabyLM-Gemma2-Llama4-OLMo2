import os
import re
import unicodedata


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


input_file = 'text_data/train_100M/bnc_spoken.train'
output_dir = 'preprocess'
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, 'bnc_spoken_preprocessed.train')


with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        cleaned = preprocess_text(line)
        if cleaned:
            outfile.write(cleaned + '\n')

print(f"Finished preprocessing: {input_file} -> {output_file}")
