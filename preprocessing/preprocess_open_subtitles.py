import re
import unicodedata
import os


data_folder = os.getenv('DATA_FOLDER')

# Load dataset
input_files = [f'data/{data_folder}open_subtitles.train']
output_dir = f'data/{data_folder}_cleaned'
os.makedirs(output_dir, exist_ok=True)

# Clean and normalize text while preserving basic punctuation
def preprocess_text(text):
    
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)  # Normalize Unicode characters
    
    # Preserve basic punctuation: . , ! ? '
    text = re.sub(r'[^a-z0-9\s.,!?\']', '', text)  # Modified regex
    
    # Preserve essential formatting
    text = re.sub(r'\s*=+\s*[^=]+?\s*=+\s*', ' ', text)  # Remove metadata markers
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # Limit repeated characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    text = re.sub(r'^\s*[^:]+:\s*', '', text)  # Remove speaker identifiers
    
    return text

# Process each file in the input list
for input_file in input_files:
    filename = os.path.basename(input_file)
    name, ext = os.path.splitext(filename)
    output_file = os.path.join(output_dir, f"{name}_preprocessed{ext}")

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            cleaned_line = preprocess_text(line)
            if cleaned_line:
                outfile.write(cleaned_line + '\n')

    print(f"Processed {input_file} -> {output_file}")