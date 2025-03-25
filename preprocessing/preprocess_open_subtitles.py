import re
import unicodedata
import os

# Load dataset
# List of input training files to be preprocessed
input_files = [
    'text_data/train_100M/switchboard.train',
    'text_data/train_100M/simple_wiki.train',
    'text_data/train_100M/open_subtitles.train',
    'text_data/train_100M/gutenberg.train'
]

# Directory where preprocessed files will be saved
output_dir = 'preprocess'
os.makedirs(output_dir, exist_ok=True)

# Function to clean and normalize a single line of text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text) 

    # Remove segments like = = = PG21374 = = =
    text = re.sub(r'\s*=+\s*[^=]+?\s*=+\s*', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)  

    # Remove all punctuation/special characters except letters/numbers/spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)  

    # Limit repeated characters (e.g., loooove -> loove)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()  

    # Remove speaker identifiers (everything before and including the first colon)
    text = re.sub(r'^\s*[^:]+:\s*', '', text)

    return text

# Process each file in the input list
for input_file in input_files:

    # Extract the base filename and create a corresponding output file name
    filename = os.path.basename(input_file)
    name, ext = os.path.splitext(filename)
    output_file = os.path.join(output_dir, f"{name}_preprocessed{ext}")

    # Open input and output files
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        # Process each line in the input file
        for line in infile:
                cleaned_line = preprocess_text(line)
                if cleaned_line:
                    outfile.write(cleaned_line + '\n')

    # Log progress to console
    print(f"Processed {input_file} -> {output_file}")
