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
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in tqdm(infile, desc="Processing Switchboard"):
        line = line.strip().lower()

        # Skip empty lines
        if not line:
            continue

        # Skip lines containing URLs
        if re.search(r'(http[s]?://|www\.)\S+', line):
            continue

        # Remove speaker tags like 'a:', 'b:'
        line = re.sub(r'^[a-z]:\s*', '', line)

        # Final cleaning
        cleaned = preprocess_text(line)

        if cleaned:
            outfile.write(cleaned + '\n')

print(f"✅ Finished preprocessing: {input_file} → {output_file}")


## Merge consecutive speeches by the same speaker

# # Preprocessing function
# def preprocess_text(text):
#     text = text.lower()
#     text = unicodedata.normalize("NFKC", text)
#     text = re.sub(r'(http[s]?://|www\.)\S+', '', text)  # Remove URLs
#     text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove punctuation/special chars
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text


# # Main processing
# previous_speaker = None
# buffer = []
# seen_lines = set()

# with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
#     for line in tqdm(infile, desc="Merging Switchboard by speaker"):
#         line = line.strip()

#         if not line:
#             continue  # Skip empty lines

#         match = re.match(r'^([a-z]):\s*(.*)', line, re.IGNORECASE)
#         if not match:
#             continue  # Skip malformed lines

#         speaker, utterance = match.groups()
#         utterance = preprocess_text(utterance)

#         if not utterance:
#             continue

#         # If same speaker, keep adding to buffer
#         if speaker == previous_speaker:
#             buffer.append(utterance)
#         else:
#             # Write previous buffer if not empty
#             if buffer:
#                 merged = ' '.join(buffer)
#                 if merged and merged not in seen_lines:
#                     outfile.write(merged + '\n')
#                     seen_lines.add(merged)
#             # Start new buffer
#             buffer = [utterance]
#             previous_speaker = speaker

#     # Write remaining buffer
#     if buffer:
#         merged = ' '.join(buffer)
#         if merged and merged not in seen_lines:
#             outfile.write(merged + '\n')
