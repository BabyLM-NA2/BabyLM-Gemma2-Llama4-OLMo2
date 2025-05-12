import os
import re

def preprocess_text(text):
    """Optimized text cleaning function"""
    # 1. Remove URLs and HTML tags
    text = re.sub(r'http\S+|www\S+|<[^>]+>', '', text)
    
    # 2. Keep letters, numbers, basic punctuation and necessary symbols (including apostrophes and hyphens)
    text = re.sub(r'[^\w\s.,!?\'\-]', '', text)
    
    # 3. Merge consecutive spaces but preserve repeated letters (removed the previous repeated letter handling)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def main():
    data_folder = os.getenv('DATA_FOLDER')
    input_file = f'data/{data_folder}/bnc_spoken.train' if data_folder != 'dev' else f'data/{data_folder}/bnc_spoken.dev'
    output_dir = f'data/{data_folder}_cleaned'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'bnc_spoken.train') if data_folder != 'dev' else os.path.join(output_dir, 'bnc_spoken.dev')

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                # Perform text cleaning
                cleaned = preprocess_text(line)
                
                if cleaned:
                    # Write cleaned text directly (no special tags added)
                    outfile.write(cleaned + '\n')
                    
    except FileNotFoundError:
        print(f"Error: Input file not found - {input_file}")
    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")

    print(f"Preprocessing complete: {input_file} -> {output_file}")

if __name__ == "__main__":
    main()