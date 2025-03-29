import argparse
import subprocess


parser = argparse.ArgumentParser(description='Preprocessing script')
parser.add_argument('--data_folder', type=str, required=False, default='train_10M',
                  help='The data folder to process')
args = parser.parse_args()

def clean_data(data_folder: str = args.data_folder) -> None:
    """Execute Bash Script for Data Cleaning"""
    
    print(f"Processing data from: {data_folder}")
    
    bash_script = './run_preprocess.sh'
    
    # Run the bash script with the data_folder argument
    try:
        subprocess.run(['chmod', '+x', bash_script])
        result = subprocess.run([bash_script, args.data_folder], 
                            check=True, 
                            text=True,
                            capture_output=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        print(e.stderr)
    

if __name__ == "__main__":
    clean_data(data_folder=args.data_folder)