# 새 파일을 만든후에는 new_data폴더에 있는 파일들을 chatgpt/claude/grok/gemini 중 맞는 폴더에 옮겨주세요.
import pandas as pd
import os

data_dir = "new_data"
output_file = "concatenated_data_2.csv"

csv_files = []
for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        csv_files.append(os.path.join(data_dir, file))

print(f"Found {len(csv_files)} CSV files:")
for file in csv_files:
    print(f"  - {file}")

dataframes = []
for file_path in csv_files:
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {file_path}: {len(df)} rows")
        dataframes.append(df)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

if dataframes:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    concatenated_df = pd.concat(dataframes, ignore_index=True)
    concatenated_df.to_csv(output_file, index=False)
    print(f"\nConcatenated {len(dataframes)} files into {output_file}")
    print(f"Total rows: {len(concatenated_df)}")
    print(f"Columns: {list(concatenated_df.columns)}")
else:
    print("No data to concatenate")