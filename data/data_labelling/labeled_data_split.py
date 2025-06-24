import pandas as pd
import ast

# Read the labeled data
df = pd.read_csv('labeled_data.csv')

# Convert string representation of list to actual list
df['label(chatgpt,claude,grok,gemini)'] = df['label(chatgpt,claude,grok,gemini)'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else None)

# Filter out rows where label is null (keep rows with all zeros)
labeled_df = df[df['label(chatgpt,claude,grok,gemini)'].notna()]

# Save the filtered data
labeled_df.to_csv('labeled_data_only.csv', index=False)

print(f"Original data: {len(df)} rows")
print(f"Labeled data (non-null): {len(labeled_df)} rows")
print(f"Filtered data saved to 'labeled_data_only.csv'")