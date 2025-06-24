import pandas as pd

df_1 =  pd.read_csv('../raw_data/junk/labeled_data_1.csv')
df_2 = pd.read_csv('../data_processing/processed_data_2.csv')
df_3 = pd.concat([df_1, df_2])
df_3.to_csv('labeled_data_concated.csv', index=False)