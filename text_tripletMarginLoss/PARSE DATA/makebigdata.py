from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv('big_reviews2.csv')
# print(f"pre-cleaning: {len(df)}")
# df['text'].replace('', np.nan, inplace=True)
# df['text'].replace(' ', np.nan, inplace=True)
# df.dropna(subset=['text'], inplace=True)
# print(f"post-cleaning: {len(df)}")


print(f"Before: {len(df)}")


# # split into train and test

for index, row in df.iterrows():
    if index == 111435 or index == 142752:
        print(row)

for index, row in df.iterrows():
    if index == 111435 or index == 142752:
        df = df.drop(index)

print(f"After: {len(df)}")

for index, row in df.iterrows():
    if index == 111435 or index == 142752:
        print(row)

f_tr, f_ts = 0.75, 0.25
df_train, df_test = train_test_split(df, test_size=f_ts)
df_train['split'] = 'train'
df_test['split'] = 'test'

df = pd.concat([df_train, df_test])
df.split.value_counts() / len(df)

df.to_csv('hopefullyfixedNaNs.csv', index=False)