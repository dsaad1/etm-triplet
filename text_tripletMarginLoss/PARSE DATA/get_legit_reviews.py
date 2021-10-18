import json
import csv
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from ctt import clean

parser = argparse.ArgumentParser(description='Parse Labelbox Data')
parser.add_argument('--json_file', type=str, help='path to .json file from labelbox')
args = parser.parse_args()
json_path = args.json_file

fields = ['ID','A','B','C','Positive']
stopwords_to_keep = {'no', 'not'}
stopwords = clean.nltk_stopwords - stopwords_to_keep

# open the labeled data
with open(json_path, 'r') as f:
    data = json.loads(f.read())

with open('review_labels.csv','w') as f:
    csvwriter = csv.writer(f) 
    csvwriter.writerow(fields)
    for i in range(len(data)):
        # Get ID
        data_id = data[i]['ID']

        # Get Text Data
        data_dict = json.loads(data[i]['Labeled Data'])['compare']
        data_a = data_dict['A']
        data_b = data_dict['B']
        data_c = data_dict['C']

        # Get review id da
        data_aid = data_dict['A_id']
        # Get Label Data
        if data[i]['Label'] == 'Skip':
            data_label = 'Skip'
        elif data[i]['Label'][0] == {}:
            data_label = 'Null'
        elif i < 1124:
            data_label = data[i]['Label'][0]['triplet_label']
        else:
            data_label = data[i]['Label'][0]['sentiment']
            
        
        # Write Data to CSV
        csvwriter.writerow([data_id, data_a, data_b, data_c, data_label])

df = pd.read_csv('review_labels.csv')

# 1. drop cases where Label is not in ['true', 'B', 'C']
is_valid = df.Positive.apply(lambda x: True if x in ['true', 'B','C'] else False)
df = df[is_valid]

# 2. replace all true with B
replace_true = df.Positive.apply(lambda x: True if x in ['true'] else False)
df.loc[replace_true, 'Positive'] = 'B'

# 4. combine duplicates where reviewers agree
is_dupe = df.duplicated(subset=['A', 'B', 'C', 'Positive'], keep='first')
df = df[~is_dupe].sort_values('A')

# 3. drop cases where reviewers disagree
is_dupe = df.duplicated(subset=['A', 'B', 'C'], keep=False)
df=df[~is_dupe]

df.Positive.value_counts()

df.to_csv('reviewers_dataset.csv', index = False)