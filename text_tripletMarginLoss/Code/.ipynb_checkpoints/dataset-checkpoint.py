from torch.utils.data import Dataset
import torch
import pickle
import pandas as pd
from ctt import clean

class AmazonDataset(Dataset):
    def __init__(self, data_csv: str, vocab, vocab_size: int, split: str, stopwords=None):
        """
        Dataset containing labeled Amazon Fine Foods reviews
        
        Args:
            data_csv (str): file path to the dataset csv
            vocab (list): list of words in the corpus
            vocab_size (int): size of the vocab of the corpus
            split (str): can take the value of 'train', 'test', or 'val'
            stopwords (list, optional): list of stopwords for data preprocessing
        """
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.word2id = dict([(w, j) for j, w in enumerate(self.vocab)])
        self.id2word = dict([(j, w) for j, w in enumerate(self.vocab)])
        self.df = pd.read_csv(data_csv)
        self.df = self.df[self.df['split'] == split]

        # define stopwords
        self.stopwords = stopwords if stopwords else clean.nltk_stopwords
#         stopwords_to_keep = {'no', 'not'}
#         self.stopwords = clean.nltk_stopwords - stopwords_to_keep

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        """
        Return:
            dict:  {'A': {'bow', 'len'}, 
                    'P': {'bow', 'len'},
                    'N': {'bow', 'len'}}
        """
        # get data from the dataframe
        row = self.df.iloc[ix]
        label = row['Positive']
        a_text = row['A']
        if label == 'B':
            p_text = row['B']
            n_text = row['C']
        elif label == 'C':
            p_text = row['C']
            n_text = row['B']
        
        # clean the data
        [a_bow, a_len], [p_bow, p_len], [n_bow, n_len] = [self.create_bow(txt)
                                                          for txt in [a_text, p_text, n_text]]
        
        
        return {'A': {'bow': a_bow, 'len': a_len},
                'P': {'bow': p_bow, 'len': p_len},
                'N': {'bow': n_bow, 'len': n_len}}

    def create_bow(self, text: str):
        """
        Args: 
            test (str): document that will be preprocessed and converted to BoW
        
        Return:
            bag of words representation of the document
            length of the bag of words
        """
        word_list = text.split()
        ix_list = [self.word2id[w] for w in word_list if w in self.vocab]

        bow = torch.zeros(self.vocab_size)

        for ix in ix_list:
            bow[ix] += 1

        bow_length = len(ix_list)
            
        return bow, bow_length
    
    def get_vocab_size(self):
        return self.vocab_size


if __name__ == '__main__':
    with open('../Data/min_df_10/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    data = AmazonDataset('../Data/review_labels_cleaned.csv',
        vocab, vocab_size, 'train')

    print(data[324]['A'])