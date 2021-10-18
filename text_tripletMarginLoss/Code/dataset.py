from torch.utils.data import Dataset
import torch
import pickle
import csv
import os
import numpy as np
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
        self.vocab_set = set(vocab)
        self.vocab_size = len(self.vocab)
        self.word2id = dict([(w, j) for j, w in enumerate(self.vocab)])
        self.id2word = dict([(j, w) for j, w in enumerate(self.vocab)])
        self.df = pd.read_csv(data_csv)
        self.df = self.df[self.df['split'] == split]
        self.bow_list = []

        # define stopwords
        # self.stopwords = stopwords if stopwords else clean.nltk_stopwords
        # stopwords_to_add = {'great', 'good'}
        # self.stopwords = clean.nltk_stopwords | stopwords_to_add
        stopwords_to_keep = {'no', 'not'}
        self.stopwords = clean.nltk_stopwords - stopwords_to_keep

        # create array of BoW of each review
        for index, row in self.df.iterrows():
            print(f'Working on {index} / {self.df.shape[0]}', end='\r')
            a_bow, a_len = self.create_bow(clean.kitchen_sink(row['A'], self.stopwords))
            b_bow, b_len = self.create_bow(clean.kitchen_sink(row['B'], self.stopwords))
            c_bow, c_len = self.create_bow(clean.kitchen_sink(row['C'], self.stopwords))
            
            self.bow_list.append({'A': {'bow': a_bow, 'len': a_len},
                                  'B': {'bow': b_bow, 'len': b_len},
                                  'C': {'bow': c_bow, 'len': c_len}})
            
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
        bows = self.bow_list[ix]
        label = row['Positive']
                
        a_dict = bows['A']
        if label == 'B':
            p_dict = bows['B']
            n_dict = bows['C']
        elif label == 'C':
            p_dict = bows['C']
            n_dict = bows['B']

        return {'A': a_dict,
                'P': p_dict,
                'N': n_dict}

    def create_bow(self, text: str):
        """
        Args: 
            text (str): document that will be preprocessed and converted to BoW
        
        Return:
            bag of words representation of the document
            length of the bag of words
        """
        word_list = text.split()
        ix_list = [self.word2id[w] for w in word_list if w in self.vocab_set]

        bow = torch.zeros(self.vocab_size)

        for ix in ix_list:
            bow[ix] += 1

        bow_length = len(ix_list)
        
        return bow, bow_length
    
    def get_vocab_size(self):
        return self.vocab_size


class BigDataset(Dataset):
    def __init__(self, vocab, vocab_size: int, split: str, stopwords=None):
        """
        Dataset containing ALL Amazon Fine Foods reviews
        
        Args:
            vocab (list): list of words in the corpus
            vocab_size (int): size of the vocab of the corpus
            split (str): can take the value of 'train', 'test', or 'val'
            stopwords (list, optional): list of stopwords for data preprocessing
        """
        self.data_csv = './Data/hopefullyfixedNaNs.csv'
        self.vocab = vocab
        self.vocab_set = set(vocab)
        self.vocab_size = len(self.vocab)
        self.word2id = dict([(w, j) for j, w in enumerate(self.vocab)])
        self.id2word = dict([(j, w) for j, w in enumerate(self.vocab)])
        self.df = pd.read_csv(self.data_csv)
        self.df = self.df[self.df['split'] == split]
        self.bow_list = []
        self.len_list = []
        
        # define stopwords
        # self.stopwords = stopwords if stopwords else clean.nltk_stopwords
        # stopwords_to_add = {'great', 'good'}
        # self.stopwords = clean.nltk_stopwords | stopwords_to_add

        stopwords_to_keep = {'no', 'not'}
        self.stopwords = clean.nltk_stopwords - stopwords_to_keep

        # create array of BoW of each review
        # for index, row in self.df.iterrows():
        #     print(f'Working on {index} / {self.df.shape[0]}', end='\r')
        #     bow, bow_len = self.create_bow(clean.kitchen_sink(row['text'], self.stopwords))
        #     if bow_len != 0:
        #         self.bow_list.append({'bow': bow, 'len': bow_len})
        
        # torch.save(self.bow_list, './Data/full_dataset_bow2_' + split + '.pt')
        
        self.bow_list = torch.load('./Data/full_dataset_bow2_' + split + '.pt')


    def __len__(self):
        return len(self.bow_list)

    def __getitem__(self, ix):
        """
        Return:

        """
        return self.bow_list[ix]

    def create_bow(self, text: str):
        """
        Args: 
            text (str): document that will be preprocessed and converted to BoW
        
        Return:
            bag of words representation of the document
            length of the bag of words
        """
        word_list = text.split()
        ix_list = [self.word2id[w] for w in word_list if w in self.vocab_set]

        bow = torch.zeros(self.vocab_size)

        for ix in ix_list:
            bow[ix] += 1

        bow_length = len(ix_list)
        
        return bow, bow_length
    
    def get_vocab_size(self):
        return self.vocab_size



if __name__ == '__main__':
    print(os.getcwd())
    with open('../Data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    data = AmazonDataset(vocab, vocab_size, 'train')

    print(data[324])