import torch
import pytorch_lightning
import numpy as np
import pandas as pd
from ctt import clean
import model as my_models
import data


class TopicGetter():
    def __init__(self, m, vocab, vocab_size, device, sample_reviews: list):
        self.sample_reviews = sample_reviews
         
        self.model = m.to(device)
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.device = device
        
        stopwords_to_keep = {'no', 'not'}
        self.stopwords = clean.nltk_stopwords - stopwords_to_keep

    def top_words_for_topic_mixture(self, review):
        words = []
        theta, _, _, _ = self.model.etm(review)
        with torch.no_grad():
            word_distribution = self.model.etm.get_beta().T @ theta.T
        top_words = np.argsort(-word_distribution[:, 0].cpu().numpy())[:20]
        for word_token in top_words:
            word = self.vocab[word_token]
            words.append(word)
        return words

    def top_words_for_top_topics(self, review):
        topics = []
        theta, _, _, _ = self.model.etm(review)
        with torch.no_grad():
            top_topics = np.argsort(-theta.cpu().numpy())[:20]
        top_topics = top_topics[0]
        for topic in top_topics:
            topics.append(self.get_words(topic))
        return topics

    # get the words for a topic
    def get_words(self, topic_ix, num_words=10):
        word_distribution = self.model.etm.get_beta()[topic_ix].detach().cpu().numpy()
        top_words = np.argsort(-word_distribution)[:num_words]

        return [self.vocab[ix] for ix in top_words]
        
    def create_bow(self, text: str):
        word2id = dict([(w, j) for j, w in enumerate(self.vocab)])
        id2word = dict([(j, w) for j, w in enumerate(self.vocab)])
        text = clean.kitchen_sink(text, self.stopwords)
        word_list = text.split()
        ix_list = [word2id[w] for w in word_list if w in self.vocab]

        bow = torch.zeros(self.vocab_size)

        for ix in ix_list:
            bow[ix] += 1

        bow_length = len(ix_list)
        
        bow_length = torch.tensor(bow_length)
        bow_length = bow_length[None]
        bow = bow[None]

        return bow, bow_length
    
    def get_topics_rev(self, rev_num):
        """
        Args:
            rev_num: the review number to retrieve the topics of, not index
        """
        review = self.sample_reviews[rev_num - 1]
        rev_bow, rev_len = self.create_bow(review)
        rev_len = rev_len.to(self.device)
        rev_bow = rev_bow.to(self.device)
        rev = {'bow': rev_bow, 'len': rev_len}
        # rev1['bow'].shape = [vocab_size], torch tensor
        # rev1['len'] is a python int

        # => need to convert length to torch scalar: torch.tensor(thing)
        # => need to add batch dimension to both: thing[None]
        
        words_for_topics = self.top_words_for_top_topics(rev)
        words_for_mixture = self.top_words_for_topic_mixture(rev)
                
        return words_for_topics, words_for_mixture 