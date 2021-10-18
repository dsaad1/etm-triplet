import dataset
import torch
import pickle
import pytorch_lightning as pt
from pytorch_lightning.trainer.supporters import CombinedLoader
import model


class MyDataModule(pt.LightningDataModule):
    def __init__(self, vocab, vocab_size, csv_path, batch_size, batch_size_val, **kwargs):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.kwargs = kwargs
        
    def setup(self, stage):
        self.triplet_dataset_train = dataset.AmazonDataset(self.csv_path, self.vocab, self.vocab_size, 'train')
        self.big_dataset_train = dataset.BigDataset(self.vocab, self.vocab_size, 'train')
        self.triplet_dataset_val = dataset.AmazonDataset(self.csv_path, self.vocab, self.vocab_size, 'val')
        self.big_dataset_val = dataset.BigDataset(self.vocab, self.vocab_size, 'test')

  
    def train_dataloader(self):
        triplet_train = torch.utils.data.DataLoader(self.triplet_dataset_train, shuffle=True,
                                                    batch_size=self.batch_size, **self.kwargs)
        big_train = torch.utils.data.DataLoader(self.big_dataset_train, shuffle=True, 
                                                batch_size=2000, **self.kwargs)
        
        loaders = {"triplet": triplet_train, "big": big_train}
        combined_loaders = CombinedLoader(loaders, "max_size_cycle")
        return combined_loaders
    
    def val_dataloader(self):
        triplet_val = torch.utils.data.DataLoader(self.triplet_dataset_val, shuffle=False,
                                           batch_size=self.batch_size_val, **self.kwargs)
        big_val = torch.utils.data.DataLoader(self.big_dataset_val, shuffle=False,
                                           batch_size=2000, **self.kwargs)
        
        loaders = {"triplet": triplet_val, "big": big_val}
        combined_loaders = CombinedLoader(loaders, "max_size_cycle")
        return combined_loaders


class MyValDataModule(pt.LightningDataModule):
    def __init__(self, vocab, vocab_size, csv_path, batch_size, batch_size_val, **kwargs):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.csv_path = csv_path
        self.batch_size_val = batch_size_val
        self.kwargs = kwargs
        
    def setup(self, stage):
        self.triplet_dataset_val = dataset.AmazonDataset(self.csv_path, self.vocab, self.vocab_size, 'val')
        self.big_dataset_val = dataset.BigDataset(self.vocab, self.vocab_size, 'test')
    
    def val_dataloader(self):
        triplet_val = torch.utils.data.DataLoader(self.triplet_dataset_val, shuffle=False,
                                                  batch_size=self.batch_size_val, **self.kwargs)
        big_val = torch.utils.data.DataLoader(self.big_dataset_val, shuffle=False,
                                              batch_size=2000, **self.kwargs)
        
        loaders = {"triplet": triplet_val, "big": big_val}
        combined_loaders = CombinedLoader(loaders, "max_size_cycle")
        return combined_loaders


def get_data_args(parser):
    # parser = argparse.ArgumentParser(description='Data Arguments')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size of the training')
    parser.add_argument('--batch_size_val', type=int, default=100, help='batch size of the validation sets')
    parser.add_argument('--csv_path', type=str, default='./Data/triplet_data.csv', help='path of the train/val data csv')
    parser.add_argument('--dict_path', type=str, default='./Data/etm_amazonreviews_K_30_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.005_Bsz_1000_RhoSize_300_trainEmbeddings_0', help='path of the pretrained ETM')
    parser.add_argument('--vocab_path', type=str, default='./Data/vocab.pkl', help='path of the overall corpus vocab, that the ETM was trained on')
    parser.add_argument('--emb_path', type=str, default='./Data/embeddings.emb', help='path of the pretrained ETM embeddings')
    parser.add_argument('--emb_np_path', type=str, default='./Data/embeddings.npz.npy', help='path of the preloaded pretrained ETM embeddings as a numpy file')
    return parser


def get_vocab(vocab_path):
    """
    Returns:
        vocab (list): list of words in the corpus
        vocab_size (int): size of the vocab of the corpus
    """
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        vocab_size = len(vocab)
        
        return vocab, vocab_size


def generate_embeddings(vocab, vocab_size, device, save_path):
    """
    Args:
        vocab (list): list of words in the corpus
        vocab_size (int): size of the vocab of the corpus
        device (torch.device): device on which to perform computation
        save_path (str): path where embedding numpy will get saved
    """
    model.ETM.generate_embeddings(vocab, vocab_size, device, save_path)


def load_embeddings(emb_path, device):
    """
    Args:
        emb_path (str): path of embeddings numpy file
        device (torch.device): device on which to perform computation
    Return:
        (tensor) pretrained embeddings 
        
    """
    return model.ETM.load_embeddings(emb_path, device)


def get_test_loader(vocab, vocab_size, csv_path, batch_size_test, **kwargs):
    triplet_dataset_test = dataset.AmazonDataset(csv_path, vocab, vocab_size, 'val')
    big_dataset_test = dataset.BigDataset(vocab, vocab_size, 'test')
    
    triplet_test = torch.utils.data.DataLoader(triplet_dataset_test, shuffle=True,
                                               batch_size=batch_size_test, **kwargs)
    big_test = torch.utils.data.DataLoader(big_dataset_test, shuffle=False,
                                           batch_size=2000, **kwargs)

    loaders = {"triplet": triplet_test, "big": big_test}
    combined_loaders = CombinedLoader(loaders, "min_size")
    return combined_loaders 

