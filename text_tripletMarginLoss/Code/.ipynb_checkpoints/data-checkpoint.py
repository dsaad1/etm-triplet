import dataset
import torch
import pickle
import model
import settings


def get_vocab():
    """
    Returns:
        vocab (list): list of words in the corpus
        vocab_size (int): size of the vocab of the corpus
    """
    with open(settings.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        vocab_size = len(vocab)
        
        return vocab, vocab_size


def make_data(vocab, vocab_size, kwargs):
    """
    Args:
        vocab (list): list of words in the corpus
        vocab_size (int): size of the vocab of the corpus
        kwargs (dict): {'num_workers', 'pin_memory'}
    Return:
        train_loader (torch.Dataloader): dataloader for train dataset
        test_loader (torch.Dataloader): dataloader for test dataset
    """
    
    #training data
    text_triplet_dataset_train = dataset.AmazonDataset(
        settings.csv_path, vocab, vocab_size, 'train')
    train_loader = torch.utils.data.DataLoader(
        text_triplet_dataset_train,
        batch_size=settings.batch_size, shuffle=True, **kwargs)

    # testing data
    text_triplet_dataset_test = dataset.AmazonDataset(
        settings.csv_path, vocab, vocab_size, 'test')
    test_loader = torch.utils.data.DataLoader(
        text_triplet_dataset_test,
        batch_size=settings.batch_size_test, shuffle=True, **kwargs)
    
    return train_loader, test_loader


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
