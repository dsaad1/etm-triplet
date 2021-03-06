import torch
import pickle
import numpy as np
import statistics
import argparse
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import sklearn
import dataset
import data
from collections import OrderedDict



# add faces encoder
# encoder class copied from VAE notebook
class ETM(nn.Module):
    def __init__(self, device, num_topics, vocab_size, t_hidden_size, rho_size, emsize, 
                 theta_act, embeddings=None, train_embeddings=True, enc_drop=0.5):
        """
        Args:
            device (torch.device): device on which to perform computation
            num_topics (int): number of topic vectors
            vocab_size (int): size of the corpus vocabulary
            t_hidden_size (int): dimension of hidden space of q_theta, must match pretrained encoder size 
            rho_size (int): dimension of rho, must match pretrained encoder size
            emsize (int): dimension of topic embeddings, must match pretrained encoder size
            theta_act (int): activation; (tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)
            embeddings(torch.tensor, optional): pretrained topic embeddings
            train_embeddings(boolean, optional, default=True): whether to train embeddings or not
                                                    should be True if not loading in embeddings
            enc_drop (float, optional, default=0.5): encoder dropout rate
        """
        super(ETM, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.emsize = emsize
        self.t_drop = nn.Dropout(enc_drop)
        self.device = device

        self.theta_act = self.get_activation(theta_act)
        
        ## define the word embedding matrix \rho
        num_embeddings, emsize = embeddings.size()
        rho = nn.Embedding(num_embeddings, emsize)
        self.rho = embeddings.clone().float()#.to(self.device)

        ## define the matrix containing the topic embeddings
        self.alphas = nn.Linear(rho_size, num_topics, bias=False)#nn.Parameter(torch.randn(rho_size, num_topics))
    
        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta = nn.Sequential(
                nn.Linear(vocab_size, t_hidden_size), 
                self.theta_act,
                nn.Linear(t_hidden_size, t_hidden_size),
                self.theta_act,
            )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act 

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def encode(self, bows):
        """Returns paramters of the variational distribution for \theta.

        Args:
            bows (torch.tensor): batch of normalized bag of words, shape batch_size x vocab_size
            
        Returns: 
            mu_theta
            log_sigma_theta
        """
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta
  
    def get_beta(self): # call and save
        try:
            logit = self.alphas(self.rho.weight) # torch.mm(self.rho, self.alphas)
        except:
            logit = self.alphas(self.rho)
        beta = F.softmax(logit, dim=0).transpose(1, 0) ## softmax over vocab dimension
        return beta

    def get_theta(self, normalized_bows):
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1) 
        return theta, kld_theta, z
    
    def decode(self, theta, beta):
        res = torch.mm(theta, beta)
        preds = torch.log(res + 1e-6)
        return preds 
 
    def normalize(self, bow_dict):
        """
        Normalizes a bag of words.

        Args:
            bow_dict (dictionary): dictionary of {'bow', 'len'} to be normalized

        Returns:
            the input bag of words, normalized 
        """
        bow = bow_dict['bow']
        bow_len = bow_dict['len']
        return bow / bow_len[:, None]
             
    def forward(self, bow_dict, theta=None, aggregate=True):
        """
        returns a topic vector
        
        Args:
            bow_dict: dictionary of {bow, bow_length}, does not need to be normalized
            theta (optional)
            aggregate (optional, default=True)
                        
        Return:
            theta - topic vectors
            kld_theta (ETM Metric)
            reconstruction loss (ETM Metric)
    
        """
        ## get a normalized bow
        normalized_bows = self.normalize(bow_dict)
        
        ## get theta
        if theta is None:
            theta, kld_theta, z = self.get_theta(normalized_bows)
            
        ## get beta
        beta = self.get_beta()
            
        ## get reconstruction loss
        preds = self.decode(theta, beta)
        recon_loss = -(preds * bow_dict['bow']).sum(1)
        if aggregate:
            recon_loss = recon_loss.mean()
            
        return theta, kld_theta, recon_loss, z

    @classmethod
    def load_embeddings(cls, emb_path, device):
        embeddings = np.load(emb_path)
        embeddings = torch.from_numpy(embeddings).to(device)
        return embeddings

    @classmethod
    def generate_embeddings(cls, vocab, vocab_size, device, save_path, emb_path, emb_size):
        """
        Loads pretrained embeddings from file path found in settings
        
        Args:
            vocab (list): list of words in the corpus
            vocab_size (int): size of the vocab of the corpus
            device (torch.device): device on which to perform computation
            
        Return:
            (tensor) pretrained embeddings 
        """
        vectors = {}
        with open(emb_path, 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                if word in vocab:
                    vect = np.array(line[1:]).astype(np.float)
                    vectors[word] = vect
        embeddings = np.zeros((vocab_size, emb_size))
        words_found = 0
        for i, word in enumerate(vocab):
            try: 
                embeddings[i] = vectors[word]
                words_found += 1
            except KeyError:
                embeddings[i] = np.random.normal(scale=0.6, size=(emb_size, ))
        np.save(save_path, embeddings)

    
###############################################################################
# TRIPLET MODEL

class TripletNet(pl.LightningModule):
    def __init__(self, embeddings, vocab_size, device, emb_size, num_topics,
                 rho_size, enc_drop, t_hidden_size, theta_act, lr=1, lr_a=1,
                 beta=0, weight_decay=0, margin=1, **kwargs):
        """
        Args:
            embeddings(torch.tensor, optional): pretrained topic embeddings   
            vocab_size (int): size of the corpus vocabulary
            device (torch.device): device on which to perform computation
            lr (float, optional): learning rate for the TML parameters
            lr_a (float, optional): learning rate for etm-loss parameters
            beta(float, optional): weight of ETM-loss terms
            weight_decay (float, optional): parameter that may punish overfitting
            margin (float, optional, default=1): margin for the triplet loss  
            freeze_encoder (boolean, optional, default=True): freeze ETM parameters
        """
        super(TripletNet, self).__init__()
        
        self.save_hyperparameters('device', 'emb_size', 'num_topics', 'rho_size', 'enc_drop',
                                  't_hidden_size', 'theta_act', 'lr', 'lr_a', 'beta', 'margin')
        
        self.embeddings = embeddings
        self.lr = lr
        self.lr_a = lr_a
        self.beta = beta
        self.vocab_size = vocab_size
        self.weight_decay = weight_decay
        self.margin = margin
        
        self.num_topics = num_topics
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.emb_size = emb_size
        self.theta_act = theta_act
        self.enc_drop = enc_drop
        
        # record best loss
        self.best_val_loss = 1000
        self.best_distance_ratio = 0
        
        # setup the loss
        self.triplet_margin_loss = nn.TripletMarginLoss(margin=margin, p=2)
        
        # create encoder
        self.etm = ETM(device=device, num_topics=self.num_topics, vocab_size=vocab_size,
                       t_hidden_size=self.t_hidden_size,
                       rho_size=self.rho_size, emsize=self.emb_size, theta_act=self.theta_act, 
                       embeddings=self.embeddings,
                       train_embeddings=False, enc_drop=self.enc_drop)
        
        # for param in self.etm.parameters():
        #     param.requires_grad = False
        
        for param in self.etm.q_theta[0].parameters():
            param.requires_grad = False

        # for param in self.etm.mu_q_theta.parameters():
        #     param.requires_grad = True
            
        # for param in self.etm.q_theta[2].parameters():
        #     param.requires_grad = True

        # for param in self.etm.logsigma_q_theta.parameters():
        #     param.requires_grad = True
            
        # for param in self.etm.alphas.parameters():
        #     param.requires_grad = True
        
        # dropout prior to computing triplet margin loss
        self.drop = nn.Dropout(0.25)
        
    def get_distance(self, tA, t2):
        return torch.linalg.norm(tA - t2, dim=-1)
    
    def training_step(self, batch, batch_idx):
        batch_t = batch['triplet']
        batch_b = batch['big']
        A, P, N = [batch_t[key] for key in ['A', 'P', 'N']]
        
        tA, tP, tN, _, _ = self(A, P, N)
        _, kldt, rcl, _ = self.etm(batch_b)
        dAP = self.get_distance(tA, tP)
        dAN = self.get_distance(tA, tN)
        
        L_triplet_loss = self.triplet_margin_loss(tA, tP, tN) 
        total_loss = self.beta * (L_triplet_loss) + (kldt + rcl)
            
        distance_ratio = (dAP < dAN).type(torch.float32).mean()

        self.log('train_dAP_avg', dAP.mean())
        self.log('train_dAN_avg', dAN.mean())
        self.log('train_triplet_loss', L_triplet_loss)
        self.log('train_kld_theta', kldt)
        self.log('train_recon_loss', rcl)
        self.log('train_d-ratio', distance_ratio)
        self.log('train_total_loss', total_loss)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        A, P, N = [batch[key] for key in ['A', 'P', 'N']]
        tA, tP, tN, kldt, rcl = self(A, P, N)
        
        L_triplet_loss = self.triplet_margin_loss(tA, tP, tN) - self.margin
        total_loss = self.beta * L_triplet_loss + (kldt + rcl)

        self.log('test_train_loss', L_triplet_loss)
        self.log('test_kld_theta', kldt)
        self.log('test_recon_loss', rcl)
        self.log('test_total_loss', total_loss)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        batch_t = batch['triplet']
        batch_b = batch['big']
        A, P, N = [batch_t[key] for key in ['A', 'P', 'N']]

        tA, tP, tN, _, _ = self(A, P, N)
        _, kldt, rcl, _ = self.etm(batch_b)
        
        dAP = self.get_distance(tA, tP)
        dAN = self.get_distance(tA, tN)
        
        L_triplet_loss = self.triplet_margin_loss(tA, tP, tN)
        total_loss = self.beta * (L_triplet_loss) + (kldt + rcl)
        distance_ratio = (dAP < dAN).type(torch.float32).mean()

        self.log('val_dAP_avg', dAP.mean())
        self.log('val_dAN_avg', dAN.mean())
        self.log('val_triplet_loss', L_triplet_loss)
        self.log('val_kld_theta', kldt)
        self.log('val_recon_loss', rcl)
        self.log('val_d-ratio', distance_ratio)
        self.log('val_total_loss', total_loss)
        
        return L_triplet_loss, distance_ratio
    
    def validation_epoch_end(self, outputs):
        val_loss = torch.tensor([out[0] for out in outputs])
        distance_ratio = torch.tensor([out[1] for out in outputs])
        avg_val_loss = val_loss.mean()
        avg_distance_ratio = distance_ratio.mean()
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
        if avg_distance_ratio > self.best_distance_ratio:
            self.best_distance_ratio = avg_distance_ratio

    def configure_optimizers(self):
        optimizer = optim.Adadelta([
            {'params': self.etm.mu_q_theta.parameters(), 'lr': self.lr},
            {'params': self.etm.q_theta[2].parameters(), 'lr': self.lr},
            {'params': self.etm.logsigma_q_theta.parameters(), 'lr': self.lr},
            {'params': self.etm.alphas.parameters(), 'lr': 0.5}])
        return optimizer   
        
    def forward(self, A, P, N):
        """
        Args:
            A (dict ['bow', 'len']): bag of words dict for document ANCHOR
            P (dict ['bow', 'len']): bag of words dict for document POSITIVE
            N (dict ['bow', 'len']): bag of words dict for document NEGATIVE
        Return: 
            weighted topic vectors of A, P, and N (Triplet Task)
            mean kld_theta of the triplet set (ETM Metric)
            reconstruction_loss of the triplet set (ETM Metric)
        """
        # first compute all of the encodings
        (tA, kldt_A, rcl_A, zA), (tP, kldt_P, rcl_P, zP), (tN, kldt_N, rcl_N, zN) = [self.etm(x) for x in (A, P, N)]
        
        # apply dropout to topic vectors
        # zA, zP, zN = [self.drop(x) for x in (zA, zP, zN)]
        
        # obtain average of kld_theta and recon_loss across the triplet
        kldt = statistics.mean([kldt_A.item(), kldt_P.item(), kldt_N.item()])
        rcl = statistics.mean([rcl_A.item(), rcl_P.item(), rcl_N.item()])

        return zA, zP, zN, kldt, rcl
    
    @staticmethod
    def get_model_args(parser):
        # parser = argparse.ArgumentParser(description='Model Arguments')
        parser.add_argument('--lr', type=float, default=1.0, help='learning rate of TML parameters (muqtheta, q_theta, logsigma)')
        parser.add_argument('--lr_a', type=float, default=1.0, help='learning rate of ETM-loss parameters (alpha)')
        parser.add_argument('--beta', type=float, default=0, help='weight of the ETM-loss terms')
        parser.add_argument('--margin', type=float, default=2, help='margin for the triplet loss')
        parser.add_argument('--weight_decay', type=float, default=0.0, help='parameter that may punish overfitting')
        parser.add_argument('--frozen', type=bool, default=False, help='freeze or unfreeze ETM encoder')
        parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
        parser.add_argument('--num_topics', type=int, default=30, help='number of topics')
        parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
        parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')
        parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
        parser.add_argument('--theta_act', type=str, default='relu', help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu')
        return parser

