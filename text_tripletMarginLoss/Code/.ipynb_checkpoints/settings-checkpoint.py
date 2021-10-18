import torch

emb_size = 300
num_topics = 30
distance_output_dim = 5
rho_size = 300
enc_drop = 0.0
t_hidden_size = 800
theta_act = 'relu'

batch_size = 16 #input batch size for training (default: 64)
batch_size_test = 1000 
epochs = 15
gamma = 0.7


csv_path = './Data/review_labels_221.csv'
# dict_path = './triplet_etm2.pt'
# dict_path = './Data/triple_trained_model'
dict_path = './Data/etm_amazonreviews_K_30_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.005_Bsz_1000_RhoSize_300_trainEmbeddings_0'
vocab_path = './Data/vocab.pkl'
emb_path = './Data/embeddings.emb'
emb_np_path = './Data/embeddings.npz'
