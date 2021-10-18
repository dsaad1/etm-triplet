# Library imports
import wandb
import os
import numpy as np
import torch
import pandas as pd

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


import dataset 
import metrics
import etm
import data
import utls as utl
import model as my_models
import settings


# Imported settings
emb_size = settings.emb_size
num_topics = settings.num_topics
rho_size = settings.rho_size
enc_drop = settings.enc_drop
t_hidden_size = settings.t_hidden_size
theta_act = settings.theta_act
batch_size = settings.batch_size
batch_size_test = settings.batch_size_test
emb_np_path = settings.emb_np_path
emb_path = settings.emb_path

checkpoint = None

# training settings
max_epochs = 20
accumulate_grad_batches = 1
log_every_n_steps = 1
val_check_interval = 1

device_num = 3
device = torch.device(device_num)
# lr = 25
# lr_w = 25

# model settings
gamma=0.7 #Learning rate step gamma (default: 0.7)
seed=42 #random seed (default: 42)
save_model=False #save the trained model (default: False)

# misc settings
no_cuda=False #disables CUDA training (default: True)
use_cuda = not no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 10, 'pin_memory': True}
torch.manual_seed(seed)


project_path = '..'
os.chdir(project_path)


# Make Data
vocab,vocab_size = data.get_vocab()
train_loader, test_loader = data.make_data(vocab, 
                            vocab_size, kwargs)
# Make Embeddings
embeddings = data.load_embeddings('./Data/embeddings.npz.npy', device)


# Define hyperparameters
lr = 1
lr_w = 1250
frozen = False


# Make Model
# if checkpoint:
#     m = my_models.TripletNet.load_from_checkpoint(checkpoint, embeddings, vocab_size, device, lr, lr_w)
# else: 
m = my_models.TripletNet(embeddings, vocab_size, device, lr=lr, freeze_encoder=frozen, margin=1)

trained_model = torch.load(settings.dict_path)
m.etm.load_state_dict(trained_model.state_dict())

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=100)
checkpoint = ModelCheckpoint(monitor='val_loss', dirpath='./models/checkpoints/')
callbacks = [early_stopping, checkpoint]

# logging
logger = WandbLogger(name='frozen:'+str(frozen)+'_lr:'+str(lr)+'_lrW:'str(lr_w), save_dir='./savedata/', project='qualitative-analysis')

trainer = Trainer(accelerator='ddp',
                  max_epochs=max_epochs,
                  accumulate_grad_batches=accumulate_grad_batches, 
                  gpus=[device_num],
                  callbacks=callbacks,
                  logger=logger,
                  reload_dataloaders_every_epoch=True,
                  log_every_n_steps=log_every_n_steps,
                  val_check_interval=val_check_interval)

trainer.fit(m, train_loader, test_loader)



