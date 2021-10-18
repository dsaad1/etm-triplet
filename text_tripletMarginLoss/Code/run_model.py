# Library imports
import wandb
import os
import torch
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import data
import model as my_models
import topic_retrieval as tr

# define args

def get_run_args(parser):
    # parser = argparse.ArgumentParser(description='Run Arguments')
    parser.add_argument('--device', type=int, default=6, help='gpu number to use')
    parser.add_argument('--checkpoint', type=str, default=None, help='pytorch lightning model checkpoint')
    parser.add_argument('--max_epochs', type=int, default=30, help='number of max epochs to train')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='')
    parser.add_argument('--log_every_n_steps', type=int, default=10, help='how many training steps to log. float denotes epochs, int denotes batches')
    parser.add_argument('--val_check_interval', type=float, default=1.0, help='how many batches to validate. float denotes epochs, int denotes batches')
    parser.add_argument('--wandb_name', type=str, default="triplet_", help='name of the wandb run')
    return parser



def main(run_args=None, model_args=None, data_args=None):
    print(run_args, model_args, data_args)

    wandb_name = run_args['wandb_name'] + 'lr' + str(model_args['lr']) + '_margin' + str(model_args['margin']) + '_beta' + str(model_args['beta'])

    device_num = run_args['device']
    device = torch.device(device_num)

    seed = 1145545
    kwargs = {'num_workers': 4, 'pin_memory': True}
    torch.manual_seed(seed)

    project_path = '..'
    os.chdir(project_path)

    # Make Data
    vocab, vocab_size = data.get_vocab(data_args['vocab_path'])
    embeddings = data.load_embeddings(data_args['emb_np_path'], device)
    data_loaders = data.MyDataModule(vocab, vocab_size, csv_path=data_args['csv_path'],
                                     batch_size=data_args['batch_size'], batch_size_val=data_args['batch_size_val'])

    # Make Model
    # if checkpoint:
    #     m = my_models.TripletNet.load_from_checkpoint(checkpoint)
    # else: 
    m = my_models.TripletNet(embeddings=embeddings, vocab=vocab, vocab_size=vocab_size,
                             device=device, data_args=data_args, kwgs=kwargs, **model_args)

    trained_model = torch.load(data_args['dict_path'])
    # m.etm.q_theta[0].load_state_dict(trained_model.q_theta[0].state_dict())
    m.etm.load_state_dict(trained_model.state_dict())

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_d-ratio', patience=4, mode='max')
    checkpoint = ModelCheckpoint(dirpath='./1models/checkpoints/',
                                 filename=wandb_name + '{epoch}')
    callbacks = [checkpoint, early_stopping]

    # logging
    logger = WandbLogger(name=wandb_name, save_dir='./savedata/', project='beta_tests')

    logger.experiment.config.update(run_args)
    logger.experiment.config.update(model_args)
    logger.experiment.config.update(data_args)
        
    trainer = Trainer(accelerator='dp',
                      gpus=[device_num],
                      callbacks=callbacks,
                      logger=logger,
                      max_epochs=run_args['max_epochs'],
                      terminate_on_nan=True,
                      accumulate_grad_batches=run_args['accumulate_grad_batches'],
                      gradient_clip_val=0.5,
                    #   profiler='simple',
                      log_every_n_steps=run_args['log_every_n_steps'],
                      val_check_interval=run_args['val_check_interval'],
                      multiple_trainloader_mode='min_size')

    logger.watch(m, log="all", log_freq=1)
    trainer.fit(m, data_loaders)
    
    # trainer.validate(m, val_dataloaders=val_loader)
    
    logger.log_metrics({'best_val-margin': m.best_val_loss - model_args['margin']})
    logger.log_metrics({'best_distance_ratio': m.best_distance_ratio})

    ########################################TOPIC RETRIEVAL########################################

    m.freeze()

    sample_reviews = [
        "Tasty and crispy. Not too crunchy and not too salty. Good chip option that you don't have to feel guilty about. I love these with my afternoon protein shake!",
        "I didn't mind paying the $20+ a bag because my dog loved them. These are so expensive that I didn't look on the bag to see where they were made until ANOTHER recent recall on Chinese dog treats/food that was killing dogs....when I looked at the Happy Hips bag, I was pretty ticked off to see that these are made in China.<br /><br />I used to joke that my dog was addicted to these, but now I am actually a little concerned that there may be some addictive substance in these Chinese treats",
        "This tahini is just wrong and taste awful, I bought this twice (first time tahinis buyer) and my hummus kept turning out weird tasting and just not right, like a chemical-ish taste and smell. I ended up trying a different brand of tahini and WOW, delicious results. This tahini is like the Spam of ham, the imitation crab of crab meat, it just doesn't taste right no matter how much you stir it or try to spice it up."
    ]
    topic_retriever = tr.TopicGetter(m, vocab, vocab_size, device, sample_reviews)

    rev1_topics, rev1_mixture = topic_retriever.get_topics_rev(1)
    print("Topics and Mixture for Review 1 retrieved!")
    rev2_topics, rev2_mixture = topic_retriever.get_topics_rev(2)
    print("Topics and Mixture for Review 2 retrieved!")
    rev3_topics, rev3_mixture = topic_retriever.get_topics_rev(3)
    print("Topics and Mixture for Review 3 retrieved!")
    
    wc = 0
    print('Review 1 Top Topics')
    for topic in rev1_topics:
        print(", ".join(topic))
    print()
    print('Review 1 Mixture')
    for word in rev1_mixture:
        print(word + ', ', end='')
        wc += 1
        if wc == 10:
            print()
    print()
    print()

    wc=0
    print('Review 2 Top Topics')
    for topic in rev2_topics:
        print(", ".join(topic))
    print()
    print('Review 2 Mixture')
    for word in rev2_mixture:
        print(word + ', ', end='')
        wc += 1
        if wc == 10:
            print()   
    print()
    print()

    wc=0
    print('Review 3 Top Topics')
    for topic in rev3_topics:
        print(", ".join(topic))
    print()
    print('Review 3 Mixture')
    for word in rev3_mixture:
        print(word + ', ', end='')
        wc += 1
        if wc == 10:
            print()


    # df = pd.DataFrame(np.array([[topic_retriever.sample_reviews[0], rev1_topics, rev1_mixture],
    #                             [topic_retriever.sample_reviews[1], rev2_topics, rev2_mixture],
    #                             [topic_retriever.sample_reviews[2], rev3_topics, rev3_mixture]]),
    #                 columns=['Review', 'Topics', 'Mixtures'])

    # wandbTable = wandb.Table(dataframe=df)

    table = wandb.Table(columns=["ID", "Review 1", "Review 2", "Review 3"], allow_mixed_types=True)
    table.add_data('Review',
                   topic_retriever.sample_reviews[0],
                   topic_retriever.sample_reviews[1], 
                   topic_retriever.sample_reviews[2])

    table.add_data('Mixture',
                   rev1_mixture,
                   rev2_mixture,
                   rev3_mixture)

    for ix in range(len(rev1_topics)):
        table.add_data('Top Topic ' + str(ix + 1),
                       rev1_topics[ix],
                       rev2_topics[ix],
                       rev3_topics[ix])

    logger.experiment.log({'Topics Table': table})



###############################################################################################

if __name__ == '__main__':
    # run_parser = get_run_args()
    # model_parser = my_models.TripletNet.get_model_args()
    # data_parser = data.get_data_args()
    
    # run_args = vars(run_parser.parse_args())
    # model_args = vars(model_parser.parse_args())
    # data_args = vars(data_parser.parse_args())
    parser = argparse.ArgumentParser(description='The Arguments')
    parser = get_run_args(parser)
    parser = my_models.TripletNet.get_model_args(parser)
    parser = data.get_data_args(parser)
        
    args = vars(parser.parse_args())
    
    run_args = {x: args[x] for x in ['device', 'checkpoint', 'max_epochs', 'accumulate_grad_batches', 'wandb_name',
                                     'log_every_n_steps', 'val_check_interval']}
    
    data_args = {x: args[x] for x in ['batch_size', 'batch_size_val', 'csv_path',
                                      'dict_path', 'vocab_path', 'emb_path', 'emb_np_path']}
    
    model_args = {x: args[x] for x in ['lr', 'lr_a', 'beta', 'margin', 'weight_decay', 'frozen', 'emb_size',
                                       'num_topics', 'rho_size', 'enc_drop', 't_hidden_size', 'theta_act']}
    
    main(run_args, model_args, data_args)
    
