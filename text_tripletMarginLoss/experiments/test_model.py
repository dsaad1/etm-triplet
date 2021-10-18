from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import os
import sys
import argparse
import torch
import wandb
sys.path.insert(1, '../Code')
import model as my_models
import data as my_data
import topic_retrieval as tr


def get_args(parser):
    parser.add_argument('--checkpoint', type=str, default='', help='file path of model checkpoint')
    parser.add_argument('--device', type=int, default=6, help='gpu number to use')
    parser.add_argument('--max_epochs', type=int, default=45, help='number of max epochs to train')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='')
    parser.add_argument('--log_every_n_steps', type=int, default=1, help='how many training steps to log. float denotes epochs, int denotes batches')
    parser.add_argument('--val_check_interval', type=float, default=1.0, help='how many batches to validate. float denotes epochs, int denotes batches')
    parser.add_argument('--wandb_name', type=str, default="triplet_", help='name of the wandb run',)
    # parser.add_argument('--csv_path', type=str, default='./Data/triplet_data.csv', help='path of the train/val data csv')
    # parser.add_argument('--emb_path', type=str, default='./Data/embeddings.emb', help='path of the pretrained ETM embeddings')
    return parser


def main(checkpoint, device, max_epochs, accumulate_grad_batches, log_every_n_steps,
         val_check_interval, wandb_name, batch_size_val, csv_path, emb_np_path, vocab_path, **kwargs):
    
    kwargs = {'num_workers': 5, 'pin_memory': True}
    device_num = device
    device = torch.device(device_num)
    checkpoint = 'experiments/' + checkpoint
    
    vocab, vocab_size = my_data.get_vocab(vocab_path)
    embeddings = my_data.load_embeddings(emb_np_path, device)
    data_loaders = my_data.MyValDataModule(vocab, vocab_size, csv_path=csv_path,
                                           batch_size=111, batch_size_val=111)    
    m = my_models.TripletNet.load_from_checkpoint(checkpoint, vocab=vocab, vocab_size=vocab_size, embeddings=embeddings)
    logger = WandbLogger(name=wandb_name, save_dir='./savedata/', project='text-triplet')
    
    trainer = Trainer(accelerator='dp',
                      gpus=[device_num],
                      logger=logger,
                      max_epochs=max_epochs,
                      accumulate_grad_batches=accumulate_grad_batches,
                      gradient_clip_val=0.5,
                      log_every_n_steps=log_every_n_steps,
                      val_check_interval=val_check_interval)
    
    trainer.validate(m, datamodule=data_loaders)

    
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    parser = get_args(parser)
    parser = my_data.get_data_args(parser)
    args = vars(parser.parse_args())
    os.chdir('..')
    main(**args)
