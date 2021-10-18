import torch
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn
import numpy as np
import wandb
import settings
import metrics
import model


def make_model(embeddings, vocab_size, device, p_drop=0.5, freeze_encoder=True):
    """
    Args:
        embeddings(torch.tensor, optional): pretrained topic embeddings
        vocab_size (int): size of the corpus vocabulary
        device (torch.device): device on which to perform computation
        p_drop (float, optional, default=0.5): 0.0-1.0 dropout rate of topics
        freeze_encoder (boolean, optional, default=True): freeze ETM parameters
        
    Return:
        m - triplet-net model containing an ETM with pretrained topic embeddings, 
            a distance computer, and a triplet probability computer
    """
    m = model.TripletNet(embeddings, vocab_size, device, p_drop=p_drop, freeze_encoder=freeze_encoder).to(device)
    trained_model = torch.load(settings.dict_path)
    m.etm.load_state_dict(trained_model.state_dict())
#     m.etm.load_state_dict(trained_model)
    return m


def run_model(model, train_loader, test_loader, lr, device, run, lr_w, weight_decay=0):
    """
    Trains and tests the model, prints metrics after each epoch 
    
    Args:
        model (TripletNet): a triplet-net object
        train_loader (torch.Dataloader): dataloader for train dataset
        test_loader (torch.Dataloader): dataloader for test dataset
        lr (float): learning rate for the ETM parameters
        device (torch.device): device on which to perform computation
        run (wandb.Run): WandB run that is being logged 
        lr_w (float): learning rate for the distance computer's parameters
        weight_decay (float, optional): parameter that may punish overfitting
    """
    #optimizer = optim.Adadelta(model.parameters(), lr=lr)
    optimizer = optim.Adadelta([
        {'params': model.etm.parameters()},#, 'weight_decay': weight_decay},
        {'params': model.distance.parameters(), 'lr': lr_w}], 
        lr=lr,
        weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=settings.gamma)
    criteria = nn.BCELoss()

    for epoch in range(1, settings.epochs + 1):
        train_loss, kldt_train, rcl_train = train(model, criteria, device, train_loader, optimizer)
        test_loss, kldt_test, rcl_test, outputs, targets = test(model, criteria, device, test_loader)
        scheduler.step()
        
        auroc_all = metrics.getAUROC_all(outputs, targets)
        accuracy_all = metrics.getAccuracy_all(outputs, targets)
        
        wandb.run.log({'Test Loss': test_loss,
                       'Train Loss': train_loss,
                       'AUROC': auroc_all,
                       'Accuracy': accuracy_all,
                       'KLDT Train': kldt_train,
                       'KLDT Test': kldt_test,
                       'RCL Train': rcl_train,
                       'RCL Test': rcl_test,
                       'Epoch': epoch})

        print("Epoch: %i. \tTrain Loss: %0.3f. Test Loss: %0.3f. AUROC_All: %0.3f. Accuracy All: %0.3f. \n\t\tKLDT_Train: %0.3f. KLDT_Test: %0.3f. RCL_Train: %0.3f. RCL_Test: %0.3f\n" % (epoch, train_loss, test_loss, auroc_all, accuracy_all, kldt_train, kldt_test, rcl_train, rcl_test))

##### TRAINING AND TESTING #####
def train(model, criteria, device, loader, optimizer):
    """
    Trains the model
    
    Args:
        model (TripletNet): a triplet-net object
        criteria (torch.nn): criterion measuring loss
        device (torch.device): device on which to perform computation
        loader (torch.Dataloader): dataloader for train dataset
        optimizer (torch.optim): optimization algorithim 
    Return:
        mean of train loss of all batches (Triplet Metric)
        mean of kld_theta across all batches (ETM Metric)
        mean of reconstruction loss across all batches (ETM Metric)
    """
    
    model.train()

    mean_batch_losses = []
    mean_kld_theta = []
    mean_recon_loss = []
    for batch_idx, batch_dict in enumerate(loader):
        A, B, C, target = [batch_dict[key] for key in ['A', 'B', 'C', "Label"]]
        target = target.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        output, kldt, rcl = model(A, B, C)
        loss = criteria(output, target)
        loss.backward()
        optimizer.step()

        mean_batch_losses.append(loss.item())
        mean_kld_theta.append(kldt)
        mean_recon_loss.append(rcl)

    return np.mean(mean_batch_losses), np.mean(mean_kld_theta), np.mean(mean_recon_loss)


def test(model, criteria, device, loader):
    """
    Tests the model
    
    Args:
        model (TripletNet): a triplet-net object
        criteria (torch.nn): criterion measuring loss
        device (torch.device): device on which to perform computation
        loader (torch.Dataloader): dataloader for train dataset
        
    Return:
        mean of train loss of all batches (Triplet Metric)
        mean of kld_theta across all batches (ETM Metric)
        mean of reconstruction loss across all batches (ETM Metric)
        outputs of the model (predictions)
        targets of the model (correct values)
    """
    model.eval()

    mean_batch_losses = []
    mean_kld_theta = []
    mean_recon_loss = []
    outputs = []
    targets = []

    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(loader):
            A, B, C, target = [batch_dict[key] for key in ['A', 'B', 'C', "Label"]]
            output, kldt, rcl = model(A, B, C)
            target = target.to(device, dtype=torch.float32)
            loss = criteria(output, target)

            # store results
            mean_batch_losses.append(loss.item())
            mean_kld_theta.append(kldt)
            mean_recon_loss.append(rcl)
            outputs.append(output)
            targets.append(target)

    outputs = torch.cat(outputs)
    targets = torch.cat(targets)

    return np.mean(mean_batch_losses), np.mean(mean_kld_theta), np.mean(mean_recon_loss), outputs, targets
