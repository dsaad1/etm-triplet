import os
import numpy as np


def make_command(beta, device_num, lr, margin, background=False):
    bg = "&" if background else ""
    cmd = f"python3.8 run_model.py --lr {lr} --margin {margin} --beta {beta} --device {device_num} --wandb_name {wandb_name} {bg}"
    
    return cmd.strip()

def main(gpus_to_avoid):
    device = 0
    run = 1
    lr = 0.05
    margin = 30
    betas = [0, 0.1 
             1, 2, 
             4, 8, 
             16, 32, 
             64, 128
             ]
    
    # lr_a = 15
    # test_betas = np.logspace(np.log10(min_beta), np.log10(max_beta), 6).astype(np.float)
    # test_lrAs = [0.05, 0.5, 1]
    # test_lrAs = [2, 5, 10]
    
    for beta in betas:
        # change device every 3 runs, skipping gpus in list gpus_to_avoid
        if (run % 3) == 0:
            device += 1
            while device in gpus_to_avoid:
                device += 1
            if device == 8:
                device = 1
        os.system(make_command(beta=beta, device_num=device, lr=lr, margin=margin, background=True))
        print(f"Test: {run}. Beta: {beta}. Device: {device}")
        run += 1
            
    
if __name__ == '__main__':
    # margin = 10
    # lr = 3
    # min_beta = 0.001
    # max_beta = 0.2
    # beta = 0.08
    # lrA_list = [5, 10]
    # lr_list = [1, 5, 15]
    # margin_list = [15, 30, 45]
    
    wandb_name = "testing_betas_"
    gpus_to_avoid = [3,5,6]
    
    os.chdir('../Code')
    main(gpus_to_avoid=gpus_to_avoid)
