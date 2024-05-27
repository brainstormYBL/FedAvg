"""
Author   : Bao-lin Yin
Data     : 10.23 2023
Version  : V1.0
Function : Storing the parameters related to the training and so on.
"""
import argparse

import torch.nn


def parameters():
    para = argparse.ArgumentParser()
    # Related to the FL
    para.add_argument('--num_client', type=int, default=20, help="The maximum number of the clients")
    para.add_argument('--frac', type=float, default=0.1, help="The ratio of the selected clients")
    para.add_argument('--flag', type=bool, default=False, help="The flag to express if all the client are selected")
    # Related to the training
    para.add_argument('--epochs', type=int, default=2000, help="The maximum epochs for the FL training")
    para.add_argument('--epochs_local', type=int, default=5, help="The maximum epochs for the local training")
    para.add_argument('--lr', type=float, default=1e-3, help="The learning rate")
    para.add_argument('--batch_size', type=int, default=32, help="The bath size used for the local training")
    para.add_argument('--ratio_train', type=float, default=0.8, help="The ratio of the date used to train")
    para.add_argument('--visdom', type=bool, default=True, help="Open the visdom")
    para.add_argument('--model_name', type=str, default="LR", help="The name of the model")
    para.add_argument('--loss_func', default=torch.nn.MSELoss(), help="The loss function of the local training.")
    args = para.parse_args()
    return args
