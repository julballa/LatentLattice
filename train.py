# Adapted from https://github.com/divelab/AIRS/blob/main/OpenProt/LatentDiff/LatentDiff/protein_autoencoder/main.py

import os
import sys
import time
import argparse

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
from torch_geometric.data import Data, Batch

import numpy as np
import random
from matplotlib import pyplot as plt

from tqdm import tqdm

from model import LatticeAuto
from utils import RMSD, KabschRMSD
import re
# from analyze_plot import analyze, get_ground_truth, get_reconstructed, plot_histogram

# global variable initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reg_criterion = torch.nn.L1Loss()
multi_class_criterion = torch.nn.CrossEntropyLoss()
binary_class_criterion = torch.nn.BCELoss()

def eval(args, model, loader, length=None):
    """
    Evaluate the model
    Args:
        model: model to evaluate
        data: valid_loader or test_loader
    Returns:
        Model error on data
    """
    model.eval()

    total_kl_x = 0
    total_kl_h = 0

    pred_dist = []
    pred_coord = []
    pred_atom_type = []

    true_dist = []
    true_coord = []
    true_atom_type = []

    count = 0

    rmsd_criterion = RMSD()

    # loop over minibatches
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        count += 1

        batch.coords = batch.coords.double()
        batch = batch.to(device)

        with torch.no_grad():
            pred_coords, pred_atom, kl_x, kl_h = model(batch)


        total_kl_x += kl_x
        total_kl_h += kl_h

        print(pred_coords.shape, pred_atom.shape)
        pred_coords_split = torch.split(pred_coords, length)
        for pred_coords in pred_coords_split:
            pred_coord.append(pred_coords.detach().cpu())
            pred_dist.append((pred_coords[0:-1] - pred_coords[1:]).pow(2).sum(-1).sqrt().detach().cpu())

        pred_atom_type.append(pred_atom.detach().cpu())

        coords_split = torch.split(batch.coords, length)
        for coords in coords_split:
            true_coord.append(coords.detach().cpu())
            true_dist.append((coords[0:-1] - coords[1:]).pow(2).sum(-1).sqrt().detach().cpu())

        true_atom_type.append(batch.x.detach().cpu())

    # accuracy for atom type prediction
    preds = torch.argmax(torch.cat(pred_atom_type, dim=0), dim=1)
    acc_atom = torch.sum(preds == torch.cat(true_atom_type, dim=0).squeeze(1)) / preds.shape[0]

    # MAE for atom position reconstruction
    mae = reg_criterion(torch.cat(pred_coord, dim=0), torch.cat(true_coord, dim=0))

    # calculate rmsd, note: this calculation doesn't use alignment, if use krmsd, batch size need to be set to 1
    rmsd = rmsd_criterion(pred_coord, true_coord)

    # MAE for edge distance
    edge_mae = reg_criterion(torch.cat(pred_dist, dim=0), torch.cat(true_dist, dim=0))
    pred_dist = torch.cat(pred_dist, dim=0)

    return edge_mae, mae, rmsd, acc_atom, total_kl_x / (step + 1), total_kl_h / (step + 1), pred_coord, true_coord, pred_atom_type, true_atom_type



def train(args, model, loader, optimizer, working_dir, loss_term='all', length=None, edgeloss_weight=0.5, kl_weight=0):
    """
    Train the model for one epoch
    Args:
        model: model to train
        loader: DataLoader
        optimizer: torch.optim
    Returns:
        Training error
    """

    # set model(s) to training mode and init misc. variables for training
    model.train()
    total_loss = 0
    total_mae = 0
    total_res_loss = 0
    total_edge_mae = 0
    total_kl_x = 0
    total_kl_h = 0

    # loop over minibatches
    t = tqdm(loader, desc="Iteration")

    fig = plt.figure()

    for step, batch in enumerate(t):
        # count total number of training steps and move the minibatch to device

        batch.coords = batch.coords.double()
        batch = batch.to(device)

        pred_coords, pred_atom, kl_x, kl_h = model(batch)

        assert torch.isnan(pred_coords).sum() == 0
        assert torch.isnan(pred_atom).sum() == 0


        # MAE loss
        loss_coords = reg_criterion(pred_coords, batch.coords)
        # cross entropy loss for atom type prediction
        loss_multi_classify = multi_class_criterion(pred_atom, batch.x.squeeze(1).to(torch.long))
      
        # edge distance loss
        edge_dist_loss = 0
        pred_coords_split = torch.split(pred_coords, length)
        coords_split = torch.split(batch.coords, length)
        count = 0
        for pred_coords, coords in zip(pred_coords_split, coords_split):
            count += 1
            pred_dist = (pred_coords[0:-1] - pred_coords[1:]).pow(2).sum(-1).sqrt()
            true_dist = (coords[0:-1] - coords[1:]).pow(2).sum(-1).sqrt()
            edge_dist_loss += reg_criterion(pred_dist, true_dist)
        edge_dist_loss = edge_dist_loss / count

        loss = loss_coords + loss_multi_classify + 0.1 * kl_x + kl_weight * kl_h + edgeloss_weight * edge_dist_loss 

        # reset accumlated gradient from previous backprop and back prop
        optimizer.zero_grad()

        loss.backward()

        # append description for tqdm progress bar
        t.set_description(f"loss_dist {edge_dist_loss:.3f}, "
                          f"loss_coords {loss_coords:.3f}, "
                          f"loss_res {loss_multi_classify:.3f}, ")


        optimizer.step()

        total_loss += loss.detach().cpu()
        total_mae += loss_coords.detach().cpu()
        total_res_loss += loss_multi_classify.detach().cpu()
        total_edge_mae += edge_dist_loss.detach().cpu()

        if type(kl_x) != int:
            total_kl_x += kl_x.detach().cpu()
        if type(kl_h) != int:
            total_kl_h += kl_h.detach().cpu()

    # return the mean loss across all minibatches
    return total_loss / (step + 1), total_mae / (step + 1), total_res_loss / (step + 1), \
           total_edge_mae / (step + 1), total_kl_x / (step + 1), total_kl_h / (step + 1)

    
def main():

    # parse arguments
    parser = argparse.ArgumentParser(description="Lattice generation")

    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')

    # model hyperparameters
    parser.add_argument("--mp_steps", type=int, default=4, help="number of steps of message passing for equivariant network")
    parser.add_argument("--emb_dim", type=int, default=32, help="dimensionality of hidden layers in GNN")
    parser.add_argument("--layers", type=int, default=3, help="number of layers in encoder and decoder")
    parser.add_argument('--pooling', type=str, default='True', help='pooling or not')
    parser.add_argument('--up_mlp', action='store_true', default=False, help='mlp after copy latent embedding')
    parser.add_argument('--residual', type=str, default='True', help='residual connection or not')
    parser.add_argument('--noise', action='store_true', default=False, help='add noise to input position')
    parser.add_argument('--transpose', action='store_true', default=False, help='decoder in transpose conv way')
    parser.add_argument('--attn', action='store_true', default=False, help='')
    parser.add_argument('--loss', type=str, default='all', help='')


    # training
    parser.add_argument("--mode", type=str, default="train", help="")
    parser.add_argument("--lr_init", type=float, default = 1e-3, help="initial learning rate")
    parser.add_argument("--epochs", type=int, default=40, help="number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=128, help="input batch size for training")
    parser.add_argument("--edgeloss_weight", type=float, default=0.5, help="weight for edge distance loss")
    parser.add_argument("--kl_weight", type=float, default=0, help="weight for kl divergence loss")

    # data
    parser.add_argument("--dataname", type=str, default="2d_square_lattice", help="data")
    parser.add_argument("--num_workers", type=int, default=0, help="num of data loader workers")
    parser.add_argument("--data_path", type=str, default="data/", help="path to data")

    # directory
    parser.add_argument("--working_dir", type=str, default="", help="working directory for logs, saved models, etc.")
    parser.add_argument("--suffix", type=str, default="", help="optional suffix added to working_dir")
    parser.add_argument("--log_dir", type=str, default="", help="tensorboard log directory")
    parser.add_argument("--checkpoint_dir", type=str, default="trained_models", help="directory to save checkpoint in working directory")
    parser.add_argument("--saved_model_dir", type=str, default=None, help="directory with checkpoint.pt")

    # parse args and display; get time
    args = parser.parse_args()

    args.pooling = True if (args.pooling == 'True') else False
    args.residual = True if (args.residual == 'True') else False

    print(args)
    cur_time = time.strftime("%Y%m%d_%H%M")

    # create the working dir
    if args.debug:
        args.working_dir = os.path.join(args.working_dir, args.suffix+"_debug")
    else:
        args.working_dir = os.path.join(args.working_dir, cur_time + args.suffix)
    if args.mode == 'train':
        os.makedirs(args.working_dir, exist_ok=True)

        # write arguments to txt
        with open(os.path.join(args.working_dir, 'args.txt'), 'w') as f:
            f.write(str(args))
    
    # set seeds and init dictionaries for model definition
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    params = {
        "mp_steps": args.mp_steps,
        "layers": args.layers,
        "num_types": 2,
        "type_dim": 2,
        "hidden_dim": args.emb_dim,
        "out_node_dim": 32,
        "in_edge_dim": 32,
        "output_pad_dim": 1,
        "pooling": args.pooling,
        "up_mlp": args.up_mlp,
        "residual": args.residual,
        "noise": args.noise,
        "transpose": args.transpose,
        "attn": args.attn,
    }

    # load data
    train_set = torch.load(os.path.join(args.data_path, 'lattice_12x12_n=1000_types=2.pt')) if args.mode == 'train' else None
    valid_set = torch.load(os.path.join(args.data_path, 'lattice_12x12_n=200_types=2.pt'))
  
    length = 144

    # initialize data loader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) if args.mode == 'train' else None
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # init model; display the number of parameters
    model = LatticeAuto(**params).double()

    print(f"Training with {torch.cuda.device_count()} GPUs!")
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n# Params: {num_params}")

    # Initialize Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=2e-4)

    # initialize scheduler
    scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)

    if args.log_dir != '' and args.mode == 'train':
        log_dir = os.path.join(args.log_dir, cur_time + args.suffix)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    best_valid_rmsd = 1000
    best_res_acc = 0
    best_valid_rmsd_epoch = 0
    best_res_acc_epoch = 0

    if args.mode == 'train':

        for epoch in range(1, args.epochs + 1):
            start = time.time()
            print("=====Epoch {}".format(epoch))
            print('Training...')
            total_loss, train_mae, res_loss, train_edge_mae, train_kl_x, train_kl_h = train(args, 
                                                                                            model,
                                                                                            train_loader,
                                                                                            optimizer,
                                                                                            args.working_dir,
                                                                                            args.loss,
                                                                                            length,
                                                                                            edgeloss_weight=args.edgeloss_weight,
                                                                                            kl_weight=args.kl_weight)

            print('Evaluating...')
            valid_edge_mae, valid_mae, rmsd, res_acc, kl_x, kl_h, pred_coord, true_coord, pred_atom_type, true_atom_type = eval(args, model, valid_loader, length)


            print("Epoch {:d}, valid_edge_mae: {:.5f}, Train_mae: {:.5f}, Validation_mae: {:.5f}, Validation_rmsd: {:.5f}, res_acc: {:.2f}, kl_x: {:.2f}, kl_h: {:.2f}, elapse: {:.5f}".
                  format(epoch, valid_edge_mae, train_mae, valid_mae, rmsd, res_acc, kl_x, kl_h, time.time() - start))

            if rmsd < best_valid_rmsd:
                best_valid_rmsd = rmsd
                best_valid_rmsd_epoch = epoch
                if args.checkpoint_dir != '':
                    print('Saving checkpoint...')
                    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                                  'scheduler_state_dict': scheduler.state_dict(), 'num_params': num_params,
                                  'mae': valid_mae, 'rmsd': rmsd, 'res_acc': res_acc}
                    checkpoint_dir = os.path.join(args.working_dir, args.checkpoint_dir)
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_bst_rmsd.pt'))

                print('Analyzing...')
                savedir = os.path.join(args.working_dir, 'reconstruction')
                if not os.path.exists(savedir):
                    os.makedirs(savedir)

            if res_acc > best_res_acc:
                best_res_acc = res_acc
                best_res_acc_epoch = epoch
                if args.checkpoint_dir != '':
                    print('Saving checkpoint...')
                    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                                  'scheduler_state_dict': scheduler.state_dict(), 'num_params': num_params,
                                  'mae': valid_mae, 'rmsd': rmsd, 'res_acc': res_acc}
                    checkpoint_dir = os.path.join(args.working_dir, args.checkpoint_dir)
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_bst_rec_acc.pt'))


            scheduler.step()
            print(f'Best validation RMSD so far: {best_valid_rmsd}, epoch: {best_valid_rmsd_epoch}')
            print(f'Best validation rec acc so far: {best_res_acc}, epoch: {best_res_acc_epoch}')

    elif args.mode == 'valid':
        print("Loading checkpoint ...")
        checkpoint = torch.load(os.path.join(args.saved_model_dir, 'checkpoint_bst_rmsd.pt'))
        print("Loading successfully, epoch: ", checkpoint['epoch'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.double()
        print('Evaluating on validation dataset...')
        valid_edge_mae, valid_mae, rmsd, res_acc,kl_x, kl_h, pred_coord, true_coord, pred_atom_type, true_atom_type = eval(args, model, valid_loader, length)
        print("Validation_rmsd: {:.5f}, res_acc: {:.2f}".
            format(rmsd, res_acc))
        

if __name__ == "__main__":
    main()
