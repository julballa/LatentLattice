# Adapted from https://github.com/divelab/AIRS/blob/main/OpenProt/LatentDiff/LatentDiff/protein_autoencoder/utils.py

import torch
import numpy as np
from typing import List
from torch import Tensor

def build_edge_idx(num_nodes):
    # Initialize edge index matrix
    E = torch.zeros((2, num_nodes * (num_nodes - 1)), dtype=torch.long)

    # Populate 1st row
    for node in range(num_nodes):
        for neighbor in range(num_nodes - 1):
            E[0, node * (num_nodes - 1) + neighbor] = node

    # Populate 2nd row
    neighbors = []
    for node in range(num_nodes):
        neighbors.append(list(np.arange(node)) + list(np.arange(node + 1, num_nodes)))
    E[1, :] = torch.Tensor([item for sublist in neighbors for item in sublist])

    return E

class KabschRMSD(torch.nn.Module):
    def __init__(self) -> None:
        super(KabschRMSD, self).__init__()

    def forward(self, coords_pred: List[Tensor], coords_true: List[Tensor]) -> Tensor:
        rmsds = []
        for coords_pred, coords_true in zip(coords_pred, coords_true):
            coords_pred_mean = coords_pred.mean(dim=0, keepdim=True)  # (1,3)
            coords_true_mean = coords_true.mean(dim=0, keepdim=True)  # (1,3)

            A = (coords_pred - coords_pred_mean).transpose(0, 1) @ (
                        coords_true - coords_true_mean)

            U, S, Vt = torch.linalg.svd(A)

            corr_mat = torch.diag(
                torch.tensor([1, 1, torch.sign(torch.det(Vt.t() @ U.t()))], device=coords_pred.device))
            rotation = (U @ corr_mat) @ Vt
            translation = coords_pred_mean - torch.t(rotation @ coords_true_mean.t())  # (1,3)

            coords_true = (rotation @ coords_true.t()).t() + translation

            rmsds.append(torch.sqrt(torch.mean(torch.sum(((coords_pred - coords_true) ** 2), dim=1))))
        return torch.tensor(rmsds).mean()

class RMSD(torch.nn.Module):
    def __init__(self) -> None:
        super(RMSD, self).__init__()

    def forward(self, coords_pred: List[Tensor], coords_true: List[Tensor]) -> Tensor:
        rmsds = []
        for coords_pred, coords_true in zip(coords_pred, coords_true):
            rmsds.append(torch.sqrt(torch.mean(torch.sum(((coords_pred - coords_true) ** 2), dim=1))))
        return torch.tensor(rmsds).mean()