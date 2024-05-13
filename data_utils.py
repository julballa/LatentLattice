import pickle
import numpy as np
import matplotlib.pyplot as plt
from lattpy import simple_square
import os
from tqdm import tqdm

import torch
from torch_geometric.data import Data

def save_lattice(lattice, atom_types, save_path):
    try:
         # Pack lattice and atom types into a dictionary
        data = {
            'lattice': lattice,
            'atom_types': atom_types
        }
        # Save using pickle
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

            print(f"File successfully saved to {save_path}")
    except Exception as e:
        print(f"Failed to save file: {e}")


# def lattice_to_pyg_data(lattice, positions, atom_types):
#     # Get the adjacency matrix and positions
#     adj_matrix = lattice.adjacency_matrix()
    
#     # Convert to edge index format expected by PyTorch Geometric
#     edge_index = adj_matrix.nonzero()
#     edge_index = torch.tensor(edge_index)

#     # Create a PyTorch Geometric data object
#     data = Data(x=atom_types, coords=positions, edge_index=edge_index, num_nodes=atom_types.shape[0])
#     return data


def plot_lattice(latt, positions, atom_types):
    # Plot the lattice with atom type color mapping
    fig, ax = plt.subplots()
    latt.plot(ax=ax, lw=None, margins=0.1, legend=None, grid=False, pscale=0.5, show_periodic=True,
              show_indices=False, index_offset=0.1, con_colors=None, adjustable='box', show=False)

    # Add color to the nodes based on atom types
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c=atom_types, cmap='viridis', s=100)
    plt.colorbar(scatter, ax=ax, label='Atom Type')

    plt.title(f'Lattice')
    plt.show()

def create_random_dataset(num_lattices, shape, num_atom_types=2, save_dir=None):
    dataset = []

    for i in range(num_lattices):
        # Initialize and build the lattice
        latt = simple_square()
        # Subtract 1 from each shape dim because build() has inclusive boundaries
        latt.build(shape=tuple([s-1 for s in shape]))
    
        positions = torch.tensor(latt.positions)

        # Generate random atom types for each node
        atom_types = torch.randint(0, num_atom_types, size=(np.prod(shape), 1))  
        # atom_types = torch.nn.functional.one_hot(atom_types, num_classes=2) # optional one-hot encoding

        pyg_latt = lattice_to_pyg_data(latt, positions, atom_types)
        dataset.append(pyg_latt)

    # Save the lattice and atom types to a file
    if save_dir is not None:
        torch.save(dataset, save_dir+f'lattice_{shape[0]}x{shape[1]}_n={num_lattices}_types={num_atom_types}.pt')

    return dataset


### Ising model dataset ###

def initialize_grid(size):
    """
    Initialize a grid for the 2D Ising model with random spins (-1 or 1).
    """
    return np.random.choice([-1, 1], size=(size, size))

def monte_carlo_step(grid, beta):
    """
    Perform one Monte Carlo step using the Metropolis algorithm.
    """
    size = grid.shape[0]
    for _ in range(size**2):
        i = np.random.randint(0, size)
        j = np.random.randint(0, size)
        S = grid[i, j]
        neighbors = grid[(i+1)%size, j] + grid[i, (j+1)%size] + grid[(i-1)%size, j] + grid[i, (j-1)%size]
        dE = 2 * S * neighbors
        if dE < 0 or np.random.rand() < np.exp(-dE * beta):
            grid[i, j] *= -1

def simulate_ising(size, temperature, steps):
    """
    Simulate the 2D Ising model.
    """
    grid = initialize_grid(size)
    beta = 1.0 / temperature

    for step in range(steps):
        monte_carlo_step(grid, beta)

    return torch.tensor(grid > 0).long()

def plot_grid(grid):
    """
    Plot the grid.
    """
    plt.imshow(grid, interpolation='nearest')
    plt.title('2D Ising Model')
    # plt.colorbar(label='Spin')
    plt.show()

def distort_positions_with_edges(positions, atom_types, edge_index, push_factor=0.2, pull_factor=0.2):
    """
    Distort positions based on atom types using PyTorch and edge index tensor.

    Parameters:
    - positions (torch.Tensor): Tensor of shape (n, 2) with x, y positions of atoms.
    - atom_types (torch.Tensor): Tensor of shape (n,) with atom types.
    - edge_index (torch.Tensor): Tensor of shape (2, m) with indices of bonded atoms.
    - push_factor (float): Factor to push away atoms of the same type.
    - pull_factor (float): Factor to pull together atoms of different types.

    Returns:
    - distorted_positions (torch.Tensor): Tensor of distorted positions.
    """
    n = positions.size(0)
    distorted_positions = positions.clone()

    for k in range(edge_index.size(1)):
        i = edge_index[0, k]
        j = edge_index[1, k]

        pos_i = positions[i]
        pos_j = positions[j]
        dist = torch.norm(pos_j - pos_i)
        direction = (pos_j - pos_i) / dist if dist != 0 else torch.zeros_like(pos_j)

        if atom_types[i] == atom_types[j]:
            # Push away
            displacement = push_factor * direction
            distorted_positions[i] -= displacement / 2
            distorted_positions[j] += displacement / 2
        else:
            # Pull together
            displacement = pull_factor * direction
            distorted_positions[i] += displacement / 2
            distorted_positions[j] -= displacement / 2

    return distorted_positions


def lattice_to_pyg_data(lattice, positions, atom_types, distort = False):
    # Get the adjacency matrix and positions
    adj_matrix = lattice.adjacency_matrix()

    # Convert to edge index format expected by PyTorch Geometric
    edge_index = adj_matrix.nonzero()
    edge_index = torch.tensor(edge_index)

    if distort:
        positions = distort_positions_with_edges(positions, atom_types, edge_index)

    # Create a PyTorch Geometric data object
    data = Data(x=atom_types, coords=positions, edge_index=edge_index, num_nodes=atom_types.shape[0])
    return data

def create_ising_dataset(num_lattices, grid_size, temperature=5.0, steps=500, save_dir=None, distort=False):
    dataset = []

    for i in tqdm(range(num_lattices)):
        latt = simple_square()
        latt.build(shape=(grid_size-1, grid_size-1))

        positions = torch.tensor(latt.positions)

        final_grid = simulate_ising(grid_size, temperature, steps)
        atom_types = final_grid.flatten().unsqueeze(1)

        pyg_latt = lattice_to_pyg_data(latt, positions, atom_types)
        dataset.append(pyg_latt)

    # Save the lattice and atom types to a file
    if save_dir is not None:
        torch.save(dataset, save_dir+f'ising_{grid_size}x{grid_size}_n={num_lattices}.pt')

    return dataset