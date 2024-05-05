import pickle
import numpy as np
import matplotlib.pyplot as plt
from lattpy import simple_square
import os

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


def lattice_to_pyg_data(lattice, positions, atom_types):
    # Get the adjacency matrix and positions
    adj_matrix = lattice.adjacency_matrix()
    
    # Convert to edge index format expected by PyTorch Geometric
    edge_index = adj_matrix.nonzero()
    edge_index = torch.tensor(edge_index)

    # Create a PyTorch Geometric data object
    data = Data(x=atom_types, coords=positions, edge_index=edge_index, num_nodes=x.shape[0])
    return data


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

def create_dataset(num_lattices, shape, num_atom_types=2, save_dir=None):
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