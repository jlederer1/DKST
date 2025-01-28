"""
DKST_datasets.py

This module provides different types of DKST datasets.
"""

# Standard library imports
import os
import json
# Local imports 
from dkst.utils.set_operations import sort_K, matrix_to_K
from dkst.utils.KST_utils import *
from dkst.utils.DKST_utils import * 
# Third-party imports
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class DKSTDataset02(Dataset):
    """
    A simple dataset of knowledge structures, conditional probabilities and responses data aggregates for training K-net.
    Loads or samples knowledge structures and response data according to given dataset configuration file.
    Retrieves a quadruple of 1D tensors, each of length 2^m, where m is the domain size.
    
    :param config_path: Path to the dataset configuration file.
    :type config_path: str
    :param baseset_path: Path to the baseset file (optional).
    :type baseset_path: str
    
    :return: A dataset of knowledge structures and response aggregates.
    :rtype: Custom torch.utils.data.Dataset
    """
    
    def __init__(self, config_path):
        # Dataset configuration 
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}.")
        with open(config_path, 'r') as f:
            config = json.load(f)
        if check_config(config):
            self.config = config
        
        # data
        self.structures = sample_knowledge_structures(m=config["m"], num_samples=config['n_structures'], no_dublicates=config['no_dublicates'])
        self.conditionals = np.array([compute_conditionals(self.structures[:, :, i]) for i in range(self.structures.shape[2])])
        self.responses = sample_response_data(self.structures, config["n_patterns"], None, config["beta"], config["eta"], config["seed"], config["factor"])
        self.state2idx = create_state2idx(config["m"], config['EOS_TOKEN'], config['PAD_TOKEN'])
        self.observations = [calculate_counts(matrix, self.state2idx, config['standadized']) for matrix in self.responses]
        self.vocab_size = len(self.state2idx.values())
        self.sequences = [[''.join(map(str, state)) for state in sort_K(matrix_to_K(trim_K(self.structures[:, :, i])))] for i in range(self.structures.shape[2])]
        self.sequences = pad_sequences(self.sequences, vocab_size=len(self.state2idx.values()))
        self.sequences = encode_sequences(self.sequences, self.state2idx)

    def __getitem__(self, i):
        # for regression
        conditional = torch.tensor(self.conditionals[i].flatten(), dtype=torch.float32)
        # for autoregressive knowledge structure generation
        input_seq, target_seq = torch.tensor(self.sequences[i, :-1], dtype=torch.long), torch.tensor(self.sequences[i, 1:], dtype=torch.long)
        # for prediction
        observation = torch.tensor(self.observations[i], dtype=torch.float32)
        return conditional, input_seq, target_seq, observation
    
    def __len__(self):
        return self.structures.shape[-1] 

def collate_fn(batch):
    """
    Collate function for DataLoader.
    Does not change the data location device.
    
    :param batch: A batch of data.
    :type batch: list of tensor tuples
    :param device: Device to move tensors to (optional).
    :type device: str

    :return: A batch of data.
    :rtype: tuple
    """
    # separate batch into lists for each component of the data
    S, input_seq, target_seq, C = zip(*batch)
    # collate lists into single tensor and move to correct device
    S = torch.stack(S)
    input_seq = torch.stack(input_seq)
    target_seq = torch.stack(target_seq)
    C = torch.stack(C)
    return S, input_seq, target_seq, C

def save_dataset(dataset, path=None, data_type="train"):
    """
    Save a torch dataset to file.

    :param dataset: Dataset to save.
    :type dataset: torch.utils.data.Dataset
    :param path: Path to save the dataset.
    :type path: str
    :param data_type: Type of the dataset (train, val, test).
    :type data_type: str
    """
    if path is None:
        # Get the directory of the current script or notebook
        base_dir = Path(__file__).resolve().parent.parent
        dataset_dir = base_dir / "data" / "datasets"

        # Find highest index in directory and increment
        files = [f for f in dataset_dir.iterdir() if f.is_file() and f.suffix == ".pth" and data_type in f.stem]
        if not files:
            path = dataset_dir / f"dataset_{data_type}_00.pth"
        else:
            files.sort()
            last_file = files[-1]
            last_index = int(last_file.stem.split("_")[-1])
            new_index = last_index + 1
            path = dataset_dir / f"dataset_{data_type}_{new_index:02d}.pth"

    torch.save(dataset, path)
    print(f"Dataset saved to {path}")

def load_dataset(file_name):
    """
    Load a torch dataset from file.

    :param file_name: Name of the file to load.
    :type file_name: str

    :return: The loaded dataset.
    :rtype: torch.utils.data.Dataset
    """
    # Get the dataset directory path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "data/datasets")
    path = os.path.join(dataset_dir, file_name)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file or directory: '{path}'")
    
    return torch.load(path)