"""
DKST_utils.py

This module provides functions for tokenization, padding of knowledge structures/ surmise relations. 
It also includes a method for aggregating response data by frequency and utility functions DKST datasets.
"""

# Standard library imports
import os
import json
import random
import time
import pickle
# Local imports 
from dkst.utils.relations import S_to_matrix
from dkst.utils.set_operations import powerset
from dkst.utils.set_operations import matrix_to_K
from dkst.utils.KST_utils import * 
# Third-party imports
import numpy as np


# tokenization
def create_state2idx(m, eos_token="<eos>", pad_token="<pad>"):  
    """
    Creates an index for all elements of the powerset for \\|Q\\|=m in alphabetic notation and conventional order.
    Also includes "<eos>" and "<pad>" tokens.
    
    :param m: Number of elements in the domain Q of interest.
    :type m: int
    :param eos_token: Token to represent the end of a sequence, defaults to "<eos>".
    :type eos_token: str
    :param pad_token: Token to represent padding, defaults to "<pad>".
    :type pad_token: str
    
    :return: Dictionary mapping states to indices.
    :rtype: dict[str, int]
    """
    powerset_list = powerset(m)
    state2idx = {state: idx for idx, state in enumerate(powerset_list)}
    state2idx[eos_token] = 2**m
    state2idx[pad_token] = 2**m + 1

    return state2idx

def encode_sequence(sequence, state2idx):
    """
    Encodes a sequence of knowledge states as a sequence of indices given a dictionary for tokenization.
    
    :param sequence: Sequence of states.
    :type sequence: list[str] or array-like
    :param state2idx: Dictionary mapping states to indices.
    :type state2idx: dict[str, int]
    
    :return: sequence of indices
    :rtype: list[int]
    """
    return [state2idx[state] for state in sequence]

def decode_sequence(sequence, idx2state):
    """
    Decodes a sequence of indices (tokens) to a sequence of knowledge states given a reverse dictionary for tokenization.
    
    :param sequence: Sequence of indices.
    :type sequence: list[int] or array-like
    :param idx2state: Dictionary mapping indices to states.
    :type idx2state: dict[int, str]
    
    :return: sequence of states
    :rtype: list[str]
    """
    return encode_sequence(sequence, idx2state)

def encode_sequences(sequences, state2idx):
    """
    Encodes a nested sequence of knowledge states as a nested sequence of indices given a dictionary for tokenization.
    
    :param sequences: Nested sequence of states.
    :type sequences: list[list[str]] or array-like
    :param state2idx: Dictionary mapping states to indices.
    :type state2idx: dict[str, int]
    
    :return: Nested sequence of indices.
    :rtype: np.ndarray
    """
    return np.array([
        encode_sequence(sequence, state2idx)
        for sequence in sequences
    ])

def decode_sequences(sequences, idx2state):
    """
    Decodes a nested sequence of indices (tokens) to a nested sequence of knowledge states given a reverse dictionary for tokenization.
    
    :param sequences: nested sequence of indices
    :type sequences: list[list[int]] or array-like
    :param idx2state: dictionary mapping indices to states
    :type idx2state: dict[int, str]
    
    :return: nested sequence of states
    :rtype: list[list[str]]
    """
    return [decode_sequence(sequence, idx2state) for sequence in sequences]

# padding 
def pad_relations(matrices):
    """
    Pad a list of binary surmise relation matrices to the same size by adding zeros.
    The result is a 3D array of shape (M, M, N) where M is the maximum number of items in the domain and N is the number of input matrices
    (E.g. [[1,2],[3,4]] -> [[1,2,0,0], [3,4,0,0], [0,0,0,0], [0,0,0,0]] for M=4).
    
    :param matrices: list of binary surmise relation matrices or list of sets of tuples for the relation.
    :type matrices: list[np.ndarray] or  list[list[tuple[int, int]]]
    
    :return: 3D array of padded binary surmise relation matrices.
    :rtype: np.ndarray
    """
    # convert relation from set notation to matrix notation, if necessary
    if isinstance(matrices[0][0], tuple):
        matrices = [S_to_matrix(matrix) for matrix in matrices]
    # determine maximum number of items
    M = max(len(matrix) for matrix in matrices)
    
    # Initialize 3D array for resulting matrices
    padded3D = np.zeros((M, M, len(matrices)), dtype=int)

    for num, matrix in enumerate(matrices):
        padded = np.zeros((M, M), dtype=int)
        for i in range(M):
            if i < len(matrix):
                row = np.concatenate((np.array(matrix[i], dtype=int), np.array([0] * (M - len(matrix[i])), dtype=int))) # Pad row with zeros
            else:
                row = np.array([0] * M, dtype=int)  # Pad columns with zeros
            padded[i] = row
        padded3D[:,:,num] = padded

    return padded3D

def pad_sequences(sequences, pad_token="<pad>", eos_token="<eos>", vocab_size=None):
    """
    Pads a list of sequences of knowledge states (in alphabetic notation) to the maximum number of states in the respective domain.
    Also adds an <eos> token and ensures the empty state is included.
    The result is a 2D array of shape (N, K) where N is the number of input sequences (ordered) and K is the maximum sequence length (n states +1).
    (E.g. ["c","bc","abc"] -> ["", "c","bc","abc","<eos>""<pad>","<pad>","<pad>","<pad>","<pad>","<pad>"]  for M=3).
    
    Note: All items need to be included in the sequneces if vocab_size is None.
    
    :param sequences: List of sequences of knowledge states.
    :type sequences: list[list[str]] or array-like
    :param pad_token: Token to represent padding.
    :type pad_token: str
    :param vocab_size: Number of unique states in the domain.
    :type vocab_size: int
    
    :return: 2D array of padded sequences.
    :rtype: np.ndarray[str]
    """
    # Ensure sequences is a list of lists, not a NumPy array
    if isinstance(sequences[0], np.ndarray):
        sequences = [list(seq) for seq in sequences]

    m = len(set(''.join([string for lst in sequences for string in lst if "<" not in string]))) # Maximum number of items
    
    if vocab_size is None: 
        vocab_size = 2 ** m + 2 # Add 2 for <eos> and <pad> tokens
    
    # Add empty state to each sequence if not included
    for i, seq in enumerate(sequences):
        if "" not in seq: 
            sequences[i] = [""] + seq

    # Allocate array prefilled with padding tokens 
    padded_sequences = np.full((len(sequences), vocab_size - 1), pad_token, dtype=f'<U{max(5,m)}') # set max length of the strings to m, but at least 5 for <eos> and <pad>  

    # Fill in each sequence and add the <eos> token
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq  # Fill in the original 
        if seq[-1] != eos_token:
            padded_sequences[i, len(seq)] = eos_token  # Add the <eos> token

    return padded_sequences

def pad_K(matrix, m=None): 
    """
    Pad a binary knowledge structure in matrix notation to uniform length 2**2**\\|Q\\| by adding zero-states.
    
    :param matrix: Binary knowledge structure in matrix notation \\|K\\| x \\|Q\\|.
    :type matrix: np.ndarray or list[list[int]]
    :param m: Domain size for padding, automatically inferred from the given knowledge structure at default.
    :type m: int
    :return: Padded binary knowledge structure in matrix notation 2**\\|Q\\| x \\|Q\\|.
    """
    # check input 
    if not isinstance(matrix[0,0], (int, np.integer)):
        raise ValueError("Input must be a binary matrix of type int.")
    
    if m is None:
        m = matrix.shape[1]  # Number of items
    n = 2 ** m  # Number of possible states
    
    # Pads the first dimension (rows) with 0 elements in the front and n - |K| elements ; (0, n - matrix.shape[0]).
    # The second dimension (columns) is unchanged; (0, 0).
    padded = np.pad(matrix, ((0, n - matrix.shape[0]), (0, 0)), mode='constant')
    
    return padded

def trim_K(matrix):
    """
    Removes trailing zero rows from a binary knowledge structure in matrix notation.
    
    :param matrix: Padded knowledge structure in matrix notation.
    :type matrix: np.ndarray 
    :return: Trimmed binary knowledge structure in matrix notation.
    :rtype: np.ndarray
    """
    # Check input
    if not isinstance(matrix[0, 0], (int, np.integer)):
        raise ValueError("Input must be a binary matrix of type int.")
    
    # Find the last row with any non-zero elements
    last_nonzero_row = np.where(np.any(matrix != 0, axis=1))[0][-1]
    
    # Slice the array up to and including the last non-zero row
    return matrix[:last_nonzero_row + 1]

# data aggregation 
def calculate_counts(response_patterns, state2idx=None, standadize=False):
    """
    Calculates the counts of all response patterns/ knowledge states in matrix notation.
    Can be used to aggregate observed response data by frequency as input to an ANN.

    :param response_patterns: List of response patterns represented as binary vectors
    :type response_patterns: list[np.ndarray[int]] or array-like or list[str] 
    :param state2idx: Dictionary mapping states to indices, created if not provided
    :type state2idx: dict[str, int]
    :param standadize: Whether to standardize the counts, defaults to False
    :type standadize: bool, optional
    
    :return: Array of counts (2 ** num_items) associated to all possible response patterns in conventional order
    :rtype: np.ndarray
    """
    if isinstance(response_patterns[0], np.ndarray): 
        m = len(response_patterns[0])  # Number of items
    else:
        m = len(set("".join([string for string in response_patterns if "<" not in string])))  # Number of items
    
    # Generate state2idx given m, if not provided
    if state2idx is None:
        state2idx = create_state2idx(m)
    # Initialize counts array
    counts = np.zeros(2 ** m, dtype=int)  
    
    if isinstance(response_patterns[0], np.ndarray): 
        response_patterns_alphabetic = matrix_to_K(response_patterns)
    else:
        response_patterns_alphabetic = response_patterns
    
    # Count the occurrences of each pattern
    for pattern in response_patterns_alphabetic:
        idx = state2idx[pattern]
        counts[idx] += 1 # Increment the count for this index
    
    if standadize:
        counts = (counts - counts.mean()) / counts.std()
    
    return counts

# pytorch datasets
def check_config(config):
    """
    Checks if a DKST dataset configuration is valid.
    
    :param config: Dataset configuration with all necessary parameters
    :type config: dict
    :return: Whether the configuration is valid
    :rtype: bool
    """
    
    required_keys = {
        "ID": str,
        "d_type": str,
        "device": str,
        "seed": (int, type(None)),
        "antisymmetric": bool,
        "m": int,
        "n_structures": int,
        "n_patterns": int,
        "no_dublicates": bool,
        "EOS_TOKEN": str,
        "PAD_TOKEN": str,
        "standadized": bool,
        "beta": (float, list),
        "eta": (float, list),
        "factor": (int, float, type(None))
    }
    
    # Check for missing keys
    for key, expected_type in required_keys.items():
        if key not in config:
            print(f"Missing key: {key}")
            return False
        if not isinstance(config[key], expected_type):
            print(f"Incorrect type for key: {key}. Expected {expected_type}, got {type(config[key])}")
            return False
    
    # Check specific value constraints
    if config["device"] not in ["cpu", "cuda", "mps"]:
        print("Invalid device. Must be 'cpu', 'cuda' or 'mps'")
        return False
    
    if not isinstance(config["beta"], (int, float)):
        if not isinstance(config["beta"], list):
            print("Parameters 'beta' must be int or float")
            return False
        else:
            if not all(isinstance(item, (int, float)) for item in config["beta"]):
                print("All elements in 'beta' must be int or float")
                return False
            
    if not isinstance(config["eta"], (int, float)):
        if not isinstance(config["eta"], list):
            print("Parameters 'eta' must be int or float")
            return False
        else:
            if not all(isinstance(item, (int, float)) for item in config["eta"]):
                print("All elements in 'eta' must be int or float")
                return False
    return True

