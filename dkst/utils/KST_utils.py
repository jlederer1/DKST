"""
KST_utils.py

This module provides functions to generate, sample, manipulate, and visualize knowledge structures and quasiorders,
and to to convert between them. Relevant combinatoric vaues are also included, 
for the number of posets, prosets, knowledge spaces and general knowledge structures and states for domain size <10.

Note on notation:
- Knowledge state k is represented either as string of respective item (e.g. "abd"), or as a binary vector (e.g. 11010).
- Knowledge structure K may be a list of states (alphabetically ordered), or a binary states matrix.
- Quasiorder S can be represented as list of string-tuples (alphabeticly ordered implication pairs), or as binary adjacency matrix.

To do: 
- representative sampling of knowledge spaces.
- check again paremeter types in docstrings and test cases...
- introduce consistent naming conventions knowledge_structure quasiorder vs. surmise_relation vs quasi_order, also in the other modules...
- work with 3D arrays instead of lists of uniform arrays where possible. 
"""

# Standard library imports
import random
import time
from itertools import product
import math
# Local imports 
from dkst.utils.relations import is_reflexive, is_transitive, is_antisymmetric, S_to_matrix
from dkst.utils.set_operations import powerset_binary
from dkst.utils.set_operations import K_to_matrix
# Third-party imports
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import scipy.special


# Combinatorics (see Online Encyclopedia of Integer Sequences, https://oeis.org/ with ids in brackets) 
num_prosets = {2: 4,    3: 29,   4: 355,  5: 6942, 6: 209527, 7: 9535241, 8:642779354, 9:63260289423} # Number of prosets on n items [A000798]
num_posets =  {2: 3,    3: 19,   4: 219,  5: 4231, 6: 130023, 7: 6129859, 8:431723379, 9:44511042511} # Number of posets on n items [A001035]
num_spaces = {2: 4,    3: 45,   4: 2271,  5: 1373701, 6: 75965474236, 7: 14087647703920103947} # Dedekind numbers of union-closed families [A102894]
num_structures = {n : 2 ** (2 ** n - 2) for n in range(2, 10)} # Number of general knowledge structures 
num_states =  {n : 2 ** n for n in range(2, 10)} # Number of possible states 

# Unitility functions

# def transform_probabilities(probabilities, exponent=1):
#     """Apply a power transformation to an array of probabilities."""
#     transformed = [p**exponent for p in probabilities]
#     total = sum(transformed)
#     normalized_transformed = [p / total for p in transformed]
#     return normalized_transformed

# Conversion
def implications_to_states(S):
    """
    Transforms a surmise relation into the corresponding quasi-ordinal knowledge space.
    This function is based on an implementation from the kst package by Milan Segedinac, 
    see https://github.com/milansegedinac/kst/blob/master/learning_spaces/kst/imp2state.py !

    :param S: Numpy array or list of str-tuples representing a surmise relation.
    :return: Corresponding quasiordinal knowledge space represented as binary numpy matrix.
    """
    # check if structure is given in matrix or set notation 
    if not isinstance(S, np.ndarray):
        S = S_to_matrix(S)
    
    # check if quasiorder is reflexive and transitive, otherwise return error
    if not (is_reflexive(S) and is_transitive(S)):
        raise ValueError("The given relation is not reflexive and transitive.")
    
    items = S.shape[0] 

    # Find implications from the surmise matrix, store as set of tuples
    implications = set()
    for i in range(items):
        for j in range(items):
            if i != j and S[i, j] == 1:
                implications.add((i, j))

    # Transformation from Implications to Knowledge States
    R_2 = np.ones((items, items))
    for i in range(items):
        for j in range(items):
            if (i != j) and ((i, j) not in implications):
                R_2[j, i] = 0

    base = []
    for i in range(items):
        tmp = []
        for j in range(items):
            if R_2[i, j] == 1:
                tmp.append(j)
        base.insert(i, tmp)

    base_list = []
    for i in range(items):
        base_list.insert(i, set())
        for j in range(len(base[i])):
            base_list[i].update(frozenset([base[i][j]]))

    G = []
    G.insert(0, {frozenset()})
    G.insert(1, set())
    for i in range(len(base[0])):
        G[1].update(frozenset([base[0][i]]))
    G[1] = {frozenset(), frozenset(G[1])}

    for i in range(1, items):
        H = {frozenset()}
        for j in G[i]:
            if not base_list[i].issubset(j):
                for d in range(i):
                    if base_list[d].issubset(j.union(base_list[i])):
                        if base_list[d].issubset(j):
                            H.update(frozenset([j.union(base_list[i])]))
                    if not base_list[d].issubset(j.union(base_list[i])):
                        H.update(frozenset([j.union(base_list[i])]))
        G.insert(i+1, G[i].union(H))

    P = np.zeros((len(G[items]), items), dtype=np.int8)
    i = 0
    sorted_g = [list(i) for i in G[items]]
    sorted_g.sort(key=lambda x: (len(x), x))

    for k in sorted_g:
        for j in range(items):
            if j in k:
                P[i, j] = 1
        i += 1

    return P

def states_to_implications(states):
    """
    Inverse transformation from a quasiordinal set of knowledge states to a surmise relation matrix.
    
    :param states: Quasiordinal knowledge structure in binary matrix notation or as list of states.
    :return: numpy array representing the surmise relation
    """
    # check if structure is given in matrix or set notation 
    if isinstance(states[0], str):
        K_to_matrix(states)
    
    m = states.shape[1] # number of items
    surmise_matrix = np.zeros((m, m), dtype=int) # initialize surmise matrix

    # Analyze each state to determine possible prerequisites
    for state in states:
        known_items = np.where(state == 1)[0]
        for j in known_items:
            # Set i as a prerequisite for j if i is known whenever j is known
            for i in known_items:
                if i != j:
                    surmise_matrix[i, j] = 1

    # Optimize by checking consistency across all states
    # If i is not a prerequisite for j in any state where j is known, set to 0
    for i in range(m):
        for j in range(m):
            if i != j:
                prerequisite = True
                for state in states:
                    if state[j] == 1 and state[i] == 0:
                        prerequisite = False
                        break
                if not prerequisite:
                    surmise_matrix[i, j] = 0
    
    # Ensure reflexivity
    np.fill_diagonal(surmise_matrix, 1)

    return surmise_matrix


# PKST
def compute_r(R, k, beta, eta):
    """
    Computes the probability of the response pattern R given the knowledge state k and error rates beta and eta.
    
    :param R: Response pattern as a binary vector.
    :type R: Numpy.array.
    :param K: Knowledge state as a binary vector.
    :type K: Numpy.array.
    :param beta: Probability of a careless error.
    :type beta: Float (if constant), or array-like.
    :param eta: Probability of a lucky guess.
    :type eta: Float (if constant), or array-like
    
    :return: Probability of the response pattern R given the knowledge state k.
    :rtype: Float.
    """
    # Convert to boolean vectors for logical operations
    R = R.astype(bool)
    K = k.astype(bool)
    
    # mark all possible responses and errors per item as boolean vectors
    LG_response = R & (~K)
    CE_response = K & (~R)
    correct_response = R & K
    incorrect_response = (~K) & (~R) 
    
    # Initialize probability of response pattern R given the knowledge state K
    prob = 1
    
    # if eta and beta are constant across items
    if not isinstance(eta, (list, np.ndarray)):
        for i in range(len(R)):
            if LG_response[i]: 
                prob *= eta
            if CE_response[i]:
                prob *= beta
            if correct_response[i]:
                prob *= 1 - beta
            if incorrect_response[i]:
                prob *= 1 - eta
    
    # if eta and beta are item-specific 
    else: 
        for i in range(len(R)):
            if LG_response[i]: 
                prob *= eta[i]
            if CE_response[i]:
                prob *= beta[i]
            if correct_response[i]:
                prob *= 1 - beta[i]
            if incorrect_response[i]:
                prob *= 1 - eta[i]
    
    return prob

def compute_rho(R, K_matrix, beta, eta, p_k=None):
    """
    Computes the conditional probability of a response pattern R given a set of knowledge states in binary matrix notation.
    
    :param R: Response pattern as a binary vector.
    :type R: Numpy.array.
    :param K_matrix: Knowledge structure represented as a binary matrix (n x m), possibly flattened or padded with empty states.
    :type K_matrix: Numpy.array.
    :param beta: Probabilities of careless errors for each item.
    :type beta: Float (if constant) or array-like.
    :param eta: Probabilities of lucky guesses for each item.
    :type eta: Float (if constant) or array-like.
    :param p_K: Prior probabilities for each knowledge state (default is uniform).
    :type p_K: Array-like.
    
    :return: Conditional probability of the response pattern R given the knowledge structure K.
    :rtype: Float.
    """
    # reshape if K is a flattened array
    if len(K_matrix.shape) == 1: 
        K_matrix = K_matrix.reshape(-1, len(R))
    
    if p_k is None:
        n_unique_states = len(np.unique(K_matrix, axis=0))
        p_k = [1 / n_unique_states for _ in range(n_unique_states)] # Uniform distribution for default prior state probabilities p(K)

    # Initialize probability of response pattern R given the knowledge structure
    rho_R = 0

    # Sum over all knowledge states
    for i in range(len(p_k)): 
        state = K_matrix[i]
        r_value = compute_r(R, state, beta, eta)
        rho_R += r_value * p_k[i]
    
    return rho_R

def compute_conditionals(K, p_k=None, beta=None, eta=None, factor=None, powerset=None):
    """
    Computes the conditional probabilities of all response patterns given a knowledge structure.
    
    :param K: Knowledge structure represented as an ordered (!) binary matrix (n x m), may be padded with empty states.
    :type K: numpy.ndarray
    :param p_k: Prior probabilities of knowledge states (default is uniform).
    :type p_k: array-like
    :param beta: Probabilities of careless errors (default is 0.1 per item).
    :type beta: float or array-like
    :param eta: Probabilities of lucky guesses (default is 0.1 per item).
    :type eta: float or array-like
    :param factor: Factor for random variation of beta and eta values (optional), None for constant error rates.
    :type factor: float
    :param powerset: Powerset of binary vectors (optional), generated if not provided.
    :type powerset: numpy.ndarray
    
    :return: Array of conditional probabilities for all response patterns given the knowledge structure.
    :rtype: numpy.array[float]
    """
    m = K.shape[1] # number of items
    n = 2 ** m # maximum number of states 
    if powerset is None:
        powerset = powerset_binary(m) # powerset for exhaustive sampling 
    
    # Initialize dictionary for posterior state probabilities rho(pattern) 
    keys = [''.join(str(x) for x in pattern) for pattern in powerset]
    vals = np.zeros(n)
    rho = dict(zip(keys, vals))

    # Initialize prior state probabilities and (item-specific) error rates
    if p_k is None:
        num_states = len(np.unique(K, axis=0))
        p_k = [1 / num_states for _ in range(num_states)] # Uniform distribution for default prior state probabilities p(K)
    if beta is None:
        beta = 0.1  # Default beta values
    if eta is None:
        eta = 0.1	# Default eta values 
    
    # if factor is given, we randomly vary error probabilities by resampling (item-specific) from a normal distribution with mean='error' and std='error' * factor
    if factor is not None:
        beta = abs(np.random.normal(beta, beta * factor, m))
    if factor is not None:
        eta = abs(np.random.normal(eta, eta * factor, m))
    
    # Compute conditional probability of each response pattern given the knowledge structure
    for i, pattern in enumerate(rho.keys()):
        rho[pattern] = compute_rho(powerset[i], K, beta, eta, p_k) 
    
    conditionals = np.array(list(rho.values()))

    return conditionals

def blim(K, num_samples=10, p_k=None, beta=None, eta=None, seed=None, factor=None):
    """
    Samples binary response patterns from a knowledge structure using the Basic Local Independence Model (BLIM).

    :param K: Knowledge structure represented as an ordered (!) binary matrix (n x m), may be padded with empty states.
    :type K: numpy.ndarray 
    :param num_samples: Number of response patterns to sample.
    :type num_samples: int
    :param p_k: Prior probabilities of knowledge states (default is uniform).
    :type p_k: array-like
    :param beta: Probabilities of careless errors (default is 0.1 per item).
    :type beta: float or array-like
    :param eta: Probabilities of lucky guesses (default is 0.1 per item).
    :type eta: float or array-like
    :param seed: Seed for random number generation (optional).
    :type seed: int
    :param factor: Factor for random variation of beta and eta values (optional), None for constant error rates.
    :type factor: float
    
    :return: Matrix with sampled response patterns represented as binary row vectors.
    :rtype: numpy.ndarray
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    m = K.shape[1] # number of items
    n = 2 ** m # maximum number of states 
    powerset_matrix = powerset_binary(m) # powerset for exhaustive sampling 
    
    # Initialize list of response patterns
    patterns = []
    
    conditionals = compute_conditionals(K, p_k=p_k, beta=beta, eta=eta, factor=factor, powerset=powerset_matrix)
    
    # Sample response patterns exhaustively from the powerset
    for _ in range(num_samples):
        # choose new patterns from powerset according to rho-distribution (with replacement)
        pattern_idx = np.random.choice(2**m, p=conditionals)
        pattern = powerset_matrix[pattern_idx]
        patterns.append(pattern)
    
    return np.array(patterns)

# Utilities for simulation studies
def sample_response_data(knowledge_structures, num_samples=10, p_k=None, beta=None, eta=None, seed=None, factor=None):
    """
    Samples binary response patterns from a list of knowledge structures using the Basic Local Independence Model (BLIM).

    :param knowledge_structures: List of knowledge structures in matrix notation, may be padded with empty states
    :type knowledge_structures: list[numpy.ndarray] or np.ndarray 
    :param num_samples: Number of response patterns to sample
    :type num_samples: int or array-like
    :param p_k: Prior probabilities of knowledge states (default is uniform) 
    :type p_k: list[np.ndarray] or array-like (?)
    :param beta: Probabilities of careless errors (default is 0.1 per item)
    :type beta: float, array-like or np.ndarray 
    :param eta: Probabilities of lucky guesses (default is 0.1 per item)
    :type eta: float, array-like or np.ndarray
    :param seed: Seed for random number generation (optional)
    :type seed: int
    :param factor: Factor for random variation of beta and eta values (optional), None for constant error rates.
    :type factor: float
    
    :return: List of matrices with sampled response patterns represented as binary row vectors
    :rtype: list[numpy.ndarray]
    """
    response_data = []

    # If input is a single matrix, swap dimensions for iteration over knowledge structures
    if len(knowledge_structures.shape) == 3:
        knowledge_structures = np.moveaxis(knowledge_structures, -1, 0)

    # Flexible handling of num_samples
    if isinstance(num_samples, int):
        n_samples = [num_samples] * len(knowledge_structures)
    else:
        n_samples = num_samples

    # Flexible handling of p_k, beta and eta
    if isinstance(p_k,np.ndarray):
        if len(p_k.shape) == 1:
            p_k = [p_k] * len(knowledge_structures)
        else: 
            p_k = np.moveaxis(p_k, -1, 0)
    elif p_k is None:
        p_k = [None] * len(knowledge_structures)

    if isinstance(beta, np.ndarray) and len(beta.shape) == 1:
        beta = [beta] * len(knowledge_structures)
    if isinstance(eta, np.ndarray) and len(eta.shape) == 1:
        eta = [eta] * len(knowledge_structures)
    if isinstance(beta, float) or beta is None:
        beta = [beta] * len(knowledge_structures)
    if isinstance(eta, float) or eta is None:
        eta = [eta] * len(knowledge_structures)

    # Sample response data for each knowledge structure witth appropriate parameters
    for i, K in enumerate(knowledge_structures):
        patterns = blim(
            K, 
            n_samples[i], 
            p_k[i], 
            beta[i], 
            eta[i], 
            seed, 
            factor
            )
        response_data.append(patterns)
    
    return response_data

def sample_knowledge_structures(m, num_samples, no_dublicates=False):
    """
    Generates a representative sample of knowledge structures in binary notation for a given domain size m.
    First target cardinalities are sampled according to a binomial distribution, then structures are sampled with equal probability for each state.
    The resulting dataset may be used for training and testing DL models on general knowledge structures.
    
    :param m: Number of items in the domain.
    :type m: int
    :param num_samples: Number of knowledge structures to sample.
    :type num_samples: int
    :param no_dublicates: Whether to include no duplicate structures in the dataset.
    :type no_dublicates: bool
    :return: Dataset of the sampled knowledge structures as a 3D numpy array, rows representing states, columns corresponding to items.
    :rtype: numpy.ndarray
    """
    # Sample cardinalities representatively
    counts = [0, 0, 1]  # for every domain size m, there is only one structure with 2 states only
    for k in range(1, 2**m - 1): # there is a maximum of 2**m states 
        #count = math.comb(2**m - 2, k) # depriceded in python 3.7 ror so...
        count = scipy.special.comb(2**m - 2, k, exact=True) # number of possible knowledge structures for m items and k states = (2^m - 2 choose k)
        counts.append(count)
    
    total = sum(counts)
    probabilities = [count / total for count in counts]
    sampled_integers = random.choices(range(len(probabilities)), probabilities, k=num_samples)
    
    sampled_integers = sorted(sampled_integers)
    unique, counts = np.unique(np.array(sampled_integers), return_counts=True)
    # this dict tells us how many structures to sample for each number of states
    counts_dict = dict(zip(unique, counts))
    
    # Initialize 3D array for first dataset entry
    dataset = np.zeros((2**m, m, 1), dtype=int) 
    
    for cardinality, n_samples in counts_dict.items():
        # Initalize 3D array for structures to sample for the given cardinality
        structures = np.zeros((2**m, m, 1), dtype=int)
        
        while structures.shape[2] < n_samples + 1:
            # Initialize matrix for new knowledge structure
            K = np.zeros((2**m, m), dtype=int)
            states = ["0" * m, "1" * m] + [bin(random.getrandbits(m))[2:].zfill(m) for _ in range(cardinality - 2)] # sampling states in binary is fastest
            unique_states = list(set(states))
            
            # replace duplicates with additional new states
            while len(unique_states) < len(states):
                state = bin(random.getrandbits(m))[2:].zfill(m)
                if state not in unique_states:
                    unique_states.append(state)
            unique_states = sorted(unique_states)
            
            # Write to array
            for i, state in enumerate(unique_states):
                K[i] = np.array([int(bit) for bit in state], dtype=int)
            K = np.expand_dims(K, axis=2)
            
            # Check for dublicate structures with same cardinality, if required
            if no_dublicates == True:
                for i in range(structures.shape[2]):
                    if np.all(structures[:, :, i] == K[:, :, 0]):
                        # resample current K if already in dataset
                        break
                else: 
                    # Add K to dataset, if no dublicates found
                    structures = np.concatenate((structures, K), axis=2)
                    
            else: 
                structures = np.concatenate((structures, K), axis=2)
        
        # remove initial zero matrix and add the rest to the dataset
        structures = structures[:, :, 1:]
        dataset = np.concatenate((dataset, structures), axis=2)

    # remove initial zero entry
    return dataset[:, :, 1:]

def generate_all_surmise_relations(n, antisymmetric=False):
    """
    Generates all valid surmise relations for a given domain size n (feasable between 2 and 5)
    A surmise relation is a binary square matrix that represents a quasi-order.
    
    :param n: Number of items in the domain.
    :type n: int
    :param antisymmetric: Whether to generate only antisymmetric surmise relations.
    :type antisymmetric: bool
    :return: List of valid surmise relations.
    :rtype: list[numpy.ndarray]
    """
    if n < 2 or n > 5:
        raise ValueError("n must be between 2 and 5")

    valid_matrices = []
    total_matrices = 2 ** (n * n)

    for matrix_tuple in tqdm(product([0, 1], repeat=n*n), total=total_matrices, desc="Processing"):
        matrix = np.array(matrix_tuple).reshape(n, n)
        # Check reflexivity, transitivity, antisymmetry
        if np.all(np.diag(matrix) == 1) and is_transitive(matrix):
            if antisymmetric and not is_antisymmetric(matrix):
                continue
            valid_matrices.append(matrix)

    return valid_matrices

def extend_quasi_order(base_quasi_order):
    """
    Extend a given quasi-order by one item, ensuring transitivity.
    This generates all possible extensions of a quasi-order given in binary square matrix notation.
    The implementation follows Ünlü, A., & Schrepp, M. (2017). Techniques for sampling quasi-orders. Arch. Data Sci. A, 2, 163-182. 
    
    :param base_quasi_order: Binary square matrix representing a quasi-order.
    :type base_quasi_order: numpy.ndarray
    :return: List of extended quasi-orders.
    :rtype: list[numpy.ndarray]
    """
    n = base_quasi_order.shape[0] + 1
    extended_matrices = []
    # Generate all possible extensions by adding one row and one column
    for extension in product([0, 1], repeat=2*(n-1)):
        extended_matrix = np.zeros((n, n), dtype=int)
        extended_matrix[:n-1, :n-1] = base_quasi_order
        # Set reflexivity for the new item
        extended_matrix[-1, -1] = 1
        # Fill in the new row and column
        extended_matrix[-1, :n-1] = extension[:n-1]
        extended_matrix[:n-1, -1] = extension[n-1:]
        if is_transitive(extended_matrix):
            extended_matrices.append(extended_matrix)
    return extended_matrices

def sample_surmise_relations(base_set, num_samples=10, antisymmetric=False, seed=None):
    """
    Samples surmise relations for n items given a corresponding base set for n-1 items using the Inductive Uniform Extension Approach.
    The implementation follows Ünlü, A., & Schrepp, M. (2017). Techniques for sampling quasi-orders. Arch. Data Sci. A, 2, 163-182. 
    If the base set is exhaustive for n-1 items, this function generates all possible surmise relations for n items.
    
    :param base_set: List of quasi-orders for n-1 items to be extended.
    :type base_set: list[numpy.ndarray]
    :param num_samples: Maximum number of quasi-orders to sample.
    :type num_samples: int
    :param antisymmetric: Whether to sample only antisymmetric quasi-orders.
    :type antisymmetric: bool
    :param seed: Seed for random number generation.
    :type seed: int
    
    :return: List of sampled quasi-orders in binary matrix notation.
    :rtype: list[numpy.ndarray]
    """
    if seed is not None:
        np.random.seed(seed)

    sampled_relations = []
    l = len(base_set)
    for base_matrix in tqdm(base_set, desc=f"Sampling from base set (len {l})"):
        # Extend each base quasi-order
        extended_matrices = extend_quasi_order(base_matrix)
        if antisymmetric:
            # Filter for antisymmetry if required
            extended_matrices = [m for m in extended_matrices if is_antisymmetric(m)]

        sampled_relations.extend(extended_matrices)
    
    # Sample the specified number of extended quasi-orders without replacement
    if len(sampled_relations) > num_samples:
        indices = np.random.choice(len(sampled_relations), size=num_samples, replace=False)
        sampled_relations = [sampled_relations[i] for i in indices]
    
    return sampled_relations

# test 
def sampling_quasiorders(seed=None, n_samples=num_prosets[5], max_items=4, min_items=2, antisymmetric=False, plot=False): 
    """
    Function to test and visualize the representative sampling of surmise relations for a given interval of domain sizes.
    
    Note: When training a DL model on these structures, it is recommended to (re)sample independantly for each domain size, 
    as a base set and its extension may be correlated and limit the diversity of the dataset or even representativeness across domain sizes?
    
    :param seed: Seed for random number generation (optional).
    :type seed: int
    :param n_samples: Maximum number of samples to generate for each domain size.
    :type n_samples: int
    :param max_items: Maximum domain size to sample relations for.
    :type max_items: int
    :param min_items: Minimum domain size to sample relations for.
    :type min_items: int
    :param antisymmetric: Whether to sample only antisymmetric relations.
    :type antisymmetric: bool
    :param plot: Whether to plot histograms of the sampled relations to visualize the distribution of cardinalities.
    :type plot: bool
    
    :return: Nested list of sampled surmise relations in binary matrix notation, one list for each domain size.
    :rtype: list[list[numpy.ndarray]]
    """
    base_set = [
        np.array([[1, 0], [0, 1]], dtype=int),  # Identity matrix (each item is independent)
        np.array([[1, 1], [0, 1]], dtype=int),  # Item 1 (a) implies Item 2 (b)
        np.array([[1, 0], [1, 1]], dtype=int),  # Item 2 (b) implies Item 1 (a)
    ]
    if not antisymmetric:
        base_set.append(np.array([[1, 1], [1, 1]], dtype=int))  # If antisymmetric, both items are equivalent 
    
    dataset = []
    if min_items <= 2:
        dataset.append(base_set)
    
    # for each required domain size, sample surmise relations
    for i in range(3, max_items + 1):
        start_time = time.time()
        base_set = sample_surmise_relations(base_set=base_set, num_samples=n_samples, antisymmetric=antisymmetric, seed=seed)
        elapsed_time = time.time() - start_time
        if i >= min_items:
            dataset.append(base_set)
        print(f"Sampling on {i} items elapsed time: {elapsed_time:.2f} seconds, {len(base_set)} structures sampled.")
        
    if plot:
        print()
        # plot histogram for each num of items
        for i, base_set in enumerate(dataset):
            if len(base_set) > 0:
                n = i + min_items
                # plot histogram
                plt.hist([np.sum(m) for m in base_set], bins=50)
                if antisymmetric:
                    plt.title(f'#items: {n}, #structures: {len(base_set)}/{num_posets[n]}') 
                else:
                    plt.title(f'#items: {n}, #structures: {len(base_set)}/{num_prosets[n]}') 
                plt.xlabel("Cardinality of Surmise Relations")
                plt.ylabel("Counts")
                plt.show()
        
        # plot histogram of complete dataset
        all_base_set = [m for base_set in dataset for m in base_set]
        plt.hist([np.sum(m) for m in all_base_set], bins=50)
        if antisymmetric:
            plt.title(f'All antisymmetric structures, #structures: {len(all_base_set)}/{sum(list(num_posets.values())[min_items:len(dataset)+min_items])}')
        else: 
            plt.title(f'All structures, #structures: {len(all_base_set)}/{sum(list(num_prosets.values())[min_items:len(dataset)+min_items])}')
        plt.xlabel("Cardinality of Surmise Relations")
        plt.ylabel("Counts")
        plt.show()

    return dataset

