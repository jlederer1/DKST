"""
mcmc_sampling.py

A knowledge space K is a union-closed family of subsets (in string notation) over a finite domain Q 
that contains both the empty set ('') and the full domain (e.g. 'abcd' for Q = {a,b,c,d}).
Exhaustively enumerating such structures is intractable even for small domains.
This module implements an MCMC procedure that operates on the full range of knowledge spaces.
It uses elementary addition and removal moves, and a global extra permutation 
move (with probability 2%) that randomly permutes the items in K, thereby exploiting symmetry.

For the addition and removal moves: q(K → K') = 1/2 · 1/N(K),
with N(K) being the number of valid candidate moves under the chosen move type.
For a permutation move, since the permutation is chosen uniformly among n! possibilities,
and its reverse is the inverse permutation, the proposal probability is symmetric,
so the move is always accepted.

The overall move probabilities are thus:
    - Permutation move: 2% (extra)
    - Addition move: 50%
    - Removal move: 50%

Author: Jakob Lederer
Date: 2025-01-15
"""

# Standard library imports
import os
import random
import math

# Local imports 
from dkst.utils.set_operations import union_base, union_closure, sort_K, powerset, is_union_closed

# Third-party imports
import numpy as np
from tqdm import tqdm


def candidate_moves_K(K, move_type, domain, full_state):
    """
    Determines candidate moves from the current knowledge space K based on the specified move type.
    
    For "Addition": Iterates over each subset s from the complete powerset of Q (excluding the empty set)
    that is not already in K; if adding s to K directly yields a union‐closed family (without computing the closure)
    and the resulting family differs from K, then s is accepted as a candidate move.
    
    For "Removal": Iterates over each subset s in K (excluding the empty set and the full state); if removing s from K
    directly yields a union‐closed family that is different from K, then s is accepted as a candidate move.
    
    :param K: Current knowledge space represented as a sorted list of subset strings.
    :type K: list[str]
    :param move_type: Type of move to consider ()"Addition" or "Removal").
    :type move_type: str
    :param domain: Sorted list of items representing the domain Q.
    :type domain: list[str]
    :param full_state: String representing the full domain (e.g., 'abcd' for Q = {a,b,c,d}).
    :type full_state: str

    :return: List of tuples (s, newK), where s is the candidate subset and newK is the resulting knowledge space (sorted).
    :rtype: list[tuple[str, list[str]]]
    """
    candidates = []
    if move_type == "Addition":
        all_states = powerset(len(domain))
        for s in all_states:
            if s and s not in K:
                newK = K + [s]
                # Test if newK is union-closed
                if is_union_closed(newK) and sort_K(newK) != sort_K(K):
                    candidates.append((s, sort_K(newK)))
    elif move_type == "Removal":
        for s in K:
            if s and s != full_state:
                newK = [x for x in K if x != s]
                # Check if reduced set is union-closed
                if is_union_closed(newK) and sort_K(newK) != sort_K(K):
                    candidates.append((s, sort_K(newK)))
    return candidates

def permute_knowledge_space(K, domain):
    """
    Performs a permutation move on the knowledge space K.
    
    A random permutation is drawn uniformly from all n! possibilities (with n = len(domain)). Each subset in K is 
    re-labeled according to this permutation, and the resulting knowledge space is sorted canonically before being returned.
    
    :param K: The current knowledge space as a list of subset strings.
    :type K: list[str]
    :param domain: Sorted list of items representing the domain Q.
    :type domain: list[str]

    :return: The permuted and sorted knowledge space.
    :rtype: list[str]
    """
    n = len(domain)
    # Generate a random permutation of indices [0, 1, ..., n-1]
    perm_indices = random.sample(range(n), n)
    # Build a mapping: old letter -> new letter
    permuted_domain = [domain[i] for i in perm_indices]
    mapping = {old: new for old, new in zip(domain, permuted_domain)}
    
    def permute_subset(s):
        # Permute each character according to the mapping and sort the result
        return ''.join(sorted(mapping[c] for c in s))
    
    newK = [permute_subset(s) for s in K]
    return sort_K(newK)

def sample_knowledge_space_mcmc(initial_K, T=1000, perm_prob=0.02, log=False):
    """
    Performs MCMC sampling (Metropolis-Hastings) on the space of knowledge spaces (i.e., union-closed families).

    Starting from an initial union-closed knowledge space (including both the empty set and the full domain),
    the algorithm iteratively applies the following moves over T iterations:
    - With probability perm_prob, a permutation move is performed.
    - An "Addition" or "Removal" move is chosen at random.
    Candidate moves are generated using candidate_moves_K, and the acceptance probability is computed as 
    A = min(1, N(K) / N(K')), where N(K) is the number of valid candidate moves from K.
    Optionally, each accepted move can be logged.

    :param initial_K: Initial knowledge space as a sorted list of subset strings.
    :type initial_K: list[str]
    :param T: Total number of MCMC iterations.
    :type T: int
    :param perm_prob: Probability of performing a permutation move (default is 0.02).
    :type perm_prob: float
    :param log: If True, logs each accepted move to a file.
    :type log: bool

    :return: The final knowledge space (sorted list of subset strings) obtained after T iterations.
    :rtype: list[str]
    """
    K = initial_K[:]  # current knowledge space
    domain = sorted(set(''.join(K)))  # domain Q
    full_state = ''.join(domain)  # full state string
    
    for i in range(T):
        r = random.random()
        if r < perm_prob:
            # Permutation move
            K_new = permute_knowledge_space(K, domain)
        
        # Decide between addition and removal moves equally and choose candidate
        move_choice = random.random()
        if move_choice < 0.5:           # Todo: maybe merge addition/removal candidate moves then choose at random? 
            move_type = "Addition"
        else:
            move_type = "Removal"
        candidates_forward = candidate_moves_K(K, move_type, domain, full_state)
        if not candidates_forward:
            continue
        candidate, K_new = random.choice(candidates_forward)

        # Retrieve reverse move candidates and compute acceptance probability
        reverse_type = "Removal" if move_type == "Addition" else "Addition"
        candidates_reverse = candidate_moves_K(K_new, reverse_type, domain, full_state)
        N_forward = len(candidates_forward)
        N_reverse = len(candidates_reverse) if candidates_reverse else 0
        if N_forward == 0 or N_reverse == 0:
            A = 0
        else:
            A = min(1, N_forward / N_reverse)
        if random.random() < A:
            K = K_new

            # logging the marcov chain
            if log:
                log_path = os.path.join("../data/outputs", "mcmc_sampling.log")
                with open(log_path, "a") as f:  # append mode
                    f.write(f"Step {i}: {sort_K(K)}\n")

    return K

def sample_multiple_knowledge_spaces(initial_K, steps_between=10000, total_samples=1000, perm_prob=0.02, log=False):
    """
    Generates multiple knowledge space samples by repeatedly running the MCMC chain.
    
    Starting from the given initial knowledge space, the algorithm runs the MCMC procedure for a specified number
    of iterations (steps_between) between samples and collects a total of total_samples samples.
    
    :param initial_K: The initial knowledge space as a sorted list of subset strings.
    :type initial_K: list[str]
    :param steps_between: Number of MCMC iterations between successive samples.
    :type steps_between: int
    :param total_samples: Total number of knowledge space samples to collect.
    :type total_samples: int
    :param perm_prob: Probability of performing a permutation move (default is 0.02).
    :type perm_prob: float
    :param log: If True, logs each sampled knowledge space to a file.
    :type log: bool

    :return: A list of knowledge space samples, each represented as a sorted list of subset strings.
    :rtype: list[list[str]]
    """
    samples = []
    current_K = initial_K[:]
    for i in tqdm(range(total_samples), desc="Sampling knowledge spaces"):
        current_K = sample_knowledge_space_mcmc(current_K, T=steps_between, perm_prob=perm_prob, log=log)
        samples.append(current_K)
        if (i + 1) % (total_samples / 10) == 0:
            print(f"Sample {i+1}: {sort_K(current_K)}")
    return samples
