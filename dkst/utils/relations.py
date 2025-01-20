"""
relations.py

This module provides functions to determine if a given relation is reflexive, transitive or antisymmetric. 
It also provides functions to compute the transitive closure/reduction of a binary relation and sort it in matrix notation.
"""

import numpy as np

def is_reflexive(matrix):
    """
    Check if the given square binary matrix is reflexive.

    A matrix is reflexive if all diagonal elements are 1.

    :param matrix: A square binary matrix (list of lists).
    :type matrix: list[list[int]]
    :return: True if the matrix is reflexive, False otherwise.
    :rtype: bool
    """
    size = len(matrix)
    for i in range(size):
        if matrix[i][i] != 1:
            return False
    return True

def is_transitive(matrix):
    """
    Check if the given square binary matrix is transitive.

    A matrix is transitive if for all i, j, k, if matrix[i][j] == 1 and matrix[j][k] == 1, 
    then matrix[i][k] must also be 1.

    :param matrix: A square binary matrix (list of lists).
    :type matrix: list[list[int]]
    :return: True if the matrix is transitive, False otherwise.
    :rtype: bool
    """
    size = len(matrix)
    for i in range(size):
        for j in range(size):
            if matrix[i][j] == 1:
                for k in range(size):
                    if matrix[j][k] == 1 and matrix[i][k] != 1:
                        return False
    return True

def is_antisymmetric(matrix):
    """
    Check if the given square binary matrix is antisymmetric.

    A matrix is antisymmetric if for all i, j: if matrix[i][j] = 1 and i ≠ j,
    then matrix[j][i] must be 0.

    :param matrix: A square binary matrix (list of lists).
    :type matrix: list[list[int]]
    :return: True if the matrix is antisymmetric, False otherwise.
    :rtype: bool
    """
    size = len(matrix)
    for i in range(size):
        for j in range(size):
            if i != j and matrix[i][j] == 1 and matrix[j][i] == 1:
                return False
    return True

def transitive_closure(matrix): 
    """
    Compute the transitive closure of a given binary relation in matrix notation or set notation, using the Warshall's algorithm.
    
    :param matrix: A square binary matrix or a list of tuples representing a binary relation.
    :type matrix: np.ndarray or list[tuple[str, str]]
    :return: The transitive closure of the given binary relation.
    :rtype: np.ndarray or list[tuple[str, str]], respectively, matching the input type.
    """
    return_set_notation = False
    if isinstance(matrix[0], tuple):
        matrix = S_to_matrix(matrix)
        return_set_notation = True

    # number of items
    m = matrix.shape[0]
    n = matrix.shape[0]
    
    # Warshall's algorithm to compute the transitive closure
    closure = matrix.copy()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                closure[i, j] = closure[i, j] or (closure[i, k] and closure[k, j])
    
    # If the input was a tuple list, return the result as a list of tuples
    if return_set_notation:
        return matrix_to_S(closure)
    
    return closure

def transitive_reduction(relation): 
    """
    Compute the transitive reduction of a given binary relation.
    If the input is in set notation, transitive_reduction_tuples is executed.
    If the input is in matrix notation, transitive_reduction_matrix is executed.
    
    :param matrix: A square binary matrix or a list of tuples representing a binary relation.
    :type matrix: np.ndarray | list[tuple[str, str]]
    
    :return: The transitive reduction of given binary relation.
    :rtype: np.ndarray | list[tuple[str, str]], depending on the input
    """
    if isinstance(relation[0], tuple):
        # reduction in set notation is slightly faster
        return transitive_reduction_tuples(relation) 
    
    elif isinstance(relation, np.ndarray):
        # Check input matrix
        if relation.shape[0] == relation.shape[1]:
            return transitive_reduction_matrix(relation)
        else:
            raise ValueError("Input matrix must be a square numpy array or a list of tuples.")

def transitive_reduction_matrix(matrix):
    """
    Compute the transitive reduction of a given binary relation in matrix notation.
    
    :param matrix: A square binary matrix representing a binary relation.
    :type matrix: np.ndarray 
    
    :return: The transitive reduction of given binary relation.
    :rtype: np.ndarray 
    """
    
    # Number of items
    m = matrix.shape[0]
    
    # Initialize transitive reduction as copy of input matrix
    reduction = matrix.copy()
    
    # Iterate through all pairs (i, j)
    for i in range(m):
        for j in range(m):
            # If there is a direct relation (i, j), check for transitivity
            if matrix[i, j] == 1:
                for k in range(m):
                    # If there is an indirect path i -> k -> j, mark (i, j) as redundant
                    if matrix[i, k] == 1 and matrix[k, j] == 1 and k != i and k != j:
                        reduction[i, j] = 0  # Remove redundant edge
    
    return reduction

def transitive_reduction_tuples(relations):
    """
    Compute the transitive reduction of a binary relation represented as a list of tuples.
    
    Note: Does not preserve the order of the input tuples.

    :param relations: A list of tuples representing the binary relation.
    :type relations: list[tuple[str, str]]

    :return: The transitive reduction of the given binary relation as a list of tuples.
    :rtype: list[tuple[str, str]]
    """
    # Convert list to set for efficient lookup
    relations_set = set(relations)
    reduction = relations_set.copy()

    # For each pair (a, b), check if there is an alternative path from a to b
    # by chaining existing relations without using (a, b) directly.
    for a, b in relations_set:
        # Temporarily remove (a, b) to avoid trivial paths
        relations_set.remove((a, b))

        # Use a set to keep track of nodes reachable from a
        reachable = set()
        # Initialize the frontier with nodes directly reachable from a
        frontier = {a}

        while frontier:
            next_frontier = set()
            for node in frontier:
                # Find all nodes that can be reached from 'node'
                successors = {v for (u, v) in relations_set if u == node}
                # If b is among the successors, we can reach b without (a, b)
                if b in successors:
                    # Edge (a, b) is redundant
                    reduction.discard((a, b))
                    # No need to search further for this pair
                    next_frontier = set()
                    break
                next_frontier.update(successors)
            # Update the reachable set and the frontier
            reachable.update(frontier)
            frontier = next_frontier - reachable

        # Add (a, b) back to the relations set for the next iteration
        relations_set.add((a, b))

    return list(reduction)


def S_to_matrix(S): 
    """
    Convert a quasiorder S to a square binary matrix.

    A quasiorder is a list of tuples (i, j) where i precedes j in the quasiorder.

    :param S: A quasiorder.
    :type S: list[tuple[char, char]]
    :return: The square binary matrix representation of the quasiorder.
    :rtype: np.ndarray
    """
    S = [(ord(i) - ord("a"), ord(j) - ord("a")) for i, j in S]
    
    m = max(max(pair) for pair in S) + 1
    matrix = np.zeros((m, m), dtype=int)

    for i, j in S:
        matrix[i][j] = 1
    
    # ensure reflexivity
    for i in range(m):
        matrix[i][i] = 1

    return matrix

def matrix_to_S(matrix, include_reflexive=False): 
    """
    Convert a square binary matrix to a quasiorder in set representation.

    A quasiorder is a list of tuples (i, j) where i precedes j in the quasiorder.

    :param matrix: A square binary matrix.
    :type matrix: np.ndarray
    :return: The quasiorder representation of the matrix.
    :rtype: list[tuple[char, char]]
    """
    S = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] == 1 and (include_reflexive or i != j):
                S.append((chr(i + ord("a")), chr(j + ord("a"))))
    return S

