"""
set_operations.py

This module provides functions to determine if a given set of string subsets is union-closed and intersection-closed,
and to compute the union closure and intersection closure of a set of string subsets.
"""

#######################
# To do: 
# - change sorting scheme in find_union_closed_base, intersection_closure, union_closure
# - check consistency of n and m in find_union_closed_base and dkst package
# - Make sure sorting is consistent for structures and relations and both in set and matrix representation; sort_alphabetic()
# - use np concistently, powerset()
#######################

from itertools import combinations

import numpy as np


def powerset(m):
    """
    Generate the powerset of a given domain size m in alphabetical set notation.
    
    :param m: The domain size.
    :type m: int
    :return: The powerset of the domain as ordered list of strings.
    :rtype: list[str]
    """
    domain = [chr(97 + i) for i in range(m)]
    powerset = set()
    
    for i in range(1 << m):  # Equivalent to 2^n
        subset = ""
        for j in range(m):
            # Check if the j-th element is in the i-th subset
            if i & (1 << j):
                subset += domain[j]
        powerset.add(subset)
    powerset = sort_K(powerset)
    
    return powerset 

def powerset_binary(m):
    """
    Generate the list of all possible combinations of 0s and 1s of length m.
    
    :param m: Length of the binary vectors
    :type m: int
    :return: List of binary vectors
    :rtype: np.ndarray
    """
    powerset = np.array([np.array([int(x) for x in format(i, '0' + str(m) + 'b')]) for i in range(2 ** m)])
    powerset = sort_K_binary(powerset)
    return powerset


def is_union_closed(sets):
    """
    Check if the given set is union-closed. The set can be either a list of string subsets
    or a binary matrix (numpy array) where rows represent knowledge states and columns represent items.

    A set is union-closed if the union of any two subsets (or knowledge states) results in a subset
    (or knowledge state) that is also in the set.

    :param sets: A list of subsets represented as strings, or a binary matrix (numpy array).
    :type sets: list[str] or numpy.ndarray
    :return: True if the set is union-closed, False otherwise.
    :rtype: bool
    """
    # Check if the input is a list of strings
    if isinstance(sets, list) and all(isinstance(s, str) for s in sets):
        for set1 in sets:
            for set2 in sets:
                union_set = ''.join(sorted(set(set1) | set(set2)))
                if union_set not in sets:
                    return False
        return True

    # Check if the input is a numpy array (binary matrix)
    elif isinstance(sets, np.ndarray):
        num_rows, num_columns = sets.shape

        for i in range(num_rows):
            for j in range(i, num_rows):
                union_row = np.bitwise_or(sets[i], sets[j])
                if not any(np.array_equal(union_row, sets[k]) for k in range(num_rows)):
                    return False
        return True

    else:
        raise ValueError("Unsupported input type. Input must be a list of strings or a numpy array.")

def is_intersection_closed(sets):
    """
    Check if the given set of string subsets is intersection-closed.

    A set of string subsets is intersection-closed if the intersection of characters from any two subsets within the set,
    when sorted, is also a subset within the set.

    :param sets: A list of subsets represented as strings.
    :type sets: list[str]
    :return: True if the set of subsets is intersection-closed, False otherwise.
    :rtype: bool
    """
    for set1 in sets:
        for set2 in sets:
            intersection_set = ''.join(sorted(set(set1) & set(set2)))
            if intersection_set not in sets:
                return False
    return True


def union_closure(sets):
    """
    Compute the union closure of a given set of string subsets.

    The union closure of a set is the smallest union-closed set containing all original subsets.

    :param sets: A list of subsets represented as strings.
    :type sets: list[str]
    :return: The union-closed set containing all possible unions of subsets.
    :rtype: list[str]
    """
    closure = set(sets)
    added = True

    while added:
        added = False
        new_elements = set()

        for set1 in closure:
            for set2 in closure:
                union_set = ''.join(sorted(set(set1) | set(set2)))
                if union_set not in closure:
                    new_elements.add(union_set)
                    added = True
        
        closure.update(new_elements)
    
    sorted_K = sort_K(closure)
    if sorted_K == [] or sorted_K[0] != '':
        sorted_K.insert(0, '')   # the closure of any family of sets always includes the empty set 
    
    return sorted_K

def intersection_closure(sets):
    """
    Compute the intersection closure of a given set of string subsets.

    The intersection closure of a set is the smallest intersection-closed set containing all original subsets.

    :param sets: A list of subsets represented as strings.
    :type sets: list[str]
    :return: The intersection-closed set containing all possible intersections of subsets.
    :rtype: list[str]
    """
    if not sets:  # If the input is empty, return an empty list
        return []

    closure = set(sets)
    added = True

    while added:
        added = False
        new_elements = set()

        for set1 in closure:
            for set2 in closure:
                intersection_set = ''.join(sorted(set(set1) & set(set2)))
                if intersection_set not in closure:
                    new_elements.add(intersection_set)
                    added = True
        
        closure.update(new_elements)
    
    sorted_K = sort_K(closure)
    if sorted_K[0] != '':
        sorted_K.insert(0, '')   # the closure of any family of sets always includes the empty set 

    return sorted_K

def union_base(sets, domain=None):
    """
    Computes the base of a knowledge space using the Dowling 1993 algorithm, see also Falmagne & Doignon 2011. 
    It automatically infers the domain from the input sets, if not specified.

    :param sets: A list of subsets representing states in the knowledge space.
    :type sets: list[str]
    :return: A list of subsets representing the base of the knowledge space.
    :rtype: list[str]
    """
    # Check if the input set is union-closed
    if not is_union_closed(sets):
        raise ValueError("The input set is not union-closed, so it doesn't have a base.")
    
    # Infer the domain Q by taking the union of all items in all sets
    if domain is None:
        domain = sorted(set().union(*sets))

    # Initialize the array T
    n = len(sets)                                                                          ### TO DO: check consistency throughout dkst package...
    m = len(domain)                                                                        ### This version of n and m is used in Falkmane & Doignon 2011..
    T = [['*' if domain[j] in sets[i] else '-' for j in range(m)] for i in range(n)]

    # Process the array to mark with '+' where an item aready occured in a previous state 
    for i in range(1, n):
        for j in range(m):
            if T[i][j] == '*':
                for p in range(i):
                    if T[p][j] == '*' and set(sets[p]).issubset(set(sets[i])):
                        T[i][j] = '+'
                        break

    # Extract the base (atoms)
    base = [sets[i] for i in range(n) if '*' in T[i]]
    
    if base[0] == '': # bases dont include the empty set
        base.pop(0)

    return base   # to do: always sort the base? 


def sort_K(sets):
    """
    Sorts a given set or list of strings by length and alphabetically and returns the sorted knowledge structure as list.
    Induces the same order as sort_K_binary() for the corresponding knowledge structure in binary matrix notation.
    
    :param sets: A list of strings representing knowledge states.
    :type sets: list[str]
    :return: A sorted list of strings representing knowledge states.
    :rtype: list[str]
    """
    # check for knowledge structure vs quasiorder
    # to do #
    
    # sorts list of strings a) by length b) alphabetically like ""<"a"<"b"<"c"<"ab"<"ac"<"bc"<"abc"<"<pad>"
    sorted_states = sorted(sets, key=lambda x: (len(x), x))
    return sorted_states

def sort_K_binary(matrix):
    """
    Sorts a given binary matrix by the sum of the elements in each row and the integer value represented by a row in binary code.
    Induces the same order as sort_K() for the corresponding knowledge structure in set notation.
    
    :param matrix: A binary matrix (numpy array).
    :type matrix: numpy.ndarray
    :return: A row-wise sorted binary matrix.
    :rtype: numpy.ndarray
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"Expected a numpy array, but got {type(matrix)} instead!")
    
    # Function to convert binary list to integer
    def binary_to_int(binary_list):
        return int("".join(str(x) for x in binary_list), 2)
    # Sort the array based on two criteria:
    # 1. The sum of the elements in the sublist.
    # 2. The integer value represented by the binary number in the sublist.
    return np.array(sorted(matrix, key=lambda sublist: (sum(sublist), -binary_to_int(sublist))))

def sort_K_binary(matrix):
    """
    Sorts a given binary matrix by the sum of the elements in each row and the integer value represented by a row in binary code.
    Induces the same order as sort_K() for the corresponding knowledge structure in set notation.
    
    :param matrix: A binary matrix (numpy array).
    :type matrix: numpy.ndarray
    :return: A row-wise sorted binary matrix.
    :rtype: numpy.ndarray
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"Expected a numpy array, but got {type(matrix)} instead!")

    # Calculate row sums and binary values for sorting
    row_sums = np.sum(matrix, axis=1)
    binary_values = np.dot(matrix, 2 ** np.arange(matrix.shape[1])[::-1])

    # Use np.lexsort to sort by row sums (ascending) and binary values (descending)
    sorted_indices = np.lexsort((-binary_values, row_sums))
    return matrix[sorted_indices]


def K_to_matrix(sets):
    """
    Convert a knowledge structure K from alphabetic set notation into a binary matrix.
    Can be used for only the base of a knowledge structure or for the entire knowledge structure.

    :param sets: A knowledge structure (sorted).
    :type sets: list[str]
    :return: The binary matrix representation of the knowledge structure (sorted).
    :rtype: np.ndarray
    """
    m = len(set(''.join(sets)))
    
    # Create a binary matrix
    matrix = np.zeros((len(sets), m), dtype=int)
    # Fill the matrix with the binary representation of the knowledge states
    a = ord("a") # ASCII value of 'a' is 97
    for i, state in enumerate(sets):
        for item in state:
            matrix[i][ord(item) - a] = 1
    return matrix

def matrix_to_K(matrix):
    """
    Convert a binary matrix into a knowledge structure K in alphabetic set notation.
    Can be used for only the base of a knowledge structure or for the entire knowledge structure.
    Respects the order of given knowledge states.

    :param matrix: A binary matrix (numpy array), possibly padded with trailing zero rows.
    :type matrix: np.ndarray
    :return: The knowledge structure representation of the matrix.
    :rtype: list[str]
    """
    # Convert the binary matrix to a list of strings
    a = ord('a') # ASCII value of 'a' is 97
    K = [''.join(chr(a + j) for j in range(matrix.shape[1]) if matrix[i][j] == 1) for i in range(matrix.shape[0])]
    return K
