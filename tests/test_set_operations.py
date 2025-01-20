import unittest

from dkst.utils.set_operations import is_union_closed, is_intersection_closed, union_closure, intersection_closure, union_base, sort_K, sort_K_binary, K_to_matrix, matrix_to_K

import numpy as np

class TestSetOperations(unittest.TestCase):

    def test_is_union_closed(self):
        self.assertTrue(is_union_closed(["a", "b", "ab", "bc", "abc"]))
        self.assertFalse(is_union_closed(["a", "b", "ab", "ac"]))
        self.assertTrue(is_union_closed([]))
        self.assertFalse(is_union_closed(["", "a", "b"]))

    def test_is_union_closed_matrix(self):
        # Case 1: Union-closed binary matrix
        matrix_sets = np.array([
            [1, 0, 0],  # State 1: knows item 'a'
            [1, 1, 0],  # State 2: knows items 'a' and 'b'
            [0, 1, 1],  # State 3: knows items 'b' and 'c'
            [1, 1, 1]   # State 4: knows items 'a', 'b', and 'c'
        ])
        self.assertTrue(is_union_closed(matrix_sets))

        # Case 2: Union-closed binary matrix
        matrix_sets = np.array([
            [1, 0, 0],  # State 1: knows item 'a'
            [0, 1, 0],  # State 2: knows item 'b'
            [1, 1, 0]   # State 3: knows items 'a' and 'b'
        ])
        self.assertTrue(is_union_closed(matrix_sets))

        # Case 3: Empty matrix (trivially union-closed)
        matrix_sets = np.empty((0, 0), dtype=int)
        self.assertTrue(is_union_closed(matrix_sets))

        # Case 4: Single-row matrix (trivially union-closed)
        matrix_sets = np.array([[1, 0, 0]])
        self.assertTrue(is_union_closed(matrix_sets))

    def test_is_intersection_closed(self):
        self.assertTrue(is_intersection_closed(["a", "ab", "b", ""]))
        self.assertFalse(is_intersection_closed(["a", "b", "ab", "c"]))
        self.assertTrue(is_intersection_closed([]))
        self.assertTrue(is_intersection_closed(["", "a", "b"]))


    def test_union_closure(self):
        self.assertEqual(union_closure(["a", "b"]), ["", "a", "b", "ab"]) 
        self.assertEqual(union_closure(["ab", "bc"]), ["", "ab", "bc", "abc"]) 
        self.assertEqual(union_closure([]), [""])  
        self.assertEqual(union_closure(["", "a"]), ["", "a"]) 
        self.assertEqual(union_closure(['e', 'f', 'g', 'be', 'cf', 'dfg', 'abeg', 'adfg', 'abcef']), ['', 'e', 'f', 'g', 'be', 'cf', 'ef', 'eg', 'fg', 'bef', 'beg', 'cef', 'cfg', 'dfg', 'efg', 'abeg', 'adfg', 'bcef', 'befg', 'cdfg', 'cefg', 'defg', 'abcef', 'abefg', 'acdfg', 'adefg', 'bcefg', 'bdefg', 'cdefg', 'abcefg', 'abdefg', 'acdefg', 'bcdefg', 'abcdefg'])  

    def test_intersection_closure(self):
        self.assertEqual(intersection_closure(["a", "ab"]), ["", "a", "ab"]) 
        self.assertEqual(intersection_closure(["ab", "bc"]), ["", "b", "ab", "bc"])  
        self.assertEqual(intersection_closure([]), [])  
        self.assertEqual(intersection_closure(["", "a"]), ["", "a"])  
        self.assertEqual(intersection_closure(["", "a", "bd", "cd"]), ["", "a", "d", "bd", "cd"]) 

    def test_union_base(self):
        # A valid union-closed set, so it has a base
        self.assertEqual(union_base(["ab", "bc", "abc"]), ["ab", "bc"])
        
        # Another valid union-closed set
        self.assertEqual(union_base(["a", "b", "ab"]), ["a", "b"])

        # This structure is not closed under union, so it shouldn't be possible to find a valid base.
        with self.assertRaises(ValueError):
            union_base(["a", "ab", "ac"])  # No "abc" means it's not union-closed
        
        # This is a union-closed set, so it has a base
        self.assertEqual(union_base(["a", "ab", "ac", "abc"]), ["a", "ab", "ac"])
        
        self.assertEqual(union_base(['', 'e', 'f', 'g', 'be', 'cf', 'ef', 'eg', 'fg', 'bef', 'beg', 'cef', 'cfg', 'dfg', 'efg', 'abeg', 'adfg', 'bcef', 'befg', 'cdfg', 'cefg', 'defg', 'abcef', 'abefg', 'acdfg', 'adefg', 'bcefg', 'bdefg', 'cdefg', 'abcefg', 'abdefg', 'acdefg', 'bcdefg', 'abcdefg']), ['e', 'f', 'g', 'be', 'cf', 'dfg', 'abeg', 'adfg', 'abcef'])   # see (Dowling, 1993)
        
        self.assertEqual(union_base(["", "a", "ab", "ac", "abc"]), ["a", "ab", "ac"])  # Base does not include the empty set, see Falmagne and Doignon (2011)


    def test_sort_K(self):
        self.assertEqual(sort_K({"b", "a", "ab", "c"}), ["a", "b", "c", "ab"])
        self.assertEqual(sort_K({"abcde", "a", "abc", "abd", "d"}), ["a", "d", "abc", "abd", "abcde"])
        self.assertTrue(np.all(K_to_matrix(sort_K({"b", "a", "ab", "c"})) == sort_K_binary(K_to_matrix(["b", "a", "ab", "c"]))))
        self.assertTrue(np.all(K_to_matrix(sort_K({"abcde", "a", "abc", "abd", "d"})) == sort_K_binary(K_to_matrix(["abcde", "a", "abc", "abd", "d"]))))

    def test_sort_K_binary(self):
        # Case 1: Sort a binary matrix
        matrix = np.array([
            [0, 1, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 1]
        ])
        expected = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1]
        ])
        self.assertTrue(np.all(sort_K_binary(matrix) == expected))
        self.assertEqual(sort_K(matrix_to_K(matrix)), matrix_to_K(expected))

        matrix = np.array([
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0]
        ])
        expected = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1]
        ])
        self.assertTrue(np.all(sort_K_binary(matrix) == expected))
        self.assertEqual(sort_K(matrix_to_K(matrix)), matrix_to_K(expected))

    def test_K_to_matrix(self):
        K = ['e', 'f', 'g', 'be', 'cf', 'dfg', 'abeg', 'adfg', 'abcef'] # the base of a knowledge space
        matrix = K_to_matrix(K)
        expected = np.array([
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 1, 1],
            [1, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 1, 1],
            [1, 1, 1, 0, 1, 1, 0]
        ])
        self.assertTrue(np.all(matrix == expected))
    
    def test_matrix_to_K(self):
        # the base of a knowledge space in binary matrix notation
        matrix = np.array([
            [1, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 1, 1],
            [1, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 1, 1]
        ])
        K = matrix_to_K(matrix)
        expected = ['abcef', 'e', 'f', 'g', 'be', 'cf', 'dfg', 'abeg', 'adfg'] 
        self.assertEqual(K, expected)


if __name__ == "__main__":
    unittest.main()
