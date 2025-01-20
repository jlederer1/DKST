import unittest

from dkst.utils.relations import is_reflexive, is_antisymmetric, is_transitive, transitive_closure, transitive_reduction, S_to_matrix, matrix_to_S

import numpy as np


class TestRelations(unittest.TestCase):
    
    def test_is_reflexive(self):
        self.assertTrue(is_reflexive([[1, 0], [0, 1]]))
        self.assertFalse(is_reflexive([[0, 1], [1, 0]]))
    
    def test_is_transitive(self):
        self.assertTrue(is_transitive([[1, 1], [0, 1]]))
        self.assertFalse(is_transitive([[1, 1], [1, 0]]))
    
    def test_is_antisymmetric(self):
        self.assertTrue(is_antisymmetric([[1, 0], [0, 1]]))
        self.assertTrue(is_antisymmetric([[0, 1], [0, 1]]))
        self.assertFalse(is_antisymmetric([[0, 1, 0], [1, 1, 1], [0, 0, 1]]))
    
    def test_transitive_closure(self):
        relation = [('a', 'b'), ('b', 'c')]
        matrix = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
        expected = [('a', 'b'), ('a', 'c'), ('b', 'c')]
        expected_matrix = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
        
        closure = transitive_closure(relation)
        self.assertEqual(set(closure), set(expected))
        closure = transitive_closure(matrix)
        self.assertTrue(np.all(closure == expected_matrix))
    
    def test_transitive_reduction(self):
        R = [('a', 'b'), ('a', 'c'), ('b', 'c'), ('c', 'd')]
        expected = [('a', 'b'), ('b', 'c'), ('c', 'd')]
        
        self.assertEqual(set(transitive_reduction(R)), set(expected))
        self.assertTrue(np.all(transitive_reduction(S_to_matrix(R)) == S_to_matrix(expected)))
    
    def test_S_to_matrix(self):
        S = [('a', 'b'), ('b', 'c'), ('a', 'c')]
        matrix = S_to_matrix(S)
        self.assertTrue(np.all(matrix == [[1, 1, 1], [0, 1, 1], [0, 0, 1]]))
        
        S = [('a', 'b'), ('b', 'c'), ('a', 'e'), ('b', 'e'), ('c', 'e')]
        matrix = S_to_matrix(S)
        self.assertTrue(np.all(matrix == [[1, 1, 0, 0, 1], [0, 1, 1, 0, 1], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]))
    
    def test_matrix_to_S(self):
        matrix = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
        S = matrix_to_S(matrix, include_reflexive=False)
        self.assertTrue(S == [('a', 'b'), ('a', 'c'), ('b', 'c')])

        matrix = np.array([[1, 1, 0, 0, 1], [0, 1, 1, 0, 1], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
        S = matrix_to_S(matrix, include_reflexive=False)
        self.assertTrue(S == [('a', 'b'), ('a', 'e'), ('b', 'c'), ('b', 'e'), ('c', 'e')])


if __name__ == "__main__":
    unittest.main()


