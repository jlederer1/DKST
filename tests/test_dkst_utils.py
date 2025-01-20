import unittest

# from dkst.utils.set_operations import is_union_closed, is_intersection_closed, union_closure, intersection_closure, union_base, sort_K
# from dkst.utils.relations import is_reflexive, is_antisymmetric, is_transitive
# from dkst.utils.KST_utils import implications_to_states, states_to_implications
from dkst.utils.relations import S_to_matrix
from dkst.utils.set_operations import K_to_matrix
from dkst.utils.DKST_utils import pad_relations, pad_sequences, pad_K, calculate_counts

import numpy as np

class TestDKSTUtils(unittest.TestCase):
    
    def test_pad_relations(self):
        # S0 = expected[:,:,0], S1 = expected[:,:,1]
        expected = np.array([
            [[1, 1], [1, 1], [0, 1]],  
            [[0, 0], [1, 1], [0, 1]],  
            [[0, 0], [0, 0], [0, 1]]   
        ], dtype=int)
        self.assertTrue(np.all(pad_relations([S_to_matrix([('a', 'b')]), S_to_matrix([('a', 'b'), ('a', 'c'), ('b', 'c')])]) == expected))       
        self.assertTrue(np.all(pad_relations([[('a', 'b')], [('a', 'b'), ('a', 'c'), ('b', 'c')]]) == expected))             
    
    def test_pad_sequences(self):
        self.assertTrue(np.all(pad_sequences([["a", "ab", "abc"]]) == np.array([['', 'a', 'ab', 'abc', '<eos>'] + 4 * ['<pad>']], dtype=str)))
        self.assertTrue(np.all(pad_sequences(np.array([["", "a", "ab", "abc"]], dtype=str)) == np.array([['', 'a', 'ab', 'abc', '<eos>'] + 4 * ['<pad>']], dtype=str)))
    
    def test_pad_K(self):
        self.assertTrue(np.all(pad_K(K_to_matrix(["a", "ab", "abc"])) == np.array([[1,0,0],[1,1,0],[1,1,1], [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]], dtype=int)))
    
    def test_calculate_counts(self):
        self.assertTrue(np.all(calculate_counts(["a", "ab", "abc"], standadize=False) == np.array([0, 1,0,0, 1,0,0, 1], dtype=int)))
        self.assertTrue(np.all(calculate_counts(["a", "ab", "abc", "a", "ab", "abc", "abc", "ab", "a", "a", "ab", "abc"], standadize=False) == np.array([0, 4,0,0, 4,0,0, 4], dtype=int)))
        self.assertTrue(np.all(calculate_counts(["a", "ab", "abc", "a", "ab", "abc", "abc", "ab", "a", "a", "ab", "abc", ""], standadize=False) == np.array([1, 4,0,0, 4,0,0, 4], dtype=int)))
        self.assertTrue(np.all(calculate_counts(K_to_matrix(["ab", "a", "abc", "a", "ab", "abc", "abc", "ab", "a", "a", "ab", "abc", ""]), standadize=False) == np.array([1, 4,0,0, 4,0,0, 4], dtype=int)))

if __name__ == "__main__":
    unittest.main()

