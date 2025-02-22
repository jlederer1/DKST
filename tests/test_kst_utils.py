import unittest

# from dkst.utils.set_operations import is_union_closed, is_intersection_closed, union_closure, intersection_closure, union_base, sort_K
# from dkst.utils.relations import is_reflexive, is_antisymmetric, is_transitive
from dkst.utils.KST_utils import implications_to_states, states_to_implications, generate_all_surmise_relations, num_prosets, num_posets, generate_all_knowledge_structures, is_union_closed
from dkst.utils.KST_utils import sampling_quasiorders, sample_knowledge_structures, blim, sample_response_data

import numpy as np

class TestKSTUtils(unittest.TestCase):

    def test_conversion(self):
        states = np.array([
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 1],
            [1, 0, 1, 0, 1, 1],
            [1, 1, 1, 0, 1, 1],
            [1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1]
        ])
        surmise_matrix = np.array([
            [1, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1]
        ])
        
        self.assertTrue(np.all(surmise_matrix == states_to_implications(implications_to_states(surmise_matrix))))
        self.assertTrue(np.all(states == implications_to_states(states_to_implications(states))))
    
    def test_generate_all_surmise_relations(self):
        n = [2,3,4] 
        for i in n:
            quasiorders = generate_all_surmise_relations(i)
            # Convert each binary matrix to a tuple of tuples (hashable for set operations)
            quasiorders_as_tuples = set(map(lambda x: tuple(map(tuple, x)), quasiorders))

            # checks if all generated quasiorders are unique
            self.assertEqual(len(quasiorders_as_tuples), len(quasiorders), f"Not all matrices are unique for n = {i}")
            # checks if the number of generated quasiorders is correct
            self.assertEqual(num_prosets[i], len(quasiorders), f"Incorrect number of quasiorders generated for n = {i}")
    
    def test_sampling_quasiorders(self):
        m_max = 5
        m_min = 3
        partialorders = sampling_quasiorders(n_samples=num_posets[m_max], max_items=m_max, min_items=m_min, antisymmetric=True)
        # Flatten nested list of samples
        partialorders = [poset for sublist in partialorders for poset in sublist]
        # Convert each binary matrix to a tuple of tuples (hashable for set operations)
        partialorders_as_tuples = set(map(lambda x: tuple(map(tuple, x)), partialorders))

        # checks if all generated partial orders are unique
        self.assertEqual(len(partialorders_as_tuples), len(partialorders), f"Not all matrices are unique, got {len(partialorders_as_tuples)} unique matrices out of {len(partialorders)}")   
        # checks if the number of generated partial orders is correct
        n_posets = sum([num_posets[i] for i in range(m_min, m_max+1)])
        self.assertEqual(n_posets, len(partialorders), "Incorrect number of partial orders generated, expected {n_posets}, got {len(partialorders)}")  

    def test_sampling_response_patterns(self):
        m = 7
        structures = sample_knowledge_structures(m=m, num_samples=100, no_dublicates=True)
        response_patterns = []
        # simulate response data from each structure
        for last_dim in range(structures.shape[2]):
            Rs = blim(structures[:,:,last_dim], num_samples=15)
            response_patterns.append(Rs)
        
        self.assertEqual(structures.shape[2], 100, "Incorrect number of knowledge structures generated")
        self.assertEqual(structures.shape[1], m, "Incorrect number of columns (items)")
        self.assertEqual(len(response_patterns[0]), 15, "Incorrect number of response patterns generated")
    
    def test_sampling_response_data(self):
        m = 10
        n_patterns = 1000
        n_structures = 10
        structures = sample_knowledge_structures(m=m, num_samples=n_structures, no_dublicates=False)
        # simulate response data from each structure
        data = sample_response_data(structures, num_samples=n_patterns)
        self.assertEqual(data[0].shape, (n_patterns,m), "Incorrect shape of response data")
        self.assertEqual(len(data), n_structures, "Incorrect shape of response data")

    def test_generate_all_knowledge_structures(self):
        # For n=2:
        # Total families should be 2^(2^2-2) = 4 when union_closed is False or True.
        structures_all = generate_all_knowledge_structures(2, union_closed=False)
        self.assertEqual(len(structures_all), 4, "Incorrect total number of knowledge structures for n=2")
        
        structures_uc = generate_all_knowledge_structures(2, union_closed=True)
        self.assertEqual(len(structures_uc), 4, "Incorrect number of union-closed knowledge structures for n=2")
        for struct in structures_uc:
            self.assertTrue(is_union_closed(struct), f"Structure {struct} is not union-closed")
        
        # For n=3:
        # Total families should be 2^(2^3-2) = 64 when union_closed is False,
        # and the Dedekind number (?) for n=3 is 45.
        structures_all_n3 = generate_all_knowledge_structures(3, union_closed=False)
        self.assertEqual(len(structures_all_n3), 64, "Incorrect total number of knowledge structures for n=3")
        
        structures_uc_n3 = generate_all_knowledge_structures(3, union_closed=True)
        self.assertEqual(len(structures_uc_n3), 45, "Incorrect number of union-closed knowledge structures for n=3")
        for struct in structures_uc_n3:
            self.assertTrue(is_union_closed(struct), f"Structure {struct} is not union-closed")
        
        # For n=4:
        # Total families should be 2^(2^4-2) = 16384 when union_closed is False,
        # and for knowledge spaces 2271.
        structures_all_n4 = generate_all_knowledge_structures(4, union_closed=False)
        self.assertEqual(len(structures_all_n4), 16384, "Incorrect total number of knowledge structures for n=4")
        
        structures_uc_n4 = generate_all_knowledge_structures(4, union_closed=True)
        self.assertEqual(len(structures_uc_n4), 2271, "Incorrect number of union-closed knowledge structures for n=4")
        for struct in structures_uc_n4:
            self.assertTrue(is_union_closed(struct), f"Structure {struct} is not union-closed")


if __name__ == "__main__":
    unittest.main()


