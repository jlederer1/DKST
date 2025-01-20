import unittest
import os

from dkst.utils.relations import *
from dkst.utils.set_operations import *
from dkst.utils.KST_utils import *
from dkst.utils.DKST_utils import *
from dkst.dkst_datasets import *

from torch.utils.data import DataLoader



class TestDKSTDatasets(unittest.TestCase):
    
    def test_Dataset02(self):
        # Dataset configuration 
        config_path = os.path.abspath("data/config/config_data_03.json")
        batch_size = 4
        D2 = DKSTDataset02(config_path)
        dataloader = DataLoader(D2, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        sample = D2.__getitem__(0)
        # print()
        # print("Shape conditionals:       ", sample[0].shape)
        # print("Shape input sequence:     ", sample[1].shape) 
        # print("Shape target sequence:    ", sample[2].shape) 
        # print("Shape input observations: ", sample[3].shape)

        self.assertEqual(len(sample), 4, "Incorrect number of elements in dataset")
        for i in range(4):
            self.assertEqual(len(sample[i]), D2.vocab_size-2, "Incorrect number of elements in sample")

        for i_batch, sample_batched in enumerate(dataloader):
            for T in sample_batched:
                self.assertTrue(T.shape[0] == batch_size, f"Batch size is not correct.")
                self.assertTrue(T.shape[1] == D2.vocab_size-2, f"Vocab size is not correct.")

if __name__ == "__main__":
    unittest.main()

