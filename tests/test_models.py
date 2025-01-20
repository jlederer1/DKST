import unittest
import os

from dkst.utils.relations import *
from dkst.utils.set_operations import *
from dkst.utils.KST_utils import *
from dkst.utils.DKST_utils import *
from dkst.dkst_datasets import *
from dkst.models import *

from torch.utils.data import DataLoader


class TestModels(unittest.TestCase):
    
    def test_Decoder(self):
        # Dataset configuration 
        config_path = os.path.abspath("data/config/config_data_03.json")
        batch_size = 4
        D2 = DKSTDataset02(config_path)
        dataloader = DataLoader(D2, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        
        # model
        model = CustomDecoderModel(config_path)
        ce_loss = CustomCELoss()
        lnloss = LengthNormLoss()
        device = "mps"
        model = model.to(device)

        for sample_batched in dataloader:
            conditionals, input_seq, target_seq, input_obs = sample_batched
            conditionals = conditionals.to(device)
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            input_obs = input_obs.to(device)

            output, embedding, attention_weights = model(conditionals, input_seq)
            # print()
            # print("Output shape: ", output.shape) 
            # print("Embedding shape: ", embedding.shape)
            self.assertEqual(output.shape[0], 2 ** model.m_items) # sequence length / num possible states
            self.assertEqual(output.shape[1], batch_size)
            self.assertEqual(output.shape[2], model.vocab_size-1) # vocab size - pad token
            self.assertEqual(embedding.shape[0], batch_size)
            self.assertEqual(embedding.shape[1], model.hidden_dim)

            # print("Attention weights:")
            # Iterate through each layer's attention weights
            self.assertEqual(len(attention_weights), model.n_layers)
            for i, attn_weight in enumerate(attention_weights):
                # print(f"Layer {i} attention weights shape: {attn_weight.shape}")
                self.assertTrue(attn_weight.shape[0] == batch_size)
                self.assertTrue(attn_weight.shape[1] == model.n_heads)
                self.assertTrue(attn_weight.shape[2] == 2 ** model.m_items) # sequence length 
                self.assertTrue(attn_weight.shape[3] == 2 ** model.m_items) # sequence length 
                break

            # Calculate cross entropy loss
            loss_ce = ce_loss(output, target_seq)
            self.assertTrue(loss_ce.item() >= 0)

            # Calculate LengthNormLoss
            loss_ln = lnloss(embedding, target_seq, model.vocab_size)
            self.assertTrue(loss_ln.item() >= 0)
            break


if __name__ == "__main__":
    unittest.main()

