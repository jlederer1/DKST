"""
DKST_datasets.py

This module provides different types of torch modules for DKST models.
As well as utility functions for training, inference and evaluation.
"""

# imports 
from typing import Optional
import math
import os
import json

import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from pathlib import Path


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.
    """
    def __init__(self, hidden_dim, max_len=5000, dropout_rate=0.0):
        """
        Constructor for the PositionalEncoding class.

        :param hidden_dim: latent dimension of transformer's token embeddings. 
        :type hidden_dim: int
        :param max_len: maximum length of input sequences, defaults to 5000.
        :type max_len: int
        :param dropout_rate: dropout rate, defaults to 0.0.
        :type dropout_rate: float
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # calculate positional encodings once in log space
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).long() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # add positional encodings to input
        x = x + self.pe[:,:x.size(1)]
        x = self.dropout(x)
        return x

class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    Customized torch transformer decoder layer.
    Provides axcess to attention weights.
    Uses MultiheadAttention for both self-attention and cross-attention.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False, bias=True, device="mps", dtype=torch.float32):
        """
        Constructor for the CustomTransformerDecoderLayer.

        :param d_model: Number of latent dimensions to encode the input sequence. 
        :type d_model: int
        :param nhead: Number of attention heads.
        :type nhead: int
        :param dim_feedforward: Dimension of the feedforward network model.
        :type dim_feedforward: int
        :param dropout: Dropout value.
        :type dropout: float
        :param activation: Non-linear activation function of transformer's feedforward layer, such as torch.nn.ReLU
        :type activation: torch.nn activation function
        :param layer_norm_eps: Epsilon value for layer normalization.
        :type layer_norm_eps: float
        :param batch_first: Whether the data is provided in batch first format, at default sequence dimension comes first.
        :type batch_first: bool
        :param norm_first: Whether normalization is applied before attention, at default normalization is applied after attention.
        :type norm_first: bool
        :param bias: Model bias.
        :type bias: bool
        """
        factory_kwargs = {'device': device, 'dtype': dtype} 
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, bias, **factory_kwargs)
        
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.activation_function = activation
        self.batch_first = batch_first
        self.norm_first = norm_first
        
        self.dropout = nn.Dropout(self.dropout_value)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, device=device, dtype=dtype)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, device=device, dtype=dtype)

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True):
        # Adjust for batch_first option if necessary
        if self.batch_first:
            tgt = tgt.transpose(0, 1)
            memory = memory.transpose(0, 1)

        # Self attention
        # Attention weights have the shape (batch_size,num_heads,tgt_seq_len,in_seq_len) since wheights are not averaged
        tgt2, self_attn_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                                 key_padding_mask=tgt_key_padding_mask,
                                                 need_weights=need_weights, average_attn_weights=False)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt) if self.norm_first else self.norm1(tgt)

        # Multi-head attention                 
        tgt2, multihead_attn_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                                           key_padding_mask=memory_key_padding_mask,
                                                           need_weights=need_weights, average_attn_weights=False)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt) if self.norm_first else self.norm2(tgt)

        # Feedforward network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt) if self.norm_first else self.norm3(tgt)

        # Adjust for batch_first option if necessary
        if self.batch_first:
            tgt = tgt.transpose(0, 1)

        return tgt, self_attn_weights, multihead_attn_weights
    
class CustomTransformerDecoder(nn.TransformerDecoder):
    """
    Customized torch transformer decoder.
    Provides axcess to attention weights.
    """
    def __init__(self, decoder_layer, num_layers, norm=None):
        """
        Constructor for the CustomTransformerDecoder.

        :param decoder_layer: Customized transformer decoder layer.
        :type decoder_layer: CustomTransformerDecoderLayer
        :param num_layers: Number of decoder layers.
        :type num_layers: int
        :param norm: Normalization layer, at default layer normalization is applied.
        :type norm: torch.nn.LayerNorm
        """
        super().__init__(decoder_layer, num_layers, norm)
        self.layers = nn.ModuleList([CustomTransformerDecoderLayer(
            d_model=decoder_layer.d_model,
            nhead=decoder_layer.nhead,
            dim_feedforward=decoder_layer.dim_feedforward,
            dropout=decoder_layer.dropout_value,
            activation=decoder_layer.activation_function
        ) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm if norm is not None else nn.LayerNorm(decoder_layer.d_model)

    def forward(self, tgt, memory,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass through the transformer decoder.

        :param tgt: Target sequence of shape (seq_length, batch_size, d_model).
        :type tgt: torch.Tensor
        :param memory: Memory from the encoder of shape (seq_length, batch_size, d_model).
        :type memory: torch.Tensor
        :param tgt_mask: Mask for the target sequence, optional.
        :type tgt_mask: Optional[torch.Tensor]
        :param memory_mask: Mask for the memory sequence, optional.
        :type memory_mask: Optional[torch.Tensor]
        :param tgt_key_padding_mask: Padding mask for the target keys, optional.
        :type tgt_key_padding_mask: Optional[torch.Tensor]
        :param memory_key_padding_mask: Padding mask for the memory keys, optional.
        :type memory_key_padding_mask: Optional[torch.Tensor]
        :return: Output sequence of shape (seq_length, batch_size, d_model) and list of attention weights, each of shape (batch_size, num_heads, seq_length, seq_length).
        :rtype: tuple(torch.Tensor, list[torch.Tensor])
        """
        output = tgt

        attention_wheights = []  # List to store attention weights for network diagnostics.

        for mod in self.layers:
            output, self_attn_weights, _ = mod(output, memory, tgt_mask=tgt_mask,
                                               memory_mask=memory_mask,
                                               tgt_key_padding_mask=tgt_key_padding_mask,
                                               memory_key_padding_mask=memory_key_padding_mask)
            attention_wheights.append(self_attn_weights)

        if self.norm:
            output = self.norm(output)

        return output, attention_wheights

class CustomDecoderModel(nn.Module):
    """
    Customized transformer decoder model for Knowledge Net.
    """
    def __init__(self, config_path):
        """
        Constructor for the CustomDecoderModel class.

        :param config_path: Path to the configuration file.
        :type config_path: str
        """
        super().__init__()
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}.")
        data_dir = os.path.dirname(os.path.dirname(config_path))
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.m_items = self.config['m']
        self.d_type = getattr(torch, self.config['d_type']) # retrieve dtype based on config string 
        self.input_dim = 2 ** self.config['m'] # observations about 2**m possible states
        self.vocab_size = 2 ** self.config['m'] + 2 # number of states + eos and pad tokens
        self.max_seq_len = self.vocab_size - 1 # num_states + eos token

        self.hidden_dim = self.config['hidden_dim']
        self.n_heads = self.config['n_heads'] 
        self.n_layers = self.config['n_layers'] 
        self.dropout_value = self.config['dropout']  
        self.activation = self.config['activation']

        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.dropout_layer = nn.Dropout(p=self.dropout_value)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.max_seq_len, dropout_rate=self.dropout_value)
        self.decoder_layer = CustomTransformerDecoderLayer(d_model=self.hidden_dim, nhead=self.n_heads, activation=self.activation, dropout=self.dropout_value)
        self.transformer_decoder = CustomTransformerDecoder(self.decoder_layer, num_layers=self.n_layers)
        self.output_proj = nn.Linear(self.hidden_dim, self.vocab_size - 1) # exclude padding token from generation

    def forward(self, conditionals, input_seq, data_embedding=None):
        """
        Forward pass through the transformer decoder.
        Output shapes are as follows:
        - Output sequence: (seq_len, batch_size, vocab_size+1)
        - Data embedding: (batch_size, hidden_dim)
        - Attention weights: list of length n_layers, each element is a tensor of self-attention wheights of shape (batch_size, num_heads, seq_length, seq_length).

        :param conditionals: Tensor of conditional probabilities of shape (batch_size, 2**m).
        :type conditionals: torch.Tensor
        :param input_seq: Input sequence as tensor of shape (batch_size, seq_length).
        :type input_seq: torch.Tensor
        :param data_embedding: Data embedding, defaults to None, shape (batch_size, hidden_dim).
        :type data_embedding: torch.Tensor
        :return: Output sequence (seq_len, batch_size, vocab_size+1), data embedding (batch_size, hidden_dim), attention weights (list ).
        :rtype: tuple
        """
        batch_size = conditionals.shape[0]
        seq_length = input_seq.shape[1]

        input_seq = self.embedding(input_seq)
        input_seq = self.pos_encoder(input_seq)
        input_seq = input_seq.permute(1, 0, 2).float()

        # projection into latent space (memory cell)
        if data_embedding is None:
            K_embedding = self.input_proj(conditionals)
            #K_embedding = self.dropout_layer(K_embedding) # Do not use, interferes with ln_loss during training.
        else:
            K_embedding = data_embedding
        memory = K_embedding.view(1, batch_size, self.hidden_dim)
        
        memory = memory.float() # memory.repeat(1, 1, 1).float()

        # masking decoder's self-attention mechanism
        tgt_mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1).to(input_seq.device)
        
        # forward pass through transformer
        output, attention_wheights = self.transformer_decoder(tgt=input_seq, memory=memory, tgt_mask=tgt_mask, memory_mask=None)
        output = self.output_proj(output)

        return output, K_embedding, attention_wheights
    
    # def forward(self, conditionals, input_seq, data_embedding=None):
        """
        Forward pass through the transformer decoder.

        :param conditionals: Tensor of conditional probabilities.
        :type conditionals: torch.Tensor
        :param input_seq: Input sequence.
        :type input_seq: torch.Tensor
        :param data_embedding: Data embedding, defaults to None.
        :type data_embedding: torch.Tensor
        :return: Output sequence, data embedding, attention weights.
        :rtype: tuple
        """
        batch_size = conditionals.shape[0]
        seq_length = input_seq.shape[1]

        input_seq = self.embedding(input_seq)
        input_seq = self.pos_encoder(input_seq)
        input_seq = input_seq.permute(1, 0, 2).float()

        # projection into latent space (memory cell)
        if data_embedding is None:
            K_embedding = self.input_proj(conditionals)
        else:
            K_embedding = data_embedding
        memory = K_embedding.view(1, batch_size, self.hidden_dim)
        
        memory = memory.float() # memory.repeat(1, 1, 1).float()

        # masking decoder's self-attention mechanism
        tgt_mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1).to(input_seq.device)
        
        # forward pass through transformer
        output, attention_wheights = self.transformer_decoder(tgt=input_seq, memory=memory, tgt_mask=tgt_mask, memory_mask=None)
        output = self.output_proj(output)

        return output, K_embedding, attention_wheights


class RegressionNetwork(nn.Module):
    """
    Regressor for response pattern aggregates by frequency.
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_val):
        """
        Constructor for the RegressionNetwork class.
        
        :param input_size: Number of input features.
        :type input_size: int
        :param hidden_size1: Number of neurons in the first hidden layer.
        :type hidden_size1: int
        :param hidden_size2: Number of neurons in the second hidden layer.
        :type hidden_size2: int
        :param output_size: Number of output features.
        :type output_size: int
        :param dropout_val: Dropout value.
        :type dropout_val: float
        """
        super(RegressionNetwork, self).__init__()
        # Define fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_val)
        # self.bn1 = nn.BatchNorm1d(hidden_size1)

    def forward(self, x):
        # Apply the first fully connected layers with ReLU activation
        x = self.activation(self.fc1(x)) 
        x = self.dropout(x)
        x = self.activation(self.fc2(x)) 
        x = self.dropout(x)
        # Linear activation for regression
        x = self.fc3(x)
        return x

class CustomCELoss(nn.Module):
    """
    Custom Cross-Entropy Loss with label smoothing and padding token masking.
    """
    def __init__(self, label_smoothing=0):
        """
        Constructor for the CustomCELoss class.

        :param label_smoothing: Smoothing factor for label smoothing, defaults to 0.
        :type label_smoothing: float
        """
        super(CustomCELoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, output, labels):
        """
        Forward pass for the CustomCELoss class.

        :param output: Predicted output tensor of shape (seq_length, batch_size, vocab_size).
        :type output: torch.Tensor
        :param labels: Ground truth labels tensor of shape (batch_size, seq_length).
        :type labels: torch.Tensor
        :return: Averaged cross-entropy loss over non-padding tokens.
        :rtype: torch.Tensor
        """
        # Identify padding tokens
        pad_token_id = output.shape[-1]
        
        # Permute output to match the expected shape for cross-entropy loss
        output_flat = output.permute(1, 2, 0)
        
        # Create a mask to ignore padding tokens
        mask = (labels != pad_token_id)

        # Compute cross-entropy loss for each predicted token
        loss = F.cross_entropy(output_flat, labels, ignore_index=pad_token_id, reduction='none', label_smoothing=self.label_smoothing)
        
        # Apply mask to the loss
        masked_loss = loss * mask

        # Average the loss over non-padding tokens
        averaged_loss = masked_loss.sum() / mask.long().sum()
        
        return averaged_loss

class LengthNormLoss(nn.Module):
    """
    Length-Norm-Loss: Evaluates the alignment between hidden representation's norms and sequence lengths using MSE.
    Regularizes K-net's training and embedding space.
    """
    def __init__(self):
        """
        Constructor for the LengthNormLoss class.
        """
        super(LengthNormLoss, self).__init__()

    def forward(self, embeddings, targets, vocab_size):
        """
        Forward pass for the LengthLoss class.

        :param embeddings: Hidden representations for each sample per batch.
        :type embedding_norms: torch.Tensor
        :param targets: Ground truth labels tensor of shape (batch_size, seq_length).
        :type targets: torch.Tensor
        :return: Length-Norm-Loss.
        :rtype: torch.Tensor
        """
        embedding_norms = embeddings.norm(dim=1)
        # Calculate sequence lengths
        seq_lengths = ((targets != vocab_size-1) &                      # Exclude padding token ids
                       (targets != vocab_size-2)).sum(dim=1).float()    # Exclude eos token ids
        
        # Calculate LNLoss
        lnloss = F.mse_loss(embedding_norms, seq_lengths) 
        return lnloss

# class K_net_v02(nn.Module):
#     """
#     Knowledge-Net version 2, K-to-K transformer architecture.
#     """
#     def __init__(self, config_path):
#         """
#         Constructor for the K_net_v02 class.
#         """
#         pass

def train(model, train_loader, ce_loss, ln_loss, ln_wheight, optimizer, penalty_weight=0, clip_norm=0, knet=None, device="mps", prediction_only=False):
    """
    Train K-net/decoder model for one epoch.
    Uses projection of conditional probabilities to hidden space in K-net training phase 1 (reconstruction).
    Uses regressor's projection of data observations to hidden space in K-net training phase 2 (prediction).

    :param model: Decoder model to train.
    :type model: nn.Module
    :param train_loader: DataLoader for the training data.
    :type train_loader: torch.utils.data.DataLoader
    :param ce_loss: Custom cross-entropy loss function.
    :type ce_loss: callable
    :param ln_loss: Custom Length-Norm-Loss function for extra regularization.
    :type ln_loss: callable
    :param ln_wheight: Weighting for the Length-Norm-Loss, range [0,1].
    :type ln_wheight: float
    :param optimizer: Optimizer for updating model parameters.
    :type optimizer: torch.optim.Optimizer
    :param clip_norm: Maximum norm for gradient clipping to prevent exploding gradients, defaults to 0, may be chosen in range [0,5] (?).
    :type clip_norm: float
    :param knet: Knowledge-Net model, determines the training phase.
    :type knet: nn.Module, optional
    :param prediction_only: If True, only perform projection into latent space, no decoding, defaults to False.
    :type prediction_only: bool, optional

    :return: Tuple containing training metrics of entire epoch, average cross-entropy loss, average Length-Norm-Loss, and the combined loss.
    :rtype: tuple(float, float, float)
    """
    model.train()
    if knet is not None and prediction_only is False: knet.train()

    total_loss_CE = 0.
    total_loss_LN = 0.
    total_loss_penalty = 0.0
    total_loss_combiened = 0.

    for conditionals, input_seq, target_seq, input_obs in train_loader:
        # Move tensors to the correct device
        conditionals = conditionals.to(device)
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        input_obs = input_obs.to(device)

        batch_size = conditionals.shape[0]

        optimizer.zero_grad()

        # Flexible forward pass 
        if knet:
            ### To be implemented! ###
            #output, embedding, attention_weights = knet(input_obs, input_seq)
            pass
        else: 
            output, embedding, _ = model(conditionals, input_seq)

        # Custom Cross-entropy loss (weighted)
        loss_ce = ce_loss(output, target_seq) * (1 - ln_wheight) / batch_size # Normalize loss by batch size 
        total_loss_CE += loss_ce
        # Custom Length-Norm-Loss (wheighted)
        loss_ln = ln_loss(embedding, target_seq, model.vocab_size) * ln_wheight / batch_size
        total_loss_LN += loss_ln 
        # Penalty loss for dublicates in the output sequence
        with torch.no_grad():
            _, predicted_tokens = torch.max(output, dim=-1)  # Get token predictions
        # (exclude padding/eos tokens from being counted as duplicates)
        non_padding_tokens = predicted_tokens[(predicted_tokens != model.vocab_size - 1) & (predicted_tokens != model.vocab_size - 2)]
        unique_counts = torch.stack([(predicted_tokens == token).sum(dim=1) for token in non_padding_tokens.unique()])
        repeat_loss = penalty_weight * unique_counts.sum(dim=0).float().mean() / batch_size  # Cast to float
        total_loss_penalty += repeat_loss
        # Total loss
        loss_combined = loss_ce + loss_ln + repeat_loss
        total_loss_combiened += loss_combined
        
        # Backward pass
        loss_combined.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

    # Calculate training statistics for current epoch
    mean_ce = total_loss_CE / len(train_loader)
    mean_ln = total_loss_LN / len(train_loader)
    mean_combined = total_loss_combiened / len(train_loader)

    return mean_ce, mean_ln, mean_combined

def eval(model, eval_loader, ce_loss, ln_loss, ln_wheight, penalty_weight=0, knet=None, device="mps", prediction_only=False):
    """
    Evaluate K-net/decoder model for one epoch.

    :param model: Decoder model to evaluate.
    :type model: nn.Module
    :param eval_loader: DataLoader for the evaluation data.
    :type eval_loader: torch.utils.data.DataLoader
    :param ce_loss: Custom cross-entropy loss function.
    :type ce_loss: callable
    :param ln_loss: Custom Length-Norm-Loss function for extra regularization.
    :type ln_loss: callable
    :param ln_wheight: Weighting for the Length-Norm-Loss, range [0,1].
    :type ln_wheight: float
    :param penalty_weight: Weighting for the penalty loss, defaults to 0.
    :type penalty_weight: float
    :param knet: Knowledge-Net model, determines the training phase.
    :type knet: nn.Module, optional
    :param prediction_only: If True, only perform projection into latent space, no decoding, defaults to False.
    :type prediction_only: bool, optional

    :return: Tuple containing evaluation metrics of current epoch, average cross-entropy loss, average Length-Norm-Loss, and the combined loss.
    :rtype: tuple(float, float, float)
    """
    model.eval()
    if knet is not None and prediction_only is False:
        knet.eval()

    total_loss_CE = 0.0
    total_loss_LN = 0.0
    total_loss_penalty = 0.0
    total_loss_combined = 0.0

    with torch.no_grad():
        for conditionals, input_seq, target_seq, input_obs  in eval_loader:
            # Move tensors to the correct device
            conditionals = conditionals.to(device)
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            input_obs = input_obs.to(device)

            batch_size = conditionals.shape[0]

            # Flexible forward pass
            if knet:
                ### To be implemented! ###
                #output, embedding, attention_weights = knet(input_obs, input_seq)
                pass
            else:
                output, embedding, _ = model(conditionals, input_seq)

            # Custom Cross-entropy loss (weighted)
            loss_ce = ce_loss(output, target_seq) * (1 - ln_wheight) / batch_size 
            total_loss_CE += loss_ce

            # Custom Length-Norm-Loss (wheighted)
            loss_ln = ln_loss(embedding, target_seq, model.vocab_size) * ln_wheight / batch_size
            total_loss_LN += loss_ln 

            # Penalty loss for dublicates in the output sequence
            _, predicted_tokens = torch.max(output, dim=-1)  # Get token predictions
            non_padding_tokens = predicted_tokens[(predicted_tokens != model.vocab_size - 1) & (predicted_tokens != model.vocab_size - 2)]
            unique_tokens = non_padding_tokens.unique() # Check if non_padding_tokens is not empty before stacking
            if unique_tokens.numel() > 0:
                unique_counts = torch.stack([(predicted_tokens == token).sum(dim=1) for token in unique_tokens])
                repeat_loss = penalty_weight * unique_counts.sum(dim=0).float().mean() / batch_size  # Cast to float
            else:
                repeat_loss = torch.tensor(0.0, device=device)
            total_loss_penalty += repeat_loss

            # Total loss
            loss_combined = loss_ce + loss_ln + repeat_loss
            total_loss_combined += loss_combined

    # Calculate evaluation statistics for current epoch
    mean_ce = total_loss_CE / len(eval_loader)
    mean_ln = total_loss_LN / len(eval_loader)
    mean_combined = total_loss_combined / len(eval_loader)

    return mean_ce, mean_ln, mean_combined

def generate_sequence(model, conditionals, device="mps", max_length=None, data_embedding=None):
    """
    Generates a sequence of knowledge states using a trained decoder model.
    Predicts a knowledge structure from data observations or reconstructs K from response data observations.

    :param model: Trained decoder model.
    :type model: nn.Module
    :param conditionals: Conditional probabilities of underlying parametarized knowledge structure.
    :type conditionals: torch.Tensor
    :param device: Device to use for inference, defaults to "mps".
    :type device: str
    :param max_length: Maximum length of the generated sequence, defaults to the maximum possible sequence length of identified domain.
    :type max_length: int, optional
    :param data_embedding: Data embedding, defaults to None.
    :type data_embedding: torch.Tensor, optional

    :return: Tuple containing the generated sequence and corresponding embedding.
    :rtype: tuple(list, torch.Tensor
    """
    if max_length is None: max_length = 2 ** model.m_items + 1 # maximum seq length
    
    # Initialize generated sequence with index of start token 
    generated_sequence = [0] # zero for empty state 

    model.eval()
    with torch.no_grad():
        # iteratively generate the output sequence 
        for _ in range(max_length):
            # generated sequence so far to tensor
            current_seq = torch.tensor([generated_sequence], dtype=torch.long).long().to(device)

            # Predict the next token
            output, embedding, _ = model(conditionals, current_seq, data_embedding=data_embedding)

            #self_attention = attention[0].cpu()
            #attention_weights += [self_attention[0,-1,:]]
            
            # add newly predicted token to sequence 
            next_token_id = output[-1].argmax(dim=-1).item()
            generated_sequence.append(next_token_id)
            
            # Break if EOS token is generated
            if next_token_id == model.vocab_size - 2:
                break

    return generated_sequence, embedding


# To do: pass regressor instead of embedding to generate_sequence_(MCDP), so embeddings are repreatedly computed, too, if dropout in projection layers (?)
# To do: consider flexibly handling batched/unbatched data? 
def generate_sequence_MCDP(
    model, # ensure correct device
    conditionals, # ensure correct device
    device="mps", 
    max_length=None, 
    data_embedding=None, 
    n_mc_samples=5
):
    """
    Generates a sequence of knowledge states using a trained decoder model with Monte Carlo Dropout (MCDP).
    Predicts a knowledge structure from data observations or reconstructs K from response data observations.

    :param model: Trained decoder model.
    :type model: nn.Module
    :param conditionals: Conditional probabilities of underlying parametarized knowledge structure.
    :type conditionals: torch.Tensor
    :param device: Device to use for inference, defaults to "mps".
    :type device: str
    :param max_length: Maximum length of the generated sequence, defaults to the maximum possible sequence length of identified domain.
    :type max_length: int, optional
    :param data_embedding: Data embedding, defaults to None.
    :type data_embedding: torch.Tensor, optional
    :param n_mc_samples: Number of Monte Carlo samples for dropout, defaults to 5.
    :type n_mc_samples: int, optional

    :return: Tuple containing the generated sequence and corresponding embedding.
    :rtype: tuple(list, torch.Tensor)
    """
    if max_length is None:
        max_length = 2 ** model.m_items + 1  # Maximum full sequence length
    
    # Initialize generated sequence with index of start token (empty state) 
    generated_sequence = [0]  # Zero for empty state as start token
    K_embedding_batch = torch.Tensor()

    # Enable only dropout layers during MCDP inference
    model.eval()
    def enable_mc_dropout(model):
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()
    
    enable_mc_dropout(model)

    with torch.no_grad():
        for _ in range(max_length):
            # Convert the generated sequence so far to a tensor
            current_seq = torch.tensor([generated_sequence], dtype=torch.long).long().to(device)

            # Perform multiple MC forward passes to predict the next token
            mc_outputs = []
            mc_embeddings = []
            for _ in range(n_mc_samples):
                output, embedding, _ = model(conditionals, current_seq, data_embedding=data_embedding)
                mc_outputs.append(output)
                mc_embeddings.append(embedding)

            # Aggregate outputs and embeddings
            mc_outputs = torch.stack(mc_outputs)  # Shape: [n_mc_samples, seq_len, vocab_size]
            mc_embeddings = torch.stack(mc_embeddings)  # Shape: [n_mc_samples, embedding_dim]

            # Majority voting for the next token
            token_predictions = mc_outputs[:, -1, :].argmax(dim=-1)  # Shape (n_mc_samples)
            # Mode calculation not implemented for MPS, so move to CPU (todo: analyse overhead...)
            token_predictions_cpu = token_predictions.cpu()  
            next_token_id = token_predictions_cpu.mode(dim=0).values.item() 

            # Add newly predicted token to the sequence
            generated_sequence.append(next_token_id)

            # Break if EOS token is generated
            if next_token_id == model.vocab_size - 2:
                break

    return generated_sequence, K_embedding_batch  

def performance(model, dataloader, regressor=None, num_samples=100, n_mc_samples=None, log=False, device="mps"):
    """
    Evaluate the performance of a trained decoder model using a given dataloader.
    Computes the accuracy, average symbolic difference, and standard deviation of the symbolic difference.
    
    :param model: Trained decoder model.
    :type model: nn.Module
    :param dataloader: DataLoader for the evaluation data.
    :type dataloader: torch.utils.data.DataLoader
    :param regressor: Regressor model for data embedding, defaults to None.
    :type regressor: nn.Module, optional
    :param num_samples: Number of samples to evaluate, defaults to 100.
    :type num_samples: int
    :param n_mc_samples: Number of Monte Carlo samples for dropout, defaults to None.
    :type n_mc_samples: int, optional
    :param log: Whether to log the performance metrics, defaults to False.
    :type log: bool
    :param device: Device to use for inference, defaults to "mps".
    :type device: str

    :return: Tuple containing the accuracy, average symmetric difference, and standard deviation of symmetric differences.
    :rtype: tuple(float, float, float)
    """
    eos_token_id = model.vocab_size - 2
    pad_token_id = model.vocab_size - 1

    # Accumulators for perfromance meassures 
    correct_predictions = 0   
    avr_sym_diffs = []

    # Determine if MC dropout is enabled
    mcdp = "enabled" if n_mc_samples else "disabled"

    # Assumes batch size 1
    # Create an iterator from the dataloader
    dataloader_iter = iter(dataloader)

    # Loop over the number of samples and retrieve a batch in each iteration
    for _ in tqdm(range(num_samples), desc=f"Testing performance with MCDP {mcdp}..."):
        try:
            batch = next(dataloader_iter)
            conditionals, input_seq, target_seq, input_obs = batch
           
            conditionals = conditionals.to(device)
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            input_obs = input_obs.to(device)

            # Generate sequence flexibly with or w/o MCDP
            if n_mc_samples is None:
                output_seq, emb = generate_sequence(model, conditionals)
            else: 
                output_seq, emb = generate_sequence_MCDP(model, conditionals.to(device), n_mc_samples=n_mc_samples)

            # Convert outputs and targets to list
            output_seq_lst = [t for t in output_seq if t != pad_token_id and t != eos_token_id]
            #output_seq_lst = output_seq[(output_seq != pad_token_id) & (output_seq != eos_token_id)].tolist()
            # To do: fix for batche processing 
            target_seq_lst = [0] + [t for t in target_seq[0].tolist() if t != pad_token_id and t != eos_token_id]

            # Compute dissimilarity between generated and target sequence
            sym_diff = len(set(output_seq_lst) ^ set(target_seq_lst))
            avr_sym_diffs.append(sym_diff)
            # Check if generated sequence matches the target sequence
            if output_seq_lst == target_seq_lst:
                correct_predictions += 1
        
        except StopIteration:
            # If the dataloader runs out of data, break the loop
            break
    
    # Calculate accuracy mean and standard deviation
    acc = correct_predictions / num_samples
    avr_sym_diff = np.mean(avr_sym_diffs)
    std_sym_diff = np.std(avr_sym_diffs)

    return acc, avr_sym_diff, std_sym_diff

def save_model(model, file_name='model_checkpoint.pth', optimizer=None, epoch=None, path=None):
    """
    Save a torch model and optimizer state to file.

    :param model: Model to save.
    :type model: nn.Module
    :param optimizer: Optimizer to save (optional).
    :type optimizer: torch.optim.Optimizer or None
    :param epoch: Current epoch (optional).
    :type epoch: int or None
    :param path: Path to save the model and optimizer state.
    :type path: str
    """
    if path is None:
        # Get the directory of the current script or notebook
        base_dir = Path(__file__).resolve().parent.parent
        models_dir = base_dir / "data" / "models" 

        if file_name is None: 
            # Find highest index in directory and increment
            files = [f for f in models_dir.iterdir() if f.is_file() and f.suffix == ".pth"]
            if not files:
                path = models_dir / f"model_00.pth"
            else:
                files.sort()
                last_file = files[-1]
                last_index = int(last_file.stem.split("_")[-1])
                new_index = last_index + 1
                path = models_dir / f"model_{new_index:02d}.pth"
        else:
            path = models_dir / file_name

    checkpoint = {'model_state_dict': model.state_dict()}
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    torch.save(checkpoint, path)
    print(f"Model saved to {path}.")

def load_model(model, file_name='decoder_00.pth', optimizer=None):
    """
    Load a torch model and optimizer state from file.

    :param model: Model to load.
    :type model: nn.Module
    :param optimizer: Optimizer to load (optional).
    :type optimizer: torch.optim.Optimizer or None
    :param file_name: Name of the file/model to load.
    :type file_name: str
    :return: Loaded model, optimizer (if provided), and epoch (if provided).
    :rtype: tuple
    """
    # Get the model directory path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "data/models")
    path = os.path.join(model_dir, file_name)

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', None)

    print(f"Model loaded from {path}.")
    return model, optimizer, epoch
