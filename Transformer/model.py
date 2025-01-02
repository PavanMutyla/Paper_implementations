import torch 
import torch.nn as nn
import math 
from Transformer.Input_embeddings import input_embeddings
from Transformer.Positional_encoding import positional_encoding
from Transformer.Layer_normalization import Layer_norm
from Transformer.Feed_forward_module import feed_forward
from Transformer.Multi_head_attention import multihead_attention
from Transformer.Encoder import Encoder
from Transformer.Decoder import Decoder
class Transformer(nn.Module):
    def __init__(self, dim : int, src_vocab_size, target_vocab_size, heads, layers, d_ff, max_sequence, dropout ):
        super().__init__()
        self.encoder_embeddings = input_embeddings(dim, src_vocab_size)
        self.decoder_embeddings = input_embeddings(dim, target_vocab_size)
        #positional encodings
        self.positional_embeddings = positional_encoding(dim, max_sequence, dropout)
        #layers encoder and decoder
        self.encoder_layers = nn.ModuleList([Encoder(dim, heads, dropout, d_ff) for _ in range(layers)]) # multiple encoder layers
        self.decoder_layers = nn.ModuleList([Decoder(dim, heads, dropout, d_ff) for _ in range(layers)]) # multiple decoder layers
        #fc and dropout
        self.fc = nn.Linear(dim, target_vocab_size)
        self.dropout = nn.Dropout(dropout)
    def generate_masks(self, src, target):
        src_mask = (src !=0 ).unsqueeze(1).unsqueeze(2) # True for non-zero elements
        target_mask = (target != 0 ).unsqueeze(1).unsqueeze(3)
        seq_len = target.size(1)
        diagonal_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool() # diagonal masking for tokens
        target_maks = target_mask & diagonal_mask
        return src_mask, target_mask
    def forward(self, src, target):
        src_mask, target_mask = self.generate_masks(src, target)
        # processed embeddings
        src_embeddings = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        target_embeddings = self.dropout(self.positional_embeddings(self.encoder_embedding(target)))
        enc_output = src_embedded
        
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
