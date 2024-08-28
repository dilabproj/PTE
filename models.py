import math
import torch
from torch import nn
import random

# =======================================================================================================
# =======================================================================================================
def generate_square_subsequent_mask(dim1, dim2):
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

# =======================================================================================================
# =======================================================================================================
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# =======================================================================================================
# =======================================================================================================
class PTE(nn.Module):
    
    def __init__(self, feature_size=316, input_length=96, output_length=12, num_layers=1, dropout=0.1):
        super(PTE, self).__init__()

        # encoder
        self.positional_encoding_layer = PositionalEncoding(feature_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  
        
        # decoder (linear)
        self.decoder = nn.Linear(feature_size, 1)
        self.fc = nn.Sequential(
          nn.Linear(input_length, input_length),
          nn.Linear(input_length, output_length)
        )
        
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src, label, evaluate):

        # encoder
        src = self.positional_encoding_layer(src)   # [512, 96, 316]
        output = self.transformer_encoder(src)      # [512, 96, 316]
        
        # decoder (linear)
        output = self.decoder(output)      # [512, 96, 1]
        output = torch.flatten(output, 1)  # [512, 96]
        output = self.fc(output)           # [512, 12]
        output = output.unsqueeze(2)       # [512, 12, 1]

        return output

# =======================================================================================================

