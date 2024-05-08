import torch
import torch.nn as nn
import math
from utils import CHAR_SPACE

def freq_encoder(cipher):
    """
    Encode frequency information into the cipher text

    """
    count = {c: 0 for c in range(37)}

    for c in cipher:
        count[c] += 1

    freq_sorted_vocab = sorted(count, key=lambda x: count[x], reverse=True)

    lis = []
    for i in range(len(cipher)):
        idx = freq_sorted_vocab.index(int(cipher[i]))
        lis.append(idx)

    return lis


class CharTokenizer():
    def __init__(self):
        vocabs = CHAR_SPACE
        self.c2id  = {vocab:i for i, vocab in enumerate(vocabs)}
        self.id2c  = {i:vocab for i, vocab in enumerate(vocabs)}

    def encode(self, text):
        return [self.c2id[c] for c in text]

    def decode(self, ids):
        return [self.id2c[id] for id in ids]

class AttentionHead(nn.Module):
    '''
    Query : given sentence that we focused on
    Key   : every sentence to check relationship with Qeury
    Value : every sentence same with Key
    '''
    def __init__(self):
        super(AttentionHead, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = K.size()

        K_T = K.transpose(2, 3)
        score = (Q @ K_T) / math.sqrt(d_tensor)

        score = self.softmax(score)
        v = score @ v

        return v, score

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        self.n_head = num_heads
        self.attention = AttentionHead()
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.w_concat = nn.Linear(embed_dim, embed_dim)

class SelfAttention(nn.Module):
    def __init__(self, embed_size, n_heads):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=n_heads)

    def forward(self, x):
        x = x.transpose(0, 1)
        x, attn_weights = self.attn(x, x, x, need_weights=True)
        x = x.transpose(0, 1)
        return x, attn_weights


class CharCNN(nn.Module):
    def __init__(self, d_model=128):
        super(CharCNN, self).__init__()
        self.c3 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=(3,), padding=1)
        self.c5 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=(5,), padding=2)
        self.c7 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=(7,), padding=3)
        self.c9 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=(9,), padding=4)
        self.c11 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=(11,), padding=5)
        self.c3sep = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=(1,))
        self.c5sep = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=(1,))
        self.c7sep = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=(1,))
        self.c9sep = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=(1,))
        self.c11sep = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=(1,))
        self.c3_prime = nn.Conv1d(in_channels=d_model*5, out_channels=d_model, kernel_size=(3,), padding=1)
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.leakyrelu3 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.leakyrelu4 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.leakyrelu5 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.leakyrelu6 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv_layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x): # (batch, seqLen, n_embed)

        x_in = x

        x1 = x.transpose(1,2)
        x1 = self.c3(x1)
        x1 = self.c3sep(x1)
        x1 = self.leakyrelu1(x1)
        
        x1 = x1.transpose(1,2)

        x2 = x.transpose(1,2)
        x2 = self.c5(x2)
        x2 = self.c5sep(x2)
        x2 = self.leakyrelu2(x2)
        
        x2 = x2.transpose(1,2)

        x3 = x.transpose(1,2)
        x3 = self.c7(x3)
        x3 = self.c7sep(x3)
        x3 = self.leakyrelu3(x3)
        
        x3 = x3.transpose(1,2)
        

        x4 = x.transpose(1,2)
        x4 = self.c9(x4)
        x4 = self.c9sep(x4)
        x4 = self.leakyrelu4(x4)
        
        x4 = x4.transpose(1,2)

        x5 = x.transpose(1,2)
        x5 = self.c11(x5)
        x5 = self.c11sep(x5)
        x5 = self.leakyrelu5(x5)
        
        x5 = x5.transpose(1,2)
        

        x = torch.cat((x1, x2, x3, x4, x5), 2)
        
        x = x.transpose(1,2)
        x = self.c3_prime(x)
        x = self.leakyrelu6(x)
        x = x.transpose(1,2)

        x = x_in + x
        
        x = self.conv_layer_norm(x)

        return x
    
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size=1024, n_heads=8, expansion_factor=4, dropout=0.1, use_cnn=True, use_attn=True):
        super(TransformerBlock, self).__init__()

        

        if not use_cnn:
            self.cnn = nn.Identity()
        else:
            self.cnn = CharCNN(d_model=embed_size)
        
        if not use_attn:
            self.attn = nn.Identity()
            self.use_attn = False
        else:
            self.attn = SelfAttention(embed_size, n_heads)

        self.ln_1 = nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)
        self.ln_2 = nn.LayerNorm(embed_size)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, expansion_factor * embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion_factor * embed_size, embed_size),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        
        x = self.cnn(x)

        if self.use_attn:
            x_interm, attn_weights = self.attn(self.ln_1(x))
        else:
            x_interm = self.ln_1(x)
            attn_weights = None

        x = self.dropout1(x_interm) + x
        x = self.dropout2(self.mlp(self.ln_2(x))) + x
        return x, attn_weights

    

class Encoder(nn.Module):
    def __init__(self, window_size, vocab_size, embed_size, num_blocks=4, expansion_factor=4, n_heads=8, dropout=0.1, all_activations=False, use_cnn=True, use_attn=True):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.window_size    = window_size
        self.all_activations = all_activations
        self.word_embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.pos_embedding  = nn.Embedding(self.window_size, self.embed_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_size, n_heads, expansion_factor, dropout=dropout, use_cnn=use_cnn, use_attn=use_attn)
            for _ in range(num_blocks)
        ])
        self.output = nn.Linear(embed_size, vocab_size)
        
        

    def forward(self, x):
        # (batch_size, window_size)
        pos = torch.arange(0, x.shape[-1], dtype=torch.long, device=x.device).unsqueeze(0) # shape (1, window_size)
        pos_embedded = self.pos_embedding(pos)

        x = self.word_embedding(x) + pos_embedded

        activations = []

        if self.all_activations:
            activations.append(x.cpu())

        for block in self.blocks:
            x, attn_weights = block(x)
            if self.all_activations:
                activations.append(x.cpu())
            
        x = self.output(x)
        
        return x, activations # (batch_size, window_size, vocab_size) logits and not probabilities
