import random as random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Dataloader
import math

## Util Functions
CHAR_SPACE = [chr(i) for i in range(ord('0'), ord('9')+1)] + [chr(i) for i in range(ord('a'), ord('z')+1)] + [' ']
ENGLISH_PRIOR_ORDER = [' ', 'e', 't', 'a', 'i', 'n', 'o', 'r', 's', 'h', 'l', 'd', 'c', 'u', 'm', 'f', 'g', 'p', 'w', 'b', 'y', 'v', 'k', '1', '0', '2', '9', 'j', 'x', '3', '5', '8', '4', 'z', '6', '7', 'q']

def preprocess_str(text):
    result = text.lower()

    # Filter out all non-number/alphabetical characters except space
    result = list([val for val in result if val in CHAR_SPACE])
    result = ''.join(result)

    # Replace multiple spaces with one space
    result = result.split()
    result = ' '.join(result)

    return result

class CharTokenizer():
    def __init__(self):
        vocabs = [chr(i) for i in range(ord('0'), ord('9')+1)] + [chr(i) for i in range(ord('a'), ord('z')+1)] + [' ']
        self.c2id  = {vocab:i for i, vocab in enumerate(vocabs)}
        self.id2c  = {i:vocab for i, vocab in enumerate(vocabs)}

    def encode(self, text):
        assert isinstance(text, list), "The input to the tokenizer needs to be a list of characters"

        return [self.c2id[c] for c in text]

    def decode(self, ids):
        return [self.id2c[id] for id in ids]

class Subst_cipher():
    def __init__(self, key=None, domain=[chr(i) for i in range(ord('0'), ord('9')+1)] + [chr(i) for i in range(ord('a'), ord('z')+1)]):
        if key == None:
            key = [ch for ch in domain]
            random.shuffle(key)

        self.p2c = {plain:cipher for plain, cipher in zip(domain, key)}
        self.c2p = {cipher:plain for plain, cipher in zip(domain, key)}

    def encrypt(self, plaintext):
        return [self.p2c[ch] if ch in self.p2c else ch for ch in plaintext]

    def decrypt(self, ciphertext):
        return [self.c2p[ch] if ch in self.p2c else ch for ch in ciphertext]
  


def compute_frequecy(text, space=CHAR_SPACE):
    """
    text: list of strings or string
    """
    count = {c: 0 for c in space}
    total = 0

    if isinstance(text, list):
        for line in text:
            for c in line:
                count[c] += 1
                total += 1
    else:
        for c in text:
            count[c] += 1
            total += 1

    freq_sorted_vocab = sorted(count, key=lambda x: count[x], reverse=True) 

    freq = {c: count[c] / total for c in freq_sorted_vocab}

    return freq, freq_sorted_vocab


def unigram_freq_decipher(cipher, english_prior_order=ENGLISH_PRIOR_ORDER):

    _, cipher_freq_order_list = compute_frequecy(cipher)

    english_prior_order = list(range(0, 37))
    
    lis = []
    for i in range(len(cipher)):
        idx = cipher_freq_order_list.index(cipher[i])
        lis.append(english_prior_order[idx])

    return ''.join(lis)

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



def most_freq_decipher(cipher, english_prior_order=ENGLISH_PRIOR_ORDER):

    return english_prior_order[0] * len(cipher)


def character_accuracy(predicted, target):
    '''
    Calculate the characteer-level accuracy of the input string compared to target string

    :param: target: The target string
    :param: predicted: The predicted string
    :return: The accuracy as a percentage
    '''

    if len(target) == 0:
        return 0.0 # or should we print error message

    correct = sum(1 for i, j in zip(target, predicted) if i == j)
    return (correct / len(target))

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

    def forward(self, Q, K, V, mask=None):
        
        Q, K, V = self.w_q(Q), self.w_k(K), self.w_v(V)       # dot product with weight matrices
        Q, K, V = self.split(Q), self.split(K), self.split(V) # split tensor by number of heads
        out, attention = self.attention(Q, K, V, mask=mask)
        
    def split(self, tensor):
        pass


class SelfAttention(nn.Module):
    def __init__(self, embed_size, n_heads):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=n_heads)

    def forward(self, x):
        return self.attn(x, x, x, need_weights=False)[0]

class TransformerBlock(nn.Module):
    def __init__(self, embed_size=1024, n_heads=8, expansion_factor=4):
        super(TransformerBlock, self).__init__()

        self.ln_1 = nn.LayerNorm(embed_size)
        self.attn = SelfAttention(embed_size, n_heads)
        self.ln_2 = nn.LayerNorm(embed_size)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, expansion_factor * embed_size),
            nn.GELU(),
            nn.Linear(expansion_factor * embed_size, embed_size),
        )

    def forward(self, x):
        x = self.attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x
        return x


class Encoder(nn.Module):
    def __init__(self, seq_len, vocab_size, embed_size, num_blocks=4, expansion_factor=4, n_heads=8, device='cpu'):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.seq_len    = seq_len
        self.device     = device
        self.word_embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.pos_embedding  = nn.Embedding(self.seq_len, self.embed_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_size,n_heads,expansion_factor)
            for _ in range(num_blocks)
        ])
        self.output = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        # (batch_size, seq_len)
        pos = torch.arange(0, x.shape[-1], dtype=torch.long, device=self.device).unsqueeze(0) # shape (1, seq_len)
        pos_embeded = self.pos_embedding(pos)
        x = self.word_embedding(x) + pos_embeded
        for block in self.blocks:
            x = block(x)
        x = self.output(x)

        return x # (batch_size, seq_len, vocab_size) logits and not probabilities
    
        

class CiphertextDataset(Dataset):
    def __init__(self, data_source, tokenizer, window_size = 2048, device='cpu'):
        self.data = data_source
        self.window_size = window_size
        self.tokenizer = tokenizer
        self.device = device
        self.total_substrings = len(self.data) - window_size + 1
        
    def __len__(self):
        return self.total_substrings

    def __getitem__(self, idx):
        if idx >= self.total_substrings:
            raise IndexError("Index out of bounds")
        plaintext = self.data[idx: idx + self.window_size]
        cipher = Subst_cipher()
        encrypted_text = cipher.encrypt(plaintext)

        plain_tokens = torch.tensor(self.tokenizer.encode([c for c in plaintext]), dtype=torch.long)
        cipher_tokens = torch.tensor(self.tokenizer.encode([c for c in encrypted_text]), dtype=torch.long)

        return plain_tokens, cipher_tokens

def train(model, dataLoader, optimizer, loss_func, char_tokenizer, n_epochs=10):
    for i in range(n_epochs):
        model.train()
        epoch_loss = 0
        j = 0
        print("epoch: ",i)
        for src, trg in dataLoader:
            j+=1

            # here
            optimizer.zero_grad()
            output = model(src)

            output = output.view(-1, output.shape[-1])
            trg = trg.view(-1)

            loss = loss_func(output, trg)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            out_in = output.argmax(dim=-1)
            acc = character_accuracy(out_in,trg)
            if j % 1000 == 999:
                print("loss of batch",j,":",epoch_loss / len(dataLoader))
                print("accuracy:", acc)
    
def evaluate(model, testLoader, loss_func):
    model.eval()
    epoch_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for src, trg in testLoader:
            output = model(src)

            output = output.view(-1, output.shape[-1])
            trg = trg.view(-1)

            loss = loss_func(output, trg)
            epoch_loss += loss.item()

            predicted  = output.argmax(dim=-1)
            trg_labels = trg
            
            accuracy = character_accuracy(predicted, trg_labels)
            total_accuracy += accuracy
            
    average_loss = epoch_loss / len(testLoader)
    average_accuracy = total_accuracy / len(testLoader)

    return average_loss, average_accuracy

if __name__ == "__main__":
    [' ', 'e', 't', 'a', 'i', 'n', 'o', 'r', 's', 'h', 'l', 'd', 'c', 'u', 'm', 'f', 'g', 'p', 'w', 'b', 'y', 'v', 'k', '1', '0', '2', '9', 'j', 'x', '3', '5', '8', '4', 'z', '6', '7', 'q']


