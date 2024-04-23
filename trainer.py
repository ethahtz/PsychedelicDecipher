import torch
from metrics import character_accuracy
import numpy as np
from utils import Subst_cipher
from model import freq_encoder
from tqdm import trange


def mask_tokens(sequences, mask_token=37, mask_ratio=0.05):
    sequences_tensor = sequences
    
    random_mask = torch.rand(sequences_tensor.shape)
    
    mask = random_mask < mask_ratio
    
    sequences_tensor[mask] = mask_token
    
    return sequences_tensor

def get_next_batch(train_dataset, tokenizer, window_size=1024, batch_size=128, mask_ratio=0.05):
    indices = np.random.randint(0, len(train_dataset) - window_size, batch_size)
    text_samples = [train_dataset[i: i+window_size] for i in indices]
    ciphertexts = []
    for text in text_samples:
        curr_cipher = Subst_cipher()
        ciphertext = curr_cipher.encrypt(text)
        ciphertext = ''.join(ciphertext)
        ciphertexts.append(ciphertext)

    tgt = torch.tensor([tokenizer.encode(seq) for seq in text_samples], dtype=torch.long)
    src = torch.tensor([freq_encoder(tokenizer.encode(seq)) for seq in ciphertexts], dtype=torch.long)
    src = mask_tokens(src, mask_ratio=mask_ratio)

    return src, tgt


def train(model, train_dataset, optimizer, loss_func, char_tokenizer, n_steps=10000, batch_size=128, n_epochs=3, mask_ratio=0.05, device='cpu'):
    for i in range(n_epochs):
        model.train()
        epoch_loss = 0
        print("epoch: ", i)

        p_bar = trange(n_steps, desc=f'Epoch {i}', unit='iters')

        for j in p_bar:
            src, trg = get_next_batch(train_dataset, char_tokenizer, model.window_size, batch_size, mask_ratio=mask_ratio)

            src = src.to(device)
            trg = trg.to(device)

            optimizer.zero_grad()
            output, attn_weights = model(src)

            output = output.view(-1, output.shape[-1])
            trg = trg.view(-1)

            loss = loss_func(output, trg)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            out_in = output.argmax(dim=-1)
            acc = character_accuracy(out_in, trg)

            p_bar.set_postfix(loss=loss.item() / batch_size, accuracy=acc)


def evaluate(model, test_dataset, char_tokenizer, loss_func, batch_size, n_batches=100, device='cpu'):
    model.eval()
    epoch_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        p_bar = trange(n_batches, desc=f'Epoch {i}', unit='iters')
        for j in p_bar:
            src, trg = get_next_batch(test_dataset, char_tokenizer, model.window_size, batch_size)
            
            src.to(device)
            trg.to(device)
            
            output = model(src)

            output = output.view(-1, output.shape[-1])
            trg = trg.view(-1)

            loss = loss_func(output, trg)
            epoch_loss += loss.item()

            predicted  = output.argmax(dim=-1)
            trg_labels = trg

            accuracy = character_accuracy(predicted, trg_labels)
            total_accuracy += accuracy

            p_bar.set_postfix(loss=loss.item() / batch_size, accuracy=accuracy)

    average_loss = epoch_loss / n_batches
    average_accuracy = total_accuracy / n_batches

    return average_loss, average_accuracy