import argparse
import torch
from model import Encoder, CharTokenizer
from trainer import evaluate
from data import get_datasets
import time

current_time = time.strftime("%Y%m%d_%H%M")

parser = argparse.ArgumentParser(description='Evaluate a PsychedelicDecipher model')

parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--model_path', type=str, required=True, help='Path to load the model')
parser.add_argument('--raw_inputs', action='store_true', help='Use raw inputs (no frequency encoding) in the model')
parser.add_argument('--no_cnn', action='store_true', help='Do not use CNN in the model')
parser.add_argument('--no_attn', action='store_true', help='Do not use attention in the model')


args = parser.parse_args()

model_window_size = int(args.model_path.split('/')[-1].split('_')[2])


if torch.cuda.is_available():
    device = torch.cuda.current_device()
else:
    device = 'cpu'

train_dataset, test_dataset = get_datasets()

char_tokenizer = CharTokenizer()

my_tokenizer = CharTokenizer()
model = Encoder(model_window_size, 37+1, 256, num_blocks=4, n_heads=8, dropout=0.0, use_cnn=(not args.no_cnn), use_attn=(not args.no_attn)) # window_size, vocab_size, embed_size
model.load_state_dict(torch.load(args.model_path))
model.to(device)


windows_to_test = [64]

while windows_to_test[-1] < model_window_size:
    windows_to_test.append(windows_to_test[-1] * 2)

for w_size in windows_to_test:
    acc, std = evaluate(model, test_dataset, my_tokenizer, 
                        args.batch_size, 
                        w_size, 
                        n_batches=100, 
                        raw_inputs = args.raw_inputs,
                        device=device)

    print(f"Test size: {w_size}, Accuracy: {acc:.3f} +/- {std:.3f}")

