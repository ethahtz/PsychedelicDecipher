import argparse
import torch
from model import Encoder, CharTokenizer
from trainer import train
from data import get_datasets
import time

current_time = time.strftime("%Y%m%d_%H%M")

parser = argparse.ArgumentParser(description='Train a PsychedelicDecipher model')

parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--n_steps', type=int, default=2500, help='Number of steps')
parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--mask_ratio', type=float, default=0.0, help='Ratio of masked characters')
parser.add_argument('--window_size', type=int, default=1024, help='Window size')
parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size')
parser.add_argument('--n_blocks', type=int, default=4, help='Number of blocks')
parser.add_argument('--n_heads', type=int, default=8, help='Number of heads')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
parser.add_argument('--model_path', type=str, default=None, help='Path to save the model')
parser.add_argument('--add_noise', action='store_true', help='Add noise to the input data')
parser.add_argument('--raw_inputs', action='store_true', help='Use raw inputs (no frequency encoding)')
parser.add_argument('--no_cnn', action='store_true', help='Do not use CNN in the model')

args = parser.parse_args()

if args.model_path is None:
    args.model_path = f'./ckpts/model_WSIZE_{args.window_size}{"_noise" if args.add_noise else ""}{"_raw" if args.raw_inputs else ""}{"_woCNN" if args.no_cnn else ""}_{current_time}.pt'


if torch.cuda.is_available():
    device = torch.cuda.current_device()
else:
    device = 'cpu'

train_dataset, test_dataset = get_datasets()

char_tokenizer = CharTokenizer()

my_tokenizer = CharTokenizer()
model = Encoder(args.window_size, 37+1, args.hidden_size, num_blocks=args.n_blocks, n_heads=args.n_heads, dropout=args.dropout, use_cnn=(not args.no_cnn)) # window_size, vocab_size, embed_size
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)
loss_func = torch.nn.CrossEntropyLoss()

train(model, train_dataset, optimizer, loss_func, my_tokenizer, 
      n_steps=args.n_steps, 
      batch_size=args.batch_size, 
      n_epochs=args.n_epochs, 
      device=device, 
      mask_ratio=args.mask_ratio, 
      to_save=True, 
      noise=args.add_noise,
      raw_inputs=args.raw_inputs,
      save_path=args.model_path)