import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchtext
import argparse
from torchtext.data.utils import get_tokenizer
import re


# Transformer model class(just encoder)
class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):  # ninp = embedding dimension
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.ninp = ninp
        self.src_mask = None
        self.encoder = nn.Embedding(ntoken, ninp)  # embedding
        self.pos_encoder = PositionalEncoding(
            ninp, dropout)  # positional embedding
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, nlayers)  # Encoder
        self.decoder = nn.Linear(ninp, ntoken)  # linear decoder
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        encoder_output = self.encoder(src) * math.sqrt(self.ninp)
        pos_encoder_output = self.pos_encoder(encoder_output)
        transformer_encoder_output = self.transformer_encoder(
            pos_encoder_output, self.src_mask)
        output = self.decoder(transformer_encoder_output)
        return encoder_output, pos_encoder_output, transformer_encoder_output, F.log_softmax(output, dim=-1)


# Positional Embedding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):  # d_model = embedding dimension
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1)  # add dimension at dimension1
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def gen_csr(m):
    # print(m.shape)
    num_blocks_row = int(m.shape[0] / 16)
    num_blocks_col = int(m.shape[1] / 16)
    # print('num_blocks_row: ', num_blocks_row)
    # print('num_blocks_col: ', num_blocks_col)
    num_of_zeros = 0
    all = 0
    data = []
    column = []
    row_ptr = [0]
    for i in range(num_blocks_row):
        row_ptr_temp = 0
        for j in range(num_blocks_col):
            all = all + 1
            temp_block = m[16*i:16*i+16, 16*j:16*j+16]
            if np.all(temp_block == 0):
                num_of_zeros = num_of_zeros + 1

            else:
                data.append(np.array(temp_block.flatten()))
                column.append(j)
                row_ptr_temp = row_ptr_temp + 1
        row_ptr.append(row_ptr_temp + row_ptr[-1])

    data = np.array(data)
    # print(len(data[0]))
    row_ptr = np.array(row_ptr)
    column = np.array(column)
    # print('num of zero matrix: ', num_of_zeros)
    # print('all matrix: ', all)
    # print('shape of data: ', data.shape)
    # print('shape of row_ptr', len(row_ptr))
    # print('shape of column', len(column))
    return row_ptr.astype(np.int32), column.astype(np.int32), data.astype(np.float32)


def save_npz(name, W, B):
    # print(name)
    if len(W.shape) == 2:
        row_ptr, column, data = gen_csr(W)
        np.savez(name, row_ptr=row_ptr, column=column,
                 data=data, B=B.astype(np.float32))
    else:
        np.savez(name, W=W.astype(np.float32), B=B.astype(np.float32))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset--WikiText2
    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                init_token='<sos>',
                                eos_token='<eos>',
                                lower=True)
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train_txt)
    ntokens = len(TEXT.vocab.stoi)  # the size of vocabulary
    model = TransformerModel(ntokens, args.emsize, args.nhead,
                             args.nhid, args.nlayers, args.dropout).to(device)
    model.load_state_dict(torch.load(
        args.model_file, map_location=torch.device('cpu')))

    layer_weights = {}
    reW = re.compile(r".*weight$")
    reB = re.compile(r".*bias$")

    for param_tensor in model.state_dict():
        if (param_tensor != "encoder.weight" and
                param_tensor != "pos_encoder.pe"):
            weight = np.array(
                model.state_dict()[param_tensor].cpu().detach().numpy())
            if reW.match(param_tensor):
                name = param_tensor[:-6]
                wtype = 0
            elif reB.match(param_tensor):
                name = param_tensor[:-4]
                wtype = 1
            else:
                raise ValueError
            if name not in layer_weights:
                layer_weights[name] = [None, None]
            layer_weights[name][wtype] = weight
    weight_names = []
    for name, weights in layer_weights.items():
        filename = f"{args.weight_folder}/{name}"
        # if re.match(r".*in_proj_$", name):
        #     Ws = np.split(weights[0], 3)
        #     Bs = np.split(weights[1], 3)
        #     save_npz(f"{filename}Q", Ws[0], Bs[0])
        #     save_npz(f"{filename}K", Ws[1], Bs[1])
        #     save_npz(f"{filename}V", Ws[2], Bs[2])
        #     weight_names += [f"{name}Q.npz\n", f"{name}K.npz\n", f"{name}V.npz\n"]
        # else:
        #     save_npz(f"{filename}", weights[0], weights[1])
        #     weight_names.append(f"{name}.npz\n")
        save_npz(f"{filename}", weights[0], weights[1])
        weight_names.append(f"{name}.npz\n")
    f = open(f"{args.weight_folder}/weights.txt", "w+")
    f.writelines(weight_names)
    f.close()

# hyperparameters


def parse_args():
    parser = argparse.ArgumentParser(description='Transformer Traning')
    parser.add_argument('--lr', default=2.32, help='learning rate')
    parser.add_argument('--device', default='cuda', help='device')
    # Transformer parameters
    parser.add_argument('--emsize', default=800, type=int,
                        help='token embedding dimension')
    parser.add_argument('--nhid', default=200, type=int,
                        help='hidden dimension')
    parser.add_argument('--nhead', default=4, type=int, help='head number')
    parser.add_argument('--nlayers', default=2, type=int, help='layer number')
    parser.add_argument('--dropout', default=0.2,
                        type=float, help='default dropout value')
    parser.add_argument('--train-batch-size', default=20, type=int,
                        help='train batch size, smaller for PC or larger for server')
    parser.add_argument('--eval-batch-size', default=10,
                        type=int, help='evaluate batch size')
    parser.add_argument('--bptt', default=35, type=int, help='')
    parser.add_argument('--model_file', default='', type=str, help='')
    parser.add_argument('--weight_folder', default='', type=str, help='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)

# 需要修改的变量
# epoch
# learning rate
# load model
