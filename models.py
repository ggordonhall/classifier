import torch
import torch.nn as nn


class DAN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, emb_layer):
        super(DAN, self).__init__()
        self.emb_layer = emb_layer
        self.inp_layer = nn.Linear(input_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        emb_inp = self.average(self.emb_layer(inp))
        inp_layer = self.inp_layer(emb_inp)
        return self.out_layer(inp_layer)

    def average(self, embeddings):
        return torch.mean(embeddings, 1)
