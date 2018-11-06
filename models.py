import torch
import torch.nn as nn
import torch.nn.functional as F


class DAN(nn.Module):
    def __init__(self, hidden_sizes, output_size, vocab_size=None, embedding_size=None, pretrained_vecs=None, elmo_config=None):
        super(DAN, self).__init__()
        if elmo_config:
            from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder

            self.emb_layer = ElmoTokenEmbedder(*elmo_config)
        else:
            self.emb_layer = nn.Embedding(vocab_size, embedding_size)
            self.emb_layer.weight.data.copy_(pretrained_vecs)

        self.inp_layer = nn.Linear(embedding_size, hidden_sizes[0])
        self.out_layer = nn.Linear(hidden_sizes[-1], output_size)

        self.hidden = nn.ModuleList()
        for k in range(len(hidden_sizes) - 1):
            self.hidden.append(nn.Linear(hidden_sizes[k], hidden_sizes[k + 1]))

    def forward(self, x):
        x = self.average(self.emb_layer(x))
        x = self.inp_layer(x)
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.out_layer(x)

    def average(self, embeddings):
        return torch.mean(embeddings, 1)
