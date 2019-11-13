import os
import math

import torch
import torch.nn as nn

class AttDecoder(nn.Module):
    """
    """
    def __init__(self, num_classes, num_layers=2, encoding_size=1024,
                 num_heads=8, dropout_p=0.0):
        """
        """
        super().__init__()

        self.layers = nn.ModuleList()
        for l in range(num_layers - 1):
            self.layers.append(nn.Linear(in_features=encoding_size,
                                         out_features=encoding_size))
        # classification layer
        self.layers.append(nn.Linear(in_features=encoding_size,
                                     out_features=num_classes))

        # attention mechanism
        if encoding_size % num_heads != 0:
            raise ValueError("The encoding size is not a multiple of num heads.")
        self.num_heads = num_heads
        self.head_size = int(encoding_size / num_heads)
        self.score_projection = nn.Linear(in_features=encoding_size,
                                          out_features=num_heads)
        self.key_projection = nn.Linear(in_features=encoding_size,
                                        out_features=encoding_size)
        self.value_projection = nn.Linear(in_features=encoding_size,
                                          out_features=self.head_size * num_heads)
        self.dropout_p = dropout_p

    def aggregate(self, encoding):
        """
        """
        print(encoding.shape)
        # flatten encoding
        batch_size, encoding_size, length, height, width = encoding.shape
        encoding = encoding.view(
            batch_size, encoding_size, -1).permute(0, 2, 1)

        mixed_values = self.value_projection(encoding)
        new_shape = mixed_values.size()[:-1] + (self.num_heads,
                                                self.head_size)
        values = mixed_values.view(*new_shape).permute(0, 2, 1, 3)

        keys = self.key_projection(encoding)
        scores = self.score_projection(keys)
        scores = scores / math.sqrt(self.head_size)
        scores = torch.nn.functional.softmax(scores, dim=1)
        scores = scores.permute(0, 2, 1)  # batch, heads, dim

        contexts = torch.matmul(scores.unsqueeze(2), values)
        contexts_flat = contexts.squeeze(2).view(batch_size, -1)
        return contexts_flat

    def classify(self, x):
        """
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, encoding):
        """
        """
        emb = self.aggregate(encoding)
        out = self.classify(emb)
        return out
