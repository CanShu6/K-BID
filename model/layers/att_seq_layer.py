import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from model.layers.mlp_layer import MLPLayers


class SequenceAttLayer(nn.Module):
    """Attention Layer. Get the representation of each user in the batch.

    Args:
        queries (torch.Tensor): candidates, [batch_size, emb_dim]
        keys (torch.Tensor): user_hist,     [batch_size, seq_len, emb_dim]
        keys_length (torch.Tensor): mask,   [batch_size, ]

    Returns:
        torch.Tensor: result
    """

    def __init__(
        self,
        mask_mat,
        att_hidden_size=(80, 40),
        activation="sigmoid",
        softmax_stag=False,
        return_seq_weight=True,
    ):
        super(SequenceAttLayer, self).__init__()
        self.att_hidden_size = att_hidden_size
        self.activation = activation
        self.softmax_stag = softmax_stag
        self.return_seq_weight = return_seq_weight
        self.mask_mat = mask_mat
        self.att_mlp_layers = MLPLayers(
            self.att_hidden_size, activation=self.activation, bn=False
        )
        self.dense = nn.Linear(self.att_hidden_size[-1], 1)

    def forward(self, queries, keys, keys_length):
        """
        :param queries:     next_item_feat_emb [batch_size, emb_dim]
        :param keys:        item_seq_feat_emb  [batch_size, seq_len, emb_dim]
        :param keys_length: item_seq_len       [batch_size, ]
        :return pooling:    output             [batch_size, 1, emb_dim]
        """
        embedding_size = queries.shape[-1]
        seq_len = keys.shape[1]
        queries = queries.repeat(1, seq_len)                    # [batch_size, emb_dim * seq_len]
        queries = queries.view(-1, seq_len, embedding_size)     # [batch_size, seq_len, embedding_size]

        # MLP Layer
        input_tensor = torch.cat(
            [queries, keys, queries - keys, queries * keys], dim=-1
        )                                                       # [batch_size, seq_len, embedding_size * 4]
        output = self.att_mlp_layers(input_tensor)
        output = self.dense(output).squeeze(2)                  # [batch_size, seq_len, 1] -> [batch_size, seq_len]

        # get mask with prefix of zero or minimal number
        mask = self.mask_mat.repeat(output.size(0), 1)          # [batch_size, seq_len]
        mask = mask >= keys_length.unsqueeze(1)                 # [batch_size, seq_len]

        if self.softmax_stag:
            mask_value = -np.inf
        else:
            mask_value = 0.0

        output = output.masked_fill(mask=mask, value=torch.tensor(mask_value))  # [batch_size, seq_len]
        output = output / (embedding_size**0.5)                                 # [batch_size, seq_len]

        if self.softmax_stag:
            output = F.softmax(output, dim=1)                                   # [batch_size, seq_len]

        # get pooling according to the weight of each key
        if not self.return_seq_weight:
            output = output.unsqueeze(1).matmul(keys)                           # [batch_size, 1, emb_dim]

        return output
