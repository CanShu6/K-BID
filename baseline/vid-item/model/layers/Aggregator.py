import torch
from torch import nn


class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

        if self.aggregator_type == 'gcn':
            self.linear = nn.Linear(self.in_dim, self.out_dim)       # W in Equation (6)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'graphsage':
            self.linear = nn.Linear(self.in_dim * 2, self.out_dim)   # W in Equation (7)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'bi-interaction':
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)      # W1 in Equation (8)
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)      # W2 in Equation (8)
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)

        else:
            raise NotImplementedError


    def forward(self, ego_embeddings, A_in):
        """
        ego_embeddings:  (n_users + n_entities, in_dim)
        A_in:            (n_users + n_entities, n_users + n_entities), torch.sparse.FloatTensor
        """
        # Equation (3)
        side_embeddings = torch.matmul(A_in, ego_embeddings)

        if self.aggregator_type == 'gcn':
            # Equation (6) & (9)
            embeddings = ego_embeddings + side_embeddings
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'graphsage':
            # Equation (7) & (9)
            embeddings = torch.cat([ego_embeddings, side_embeddings], dim=1)
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'bi-interaction':
            # Equation (8) & (9)
            sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
            bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))
            embeddings = bi_embeddings + sum_embeddings

        embeddings = self.message_dropout(embeddings)           # (n_users + n_entities, out_dim)
        return embeddings
