import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Aggregator


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class SUR(nn.Module):

    def __init__(self, args, n_users, n_entities, n_relations, A_in=None,
                 user_pre_embed=None, item_pre_embed=None):

        super(SUR, self).__init__()
        self.args = args
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embed_dim = args.embed_dim

        self.aggregation_type = args.aggregation_type
        self.conv_dim_list = [args.embed_dim] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.social_l2loss_lambda = args.social_l2loss_lambda

        self.entity_embed = nn.Embedding(self.n_entities, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.embed_dim)

        # Parameters of SUR Module
        self.h_trans_w1, self.h_trans_w2, self.h_bias_b, self.r_trans_w1, self.r_trans_w2, self.r_bias_b = \
            [nn.Parameter(torch.Tensor(self.embed_dim)) for _ in range(6)]
        self.sem_trans_w = nn.Parameter(torch.Tensor(2 * self.embed_dim, self.embed_dim))

        # Graph Signal Propagation
        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type))

        # Attentive Factors
        self.A_in = nn.Parameter(torch.sparse.FloatTensor(self.n_entities, self.n_entities))
        if A_in is not None:
            self.A_in.data = A_in
        self.A_in.requires_grad = False

        # Pretrained Embeddings
        if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
            other_entity_embed = nn.Parameter(
                torch.Tensor(self.n_entities - item_pre_embed.shape[0] - user_pre_embed.shape[0], self.embed_dim))
            nn.init.xavier_uniform_(other_entity_embed)
            entity_embed = torch.cat([user_pre_embed, item_pre_embed, other_entity_embed], dim=0)
            self.entity_embed.weight = nn.Parameter(entity_embed)
        else:
            nn.init.xavier_uniform_(self.entity_embed.weight)

        # Init Params
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.sem_trans_w)
        nn.init.constant_(self.h_trans_w1, 1)
        nn.init.constant_(self.h_trans_w2, 1)
        nn.init.constant_(self.r_trans_w1, 1)
        nn.init.constant_(self.r_trans_w2, 1)
        nn.init.constant_(self.h_bias_b, 0)
        nn.init.constant_(self.r_bias_b, 0)


    def calc_social_embeddings(self):
        ego_embed = self.entity_embed.weight
        all_embed = [ego_embed]

        for idx, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(ego_embed, self.A_in)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        all_embed = torch.cat(all_embed, dim=1)         # (n_users + n_entities, concat_dim)
        return all_embed


    def calc_social_loss(self, inviter_ids, voter_pos_ids, voter_neg_ids):
        """
        inviter_ids:       (social_batch_size)
        voter_pos_ids:   (social_batch_size)
        voter_neg_ids:   (social_batch_size)
        """
        all_embed = self.calc_social_embeddings()                         # (n_entities, concat_dim)
        inviter_embed = all_embed[inviter_ids]                        # (social_batch_size, concat_dim)
        voter_pos_embed = all_embed[voter_pos_ids]                    # (social_batch_size, concat_dim)
        voter_neg_embed = all_embed[voter_neg_ids]                    # (social_batch_size, concat_dim)

        pos_score = torch.sum(inviter_embed * voter_pos_embed, dim=1)   # (social_batch_size)
        neg_score = torch.sum(inviter_embed * voter_neg_embed, dim=1)   # (social_batch_size)

        social_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        social_loss = torch.mean(social_loss)

        l2_loss = _L2_loss_mean(inviter_embed) + _L2_loss_mean(voter_pos_embed) + _L2_loss_mean(voter_neg_embed)
        loss = social_loss + self.social_l2loss_lambda * l2_loss
        return loss


    def calc_sur(self, h_embed, r_embed):
        batch_size = h_embed.shape[0]
        sem = torch.bmm(h_embed.reshape(batch_size, -1, 1),
                        r_embed.reshape(batch_size, 1, -1))  # (kg_batch_size, embed_dim, embed_dim)

        cross_h = torch.matmul(sem, self.h_trans_w1) + \
            torch.matmul(sem.transpose(1, 2), self.r_trans_w2) + \
            self.h_bias_b                                   # (kg_batch_size, embed_dim)
        cross_r = torch.matmul(sem, self.h_trans_w2) + \
            torch.matmul(sem.transpose(1, 2), self.r_trans_w1) + \
            self.r_bias_b                                   # (kg_batch_size, embed_dim)

        return cross_h, cross_r


    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        h_embed = self.entity_embed(h)                                          # (kg_batch_size, embed_dim)
        r_embed = self.relation_embed(r)                                        # (kg_batch_size, relation_dim)
        pos_t_embed = self.entity_embed(pos_t)                                  # (kg_batch_size, embed_dim)
        neg_t_embed = self.entity_embed(neg_t)                                  # (kg_batch_size, embed_dim)

        cross_h, cross_r = self.calc_sur(h_embed, r_embed)
        pred_t = torch.cat([cross_h, cross_r], dim=1) @ self.sem_trans_w        # (kg_batch_size, embed_dim)

        pos_score = torch.sum(pred_t * pos_t_embed, dim=1)                      # (kg_batch_size)
        neg_score = torch.sum(pred_t * neg_t_embed, dim=1)                      # (kg_batch_size)

        kg_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(cross_h) + _L2_loss_mean(cross_r) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss


    def update_attention(self, h_batch, t_batch, r_batch):
        h_embed = self.entity_embed.weight[h_batch]
        r_embed = self.relation_embed.weight[r_batch]
        cross_h, cross_r = self.calc_sur(h_embed, r_embed)

        cross_h_r = torch.cat([cross_h, cross_r], dim=1)
        t_embed = self.entity_embed.weight[t_batch]

        batch_att = torch.sum(F.leaky_relu(cross_h_r @ self.sem_trans_w * t_embed), dim=1)
        return batch_att


    def calc_score(self, inviter_ids, voter_ids):
        """
        inviter_ids:  (n_users)
        voter_ids:  (n_items)
        """
        all_embed = self.calc_social_embeddings()             # (n_entities, concat_dim)
        inviter_embed = all_embed[inviter_ids]            # (n_inviters, concat_dim)
        voter_embed = all_embed[voter_ids]                # (n_voters, concat_dim)

        social_score = torch.matmul(inviter_embed, voter_embed.transpose(0, 1))    # (n_inviters, n_voters)
        return social_score


    def forward(self, *input, mode):
        if mode == 'train_social':
            return self.calc_social_loss(*input)
        if mode == 'train_kg':
            return self.calc_kg_loss(*input)
        if mode == 'update_att':
            return self.update_attention(*input)
        if mode == 'predict':
            return self.calc_score(*input)


