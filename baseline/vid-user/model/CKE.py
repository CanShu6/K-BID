import torch
import torch.nn as nn
import torch.nn.functional as F


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class CKE(nn.Module):

    def __init__(self, args, n_users, n_entities, n_relations, user_pre_embed=None):

        super(CKE, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim

        self.social_l2loss_lambda = args.social_l2loss_lambda
        self.kg_l2loss_lambda = args.kg_l2loss_lambda

        self.user_embed = nn.Embedding(self.n_users, self.embed_dim)
        self.entity_embed = nn.Embedding(self.n_entities, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))

        if (self.use_pretrain == 1) and (user_pre_embed is not None):
            self.user_embed.weight = nn.Parameter(user_pre_embed)
        else:
            nn.init.xavier_uniform_(self.user_embed.weight)

        nn.init.xavier_uniform_(self.entity_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.trans_M)


    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)                 # (kg_batch_size, relation_dim)
        W_r = self.trans_M[r]                            # (kg_batch_size, embed_dim, relation_dim)

        h_embed = self.entity_embed(h)                   # (kg_batch_size, embed_dim)
        pos_t_embed = self.entity_embed(pos_t)           # (kg_batch_size, embed_dim)
        neg_t_embed = self.entity_embed(neg_t)           # (kg_batch_size, embed_dim)

        # Equation (2)
        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)             # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)     # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)     # (kg_batch_size, relation_dim)

        r_embed = F.normalize(r_embed, p=2, dim=1)
        r_mul_h = F.normalize(r_mul_h, p=2, dim=1)
        r_mul_pos_t = F.normalize(r_mul_pos_t, p=2, dim=1)
        r_mul_neg_t = F.normalize(r_mul_neg_t, p=2, dim=1)

        # Equation (3)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)     # (kg_batch_size)

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss


    def calc_social_loss(self, inviter_ids, voter_pos_ids, voter_neg_ids):
        """
        inviter_ids:       (social_batch_size)
        voter_pos_ids:     (social_batch_size)
        voter_neg_ids:     (social_batch_size)
        """
        inviter_embed = self.user_embed(inviter_ids)                      # (social_batch_size, embed_dim)
        voter_pos_embed = self.user_embed(voter_pos_ids)                  # (social_batch_size, embed_dim)
        voter_neg_embed = self.user_embed(voter_neg_ids)                  # (social_batch_size, embed_dim)

        inviter_kg_embed = self.entity_embed(inviter_ids)                 # (social_batch_size, embed_dim)
        voter_pos_kg_embed = self.entity_embed(voter_pos_ids)             # (social_batch_size, embed_dim)
        voter_neg_kg_embed = self.entity_embed(voter_neg_ids)             # (social_batch_size, embed_dim)

        # Equation (5)
        inviter_social_embed = inviter_embed + inviter_kg_embed                 # (social_batch_size, embed_dim)
        voter_pos_social_embed = voter_pos_embed + voter_pos_kg_embed           # (social_batch_size, embed_dim)
        voter_neg_social_embed = voter_neg_embed + voter_neg_kg_embed           # (social_batch_size, embed_dim)

        # Equation (6)
        pos_score = torch.sum(inviter_social_embed * voter_pos_social_embed, dim=1)    # (social_batch_size)
        neg_score = torch.sum(inviter_social_embed * voter_neg_social_embed, dim=1)    # (social_batch_size)

        social_loss = (-1.0) * torch.log(1e-10 + F.sigmoid(pos_score - neg_score))
        social_loss = torch.mean(social_loss)

        l2_loss = _L2_loss_mean(inviter_embed) + _L2_loss_mean(voter_pos_social_embed) + _L2_loss_mean(voter_neg_social_embed)
        loss = social_loss + self.social_l2loss_lambda * l2_loss
        return loss


    def calc_loss(self, inviter_ids, voter_pos_ids, voter_neg_ids, h, r, pos_t, neg_t):
        """
        inviter_ids:     (social_batch_size)
        voter_pos_ids:   (social_batch_size)
        voter_neg_ids:   (social_batch_size)

        h:              (kg_batch_size)
        r:              (kg_batch_size)
        pos_t:          (kg_batch_size)
        neg_t:          (kg_batch_size)
        """
        kg_loss = self.calc_kg_loss(h, r, pos_t, neg_t)
        social_loss = self.calc_social_loss(inviter_ids, voter_pos_ids, voter_neg_ids)
        loss = kg_loss + social_loss
        return loss


    def calc_score(self, inviter_ids, voter_ids):
        """
        inviter_ids:  (n_inviters)
        voter_ids:    (n_voters)
        """
        inviter_embed = self.user_embed(inviter_ids)               # (n_inviters, embed_dim)
        inviter_kg_embed = self.entity_embed(inviter_ids)          # (n_inviters, embed_dim)
        inviter_social_embed = inviter_embed + inviter_kg_embed    # (n_inviters, embed_dim)

        voter_embed = self.user_embed(voter_ids)                   # (n_voters, embed_dim)
        voter_kg_embed = self.entity_embed(voter_ids)              # (n_voters, embed_dim)
        voter_social_embed = voter_embed + voter_kg_embed          # (n_voters, embed_dim)

        social_score = torch.matmul(inviter_social_embed, voter_social_embed.transpose(0, 1))  # (n_inviters, n_voters)
        return social_score


    def forward(self, *input, is_train):
        if is_train:
            return self.calc_loss(*input)
        else:
            return self.calc_score(*input)
