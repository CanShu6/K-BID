import torch
from torch import nn
from torch.nn.init import xavier_normal_, constant_

from model.layers.att_seq_layer import SequenceAttLayer
from model.layers.mlp_layer import MLPLayers
from model.layers.self_attention import MultiHeadSelfAttention


class VID(nn.Module):
    def __init__(self, args, feature_nunique, device):
        super(VID, self).__init__()
        self.device = device
        self.max_seq_len = args.max_seq_len
        self.emb_dim = args.emb_dim
        self.mlp_hidden_size = eval(args.mlp_hidden_size)
        self.dropout_prob = args.dropout_prob

        # init Embedding layer
        self.item_emb = nn.Embedding(feature_nunique['item_id'], self.emb_dim)
        self.inviter_emb = nn.Embedding(feature_nunique['user_id'], self.emb_dim)
        self.voter_emb = nn.Embedding(feature_nunique['user_id'], self.emb_dim)

        # init Attention layer
        self.use_short_pref = args.use_short_pref
        mask_mat = torch.arange(self.max_seq_len).to(self.device).view(1, -1)
        self.att_list = [4 * self.emb_dim] + self.mlp_hidden_size
        self.attention = SequenceAttLayer(mask_mat, self.att_list, activation="Sigmoid",
                                          softmax_stag=False, return_seq_weight=False,)

        # init Self Attention Layer
        self.use_self_attention = args.use_self_attention
        if self.use_self_attention == 1:
            self.self_attention = MultiHeadSelfAttention(self.emb_dim, self.emb_dim, args.head_num)

        # init MLP layers
        if self.use_short_pref:
            self.dnn_list = [4 * self.emb_dim] + self.mlp_hidden_size
        else:
            self.dnn_list = [3 * self.emb_dim] + self.mlp_hidden_size
        self.dnn_mlp_layers = MLPLayers(self.dnn_list, activation="Dice", dropout=self.dropout_prob, bn=True)

        # init Prediction layer
        self.dnn_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        # self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCEWithLogitsLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def calc_cross_score(self, inviter_id, share_item, voter_id, item_seq_len, item_seq):
        """
        :param inviter_id:      [batch_size]
        :param share_item:      [batch_size]
        :param voter_id:        [batch_size]
        :param item_seq_len:    [batch_size]
        :param item_seq:        [batch_size, max_seq_len]
        :return pred:           [batch_size]
        """
        batch_size = inviter_id.size()[0]
        inviter_emb = self.inviter_emb(inviter_id)                  # [batch_size, emb_dim]
        share_item_feat_emb = self.item_emb(share_item)             # [batch_size, emb_dim]
        voter_emb = self.voter_emb(voter_id)                        # [batch_size, emb_dim]
        item_seq_feat_emb = self.item_emb(item_seq)                 # [batch_size, seq_len, emb_dim]

        # attention
        if self.use_self_attention:
            if self.use_short_pref:
                pooled_item_seq_emb = self.attention(share_item_feat_emb, item_seq_feat_emb, item_seq_len)
                pooled_item_seq_emb = pooled_item_seq_emb.squeeze(1)  # [batch_size, emb_dim]
                att_in = torch.cat([inviter_emb.unsqueeze(1), share_item_feat_emb.unsqueeze(1),
                                    pooled_item_seq_emb.unsqueeze(1), voter_emb.unsqueeze(1)],
                                   dim=1)                             # [batch_size, 4, emb_dim]
            else:
                att_in = torch.cat([inviter_emb.unsqueeze(1), share_item_feat_emb.unsqueeze(1), voter_emb.unsqueeze(1)],
                                   dim=1)                           # [batch_size, 3, emb_dim]

            att_out = self.self_attention(att_in)                   # [batch_size, 4(3), emb_dim]
            vid_in = att_out.reshape(batch_size, -1)                # [batch_size, emb_dim * 4(3)]
        else:
            if self.use_short_pref:
                pooled_item_seq_emb = self.attention(share_item_feat_emb, item_seq_feat_emb, item_seq_len)
                pooled_item_seq_emb = pooled_item_seq_emb.squeeze(1)    # [batch_size, emb_dim]
                vid_in = torch.cat([inviter_emb, share_item_feat_emb,
                                    pooled_item_seq_emb, voter_emb],
                                   dim=-1)                              # [batch_size, emb_dim * 4]
            else:
                vid_in = torch.cat([inviter_emb, share_item_feat_emb, voter_emb],
                                   dim=-1)                              # [batch_size, emb_dim * 3]

        # get the prediction score
        vid_out = self.dnn_mlp_layers(vid_in)
        pred = self.dnn_predict_layer(vid_out)                      # [batch_size, 1]
        pred = pred.squeeze(1)                                      # [batch_size]
        return pred

    def calc_u2u_score(self, inviter_id, voter_id):
        """
        Calculate the rate between users
        :param inviter_id:      [batch_size]
        :param voter_id:        [batch_size]
        """
        inviters = self.inviter_emb(inviter_id)
        voters = self.voter_emb(voter_id)
        pred = torch.sum(inviters * voters, dim=1)
        return pred

    def forward(self, inviter_id, share_item, voter_id, item_seq_len, item_seq, label):
        pred = self.calc_cross_score(inviter_id, share_item, voter_id, item_seq_len, item_seq)
        cross_loss = self.bce_loss(pred, label)
        # pred = self.calc_u2u_score(inviter_id, voter_id)
        # u2u_loss = self.bce_loss(pred, label)

        # loss = cross_loss + u2u_loss
        return cross_loss

    def predict(self, inviter_id, share_item, voter_id, item_seq_len, item_seq):
        cross_score = self.calc_cross_score(inviter_id, share_item, voter_id, item_seq_len, item_seq)
        # u2u_score = self.calc_u2u_score(inviter_id, voter_id)
        return cross_score
