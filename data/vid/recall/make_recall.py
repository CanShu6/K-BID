import json
import os.path

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def gen_u2u_recall():
    output_file = 'U2U-Recall.json'
    if os.path.exists(output_file):
        print('Skip User KG Embedding Recall.')
        return

    kg_id2e_remap = pd.read_csv('../user-kg/entity_remap.csv', sep=',',
                                header=0, names=['remap_entity_id', 'raw_entity_id'])
    kg_e2id_dict = dict(zip(kg_id2e_remap['raw_entity_id'], kg_id2e_remap['remap_entity_id']))

    user_embedding = np.load('../../../pretrain/user-kg/embed_dim176/user_embed.npy')

    if USE_FAISS:

        index = faiss.IndexFlatIP(user_embedding.shape[1])
        index.add(user_embedding)

        _, uid_mat = index.search(np.ascontiguousarray(user_embedding), RECALL_NUM + 1)

        exclude_self_mat = []
        for row_idx in tqdm(range(uid_mat.shape[0]), desc='Excluding Interacted Users'):
            new_row = []
            for col_idx in range(uid_mat.shape[1]):
                if uid_mat[row_idx, col_idx] == row_idx:
                    continue
                if len(new_row) < RECALL_NUM:
                    new_row.append(uid_mat[row_idx, col_idx])
                else:
                    break
            assert len(new_row) == RECALL_NUM, 'Recall Num not satisfied!'
            exclude_self_mat.append(new_row)

        uid_mat = pd.DataFrame(exclude_self_mat, columns=[f'uid_col{u}' for u in range(RECALL_NUM)])
    else:

        user_embedding = torch.from_numpy(user_embedding)
        u2u_sim = torch.matmul(user_embedding, torch.transpose(user_embedding, 0, 1))
        user_num = u2u_sim.size()[0]

        x = y = list(range(user_num))
        u2u_sim[x, y] = -np.inf
        _, uid_mat = u2u_sim.topk(RECALL_NUM)
        uid_mat = pd.DataFrame(uid_mat.numpy(), columns=[f'uid_col{u}' for u in range(RECALL_NUM)])

    for column in uid_mat.columns:
        uid_mat[column] = uid_mat[[column]].merge(right=kg_id2e_remap, left_on=column,
                                                  right_on='remap_entity_id', how='left')['raw_entity_id']

    recalls = dict()
    for vid_inviter_id in tqdm(test_dataset['inviter_id'], desc='Generating Recalls'):
        raw_inviter_id = vid_id2u_dict[vid_inviter_id]
        kg_inviter_id = kg_e2id_dict[raw_inviter_id]
        recalls[vid_inviter_id] = [vid_u2id_dict[u] for u in uid_mat.iloc[kg_inviter_id].tolist()]

    json.dump(recalls, open(output_file, 'w', encoding='utf-8'))
    print('User KG Embedding Recall done.')
    return recalls


def gen_i2u_recall():
    output_file = './I2U-Recall.json'
    if os.path.exists(output_file):
        print('Skip Item KG Embedding Recall.')
        return

    kg_id2e_remap = pd.read_csv('../item-kg/entity_remap.csv', sep=',',
                                header=0, names=['remap_entity_id', 'raw_entity_id'])
    kg_e2id_dict = dict(zip(kg_id2e_remap['raw_entity_id'], kg_id2e_remap['remap_entity_id']))

    kg_id2u_remap = pd.read_csv('../item-kg/user_remap.csv', sep=',',
                                header=0, names=['remap_user_id', 'raw_user_id'])

    tianchi_train = pd.read_csv('../item_share_train_info_sorted.csv', sep=',', header=0,
                                names=['inviter_id', 'item_id', 'voter_id', 'timestamp'])
    cf_i2u_df = pd.concat([tianchi_train[['inviter_id', 'item_id']].rename(columns={'inviter_id': 'user_id'}),
                           tianchi_train[['voter_id', 'item_id']].rename(columns={'voter_id': 'user_id'})],
                          ignore_index=True).drop_duplicates(ignore_index=True)

    cf_i2u_df['user_id'] = cf_i2u_df[['user_id']].merge(right=kg_id2u_remap, left_on='user_id', right_on='raw_user_id',
                                                        how='left')['remap_user_id']
    cf_i2u_df['item_id'] = \
        cf_i2u_df[['item_id']].merge(right=kg_id2e_remap, left_on='item_id', right_on='raw_entity_id',
                                     how='left')['remap_entity_id']

    cf_i2u_df = cf_i2u_df.groupby('item_id').agg({'user_id': list}).reset_index()
    cf_i2u_dict = dict(zip(cf_i2u_df['item_id'], cf_i2u_df['user_id']))
    max_interacted_users = max([len(x) for x in cf_i2u_dict.values()])

    user_embedding = np.load('../../../pretrain/item-kg/embed_dim176/user_embed.npy')
    item_embedding = np.load('../../../pretrain/item-kg/embed_dim176/item_embed.npy')

    if USE_FAISS:

        index = faiss.IndexFlatIP(user_embedding.shape[1])
        index.add(user_embedding)

        _, iid_mat = index.search(np.ascontiguousarray(item_embedding), RECALL_NUM + max_interacted_users)

        exclude_inter_mat = []
        for row_idx in tqdm(range(iid_mat.shape[0]), desc='Excluding Interacted Users'):
            if row_idx not in cf_i2u_dict.keys():
                exclude_inter_mat.append(iid_mat[row_idx].tolist()[:RECALL_NUM])
            else:
                new_row = []
                for col_idx in range(iid_mat.shape[1]):
                    if iid_mat[row_idx, col_idx] in cf_i2u_dict[row_idx]:
                        continue
                    if len(new_row) < RECALL_NUM:
                        new_row.append(iid_mat[row_idx, col_idx])
                    else:
                        break
                assert len(new_row) == RECALL_NUM, 'Recall Num not satisfied!'
                exclude_inter_mat.append(new_row)

        iid_mat = pd.DataFrame(exclude_inter_mat, columns=[f'iid_col{u}' for u in range(RECALL_NUM)])
    else:
        i2u_sim = torch.matmul(item_embedding, torch.transpose(user_embedding, 0, 1))
        item_num = i2u_sim.size()[0]

        for i in tqdm(range(item_num), desc='Excluding Interacted Users'):
            if i in cf_i2u_dict.keys():
                i2u_sim[i, cf_i2u_dict[i]] = -np.inf

        _, iid_mat = i2u_sim.topk(RECALL_NUM)
        iid_mat = pd.DataFrame(iid_mat.numpy(), columns=[f'iid_col{u}' for u in range(RECALL_NUM)])

    for column in iid_mat.columns:
        iid_mat[column] = iid_mat[[column]].merge(right=kg_id2u_remap, left_on=column,
                                                  right_on='remap_user_id', how='left')['raw_user_id']

    recalls = dict()
    for vid_item_id in tqdm(test_dataset['share_item'], desc='Generating Recalls'):
        raw_item_id = vid_id2i_dict[vid_item_id]
        kg_item_id = kg_e2id_dict[raw_item_id]
        recalls[vid_item_id] = [vid_u2id_dict[u] for u in iid_mat.iloc[kg_item_id].tolist()]

    json.dump(recalls, open(output_file, 'w', encoding='utf-8'))
    print('Item KG Embedding Recall done.')
    return recalls


if __name__ == '__main__':
    RECALL_NUM = 20
    USE_FAISS = True

    test_dataset = pd.read_csv('../vid-test.inter', sep=',', header=0,
                               names=['id', 'inviter_id', 'share_item', 'voter_id', 'timestamp'])

    vid_u2id_remap = pd.read_csv('../remap_user_id.csv', sep=',',
                                 header=0, names=['raw_user_id', 'remap_user_id'])
    vid_id2u_dict = dict(zip(vid_u2id_remap['remap_user_id'], vid_u2id_remap['raw_user_id']))
    vid_u2id_dict = dict(zip(vid_u2id_remap['raw_user_id'], vid_u2id_remap['remap_user_id']))

    vid_i2id_remap = pd.read_csv('../remap_item_id.csv', sep=',',
                                 header=0, names=['raw_item_id', 'remap_item_id'])
    vid_id2i_dict = dict(zip(vid_i2id_remap['remap_item_id'], vid_i2id_remap['raw_item_id']))
    vid_i2id_dict = dict(zip(vid_i2id_remap['raw_item_id'], vid_i2id_remap['remap_item_id']))

    gen_u2u_recall()
    gen_i2u_recall()
