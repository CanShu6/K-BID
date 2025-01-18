import os.path
import random

import pandas as pd
from tqdm import tqdm


def remap(df):
    df = df.merge(right=user_remap, left_on='inviter_id', right_on='raw_user_id', how='left') \
        [['user_id', 'raw_item_id', 'voter_id', 'timestamp']]
    df.rename(columns={'user_id': 'inviter_id'}, inplace=True)

    df = df.merge(right=user_remap, left_on='voter_id', right_on='raw_user_id', how='left') \
        [['inviter_id', 'raw_item_id', 'user_id', 'timestamp']]
    df.rename(columns={'user_id': 'voter_id'}, inplace=True)

    df = df.merge(right=item_remap, on='raw_item_id', how='left') \
        [['inviter_id', 'item_id', 'voter_id', 'timestamp']]

    print('fields remapped.')
    return df


def make_train():
    with open('./vid-train.inter', 'w', encoding='utf-8') as file:
        file.write('inviter_id,share_item,voter_id,item_seq,label\n')
        for _, row in tqdm(sampled_train_df.iterrows(), total=len(sampled_train_df)):

            item_seq = make_pos(row['voter_id'], row['timestamp'])

            if len(item_seq) == 0:
                continue
            file.write(f"{row['inviter_id']},{row['item_id']},"
                       f"{row['voter_id']},{' '.join([str(x) for x in item_seq])},1\n")

            neg_user_id, item_seq = make_neg(row['inviter_id'], row['timestamp'])
            file.write(f"{row['inviter_id']},{row['item_id']},"
                       f"{neg_user_id},{' '.join([str(x) for x in item_seq])},0\n")
    print('train samples done.')


def make_test():
    with open('./vid-test.inter', 'w', encoding='utf-8') as file:
        file.write('id,inviter_id,share_item,voter_id,timestamp\n')
        for idx, (_, row) in tqdm(enumerate(sampled_test_df.iterrows()), total=len(sampled_test_df)):

            item_seq = make_pos(row['voter_id'], row['timestamp'])

            if len(item_seq) == 0:
                continue
            file.write(f"{idx},{row['inviter_id']},{row['item_id']},{row['voter_id']},{row['timestamp']}\n")

            # neg_users = []
            # while len(neg_users) < TEST_NEG_NUM:
            #     neg_user_id, item_seq = make_neg(row['inviter_id'], row['timestamp'], neg_users)
            #     neg_users.append(neg_user_id)
            #     file.write(f"{row['inviter_id']},{row['item_id']},{neg_user_id},{item_seq},0\n")
    print('test samples done.')


def make_pos(voter_id, timestamp):
    """
    :param voter_id
    :param timestamp
    """
    item_seq = user2item[(user2item['user_id'] == voter_id) & (user2item['timestamp'] < timestamp)]
    item_seq = item_seq.sort_values('timestamp')['item_id'].tolist()[-MAX_SEQ_LEN:]
    return item_seq


def make_neg(inviter_id, timestamp, others=None):
    """
    :param inviter_id
    :param timestamp
    :param others
    """
    others = [] if others is None else others
    relate_users = user2user[inviter_id] + others
    neg_user_id = random.randint(1, user_num - 1)
    item_seq = user2item[(user2item['user_id'] == neg_user_id) & (user2item['timestamp'] < timestamp)]
    item_seq = item_seq.sort_values('timestamp')['item_id'].tolist()[-MAX_SEQ_LEN:]

    while neg_user_id in relate_users or len(item_seq) == 0:
        neg_user_id = random.randint(1, user_num - 1)
        item_seq = user2item[(user2item['user_id'] == neg_user_id) & (user2item['timestamp'] < timestamp)]
        item_seq = item_seq.sort_values('timestamp')['item_id'].tolist()[-MAX_SEQ_LEN:]

    return neg_user_id, item_seq


if __name__ == '__main__':

    user_remap = pd.read_csv('./remap_user_id.csv')
    item_remap = pd.read_csv('./remap_item_id.csv')

    user_num = user_remap['user_id'].max() + 1
    item_num = item_remap['item_id'].max() + 1

    tianchi_train = pd.read_csv('./item_share_train_info_sorted.csv', header=0, sep=',',
                                names=['inviter_id', 'raw_item_id', 'voter_id', 'timestamp'])
    tianchi_train['timestamp'] = pd.to_datetime(tianchi_train['timestamp'])
    tianchi_train = remap(tianchi_train)

    tianchi_test = pd.read_csv('./item_share_test_info_sorted.csv', header=0, sep=',',
                               names=['inviter_id', 'raw_item_id', 'voter_id', 'timestamp'])
    tianchi_test['timestamp'] = pd.to_datetime(tianchi_test['timestamp'])
    tianchi_test = remap(tianchi_test)

    user2item = pd.concat(
        [tianchi_train[['inviter_id', 'item_id', 'timestamp']].rename(columns={'inviter_id': 'user_id'}),
         tianchi_train[['voter_id', 'item_id', 'timestamp']].rename(columns={'voter_id': 'user_id'})],
        axis=0, ignore_index=True)
    user2item.drop_duplicates(['user_id', 'item_id', 'timestamp'], ignore_index=True, inplace=True)
    user2item.sort_values(['user_id', 'timestamp'], inplace=True)

    user2user = pd.concat([tianchi_train[['inviter_id', 'voter_id']],
                           tianchi_train[['voter_id', 'inviter_id']].rename(
                               columns={'voter_id': 'inviter_id', 'inviter_id': 'voter_id'})],
                          axis=0, ignore_index=True)
    user2user.drop_duplicates(['inviter_id', 'voter_id'], ignore_index=True, inplace=True)
    user2user = user2user.groupby('inviter_id').agg(list).reset_index()
    user2user = dict(zip(user2user['inviter_id'], user2user['voter_id']))

    MAX_SEQ_LEN = 20
    TEST_NEG_NUM = 10
    DATASET_VERSION = 2

    if DATASET_VERSION == 1:

        sampled_train_df = tianchi_train.iloc[:int(len(tianchi_train) * 0.8)]
        sampled_test_df = tianchi_train[~tianchi_train.index.isin(sampled_train_df.index)]
    else:

        sampled_train_df = tianchi_train
        sampled_test_df = tianchi_test

    if not os.path.exists('./vid-train.inter'):
        make_train()

    if not os.path.exists('./vid-test.inter'):
        make_test()
