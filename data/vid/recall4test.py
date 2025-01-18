import json

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
    return df


if __name__ == '__main__':
    MAX_SEQ_LEN = 20


    user_remap = pd.read_csv('remap_user_id.csv')
    item_remap = pd.read_csv('remap_item_id.csv')


    tianchi_train = pd.read_csv('item_share_train_info_sorted.csv', sep=',', header=0,
                                names=['inviter_id', 'raw_item_id', 'voter_id', 'timestamp'])
    tianchi_train['timestamp'] = pd.to_datetime(tianchi_train['timestamp'])
    tianchi_train = remap(tianchi_train)


    user2item = pd.concat([tianchi_train[['inviter_id', 'item_id', 'timestamp']].rename(columns={'inviter_id': 'user_id'}),
                           tianchi_train[['voter_id', 'item_id', 'timestamp']].rename(columns={'voter_id': 'user_id'})],
                          axis=0, ignore_index=True)
    user2item.drop_duplicates(['user_id', 'item_id', 'timestamp'], ignore_index=True, inplace=True)


    test_dataset = pd.read_csv('./vid-test.inter', sep=',', header=0,
                               names=['id', 'inviter_id', 'share_item', 'voter_id', 'timestamp'])
    test_dataset['timestamp'] = pd.to_datetime(test_dataset['timestamp'])


    u2u_recall_dict = json.load(open('recall/U2U-Recall.json', 'r', encoding='utf-8'))
    i2u_recall_dict = json.load(open('recall/I2U-Recall.json', 'r', encoding='utf-8'))


    with open('./vid-recall.inter', 'w', encoding='utf-8') as file:
        file.write('id,recall_voter,recall_item_seq\n')
        for _, row in tqdm(test_dataset.iterrows(), total=len(test_dataset)):
            uid = row['id']
            inviter_id = row['inviter_id']
            share_item = row['share_item']
            label_voter_id = row['voter_id']
            share_time = row['timestamp']

            recall_voters = set(u2u_recall_dict[str(inviter_id)] + i2u_recall_dict[str(share_item)])

            for recall_voter in recall_voters:
                item_seq = user2item[(user2item['user_id'] == recall_voter) & (user2item['timestamp'] < share_time)]
                item_seq = item_seq.sort_values('timestamp')['item_id'].tolist()[-MAX_SEQ_LEN:]

                file.write(f'{uid},{recall_voter},{" ".join([str(x) for x in item_seq])}\n')

