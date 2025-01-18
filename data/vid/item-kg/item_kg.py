import pandas as pd


def create_triples(h, r, t):
    triples = pd.DataFrame(columns=['h', 'r', 't'])
    triples['h'] = h
    triples['t'] = t
    triples['r'] = r
    return triples


def remap(raw_dfs, remap_df, left_on='all', right_on='raw_entity_id', remap_field='remap_entity_id'):
    for raw_df in raw_dfs:
        if raw_df is None or type(raw_df) is not pd.DataFrame or raw_df.size == 0:
            continue
        if left_on == 'all':
            for column in raw_df.columns:
                raw_df[column] = raw_df.merge(right=remap_df, left_on=column, right_on=right_on,
                                              how='left')[remap_field]
        else:
            raw_df[left_on] = raw_df.merge(right=remap_df, left_on=left_on, right_on=right_on,
                                           how='left')[remap_field]


if __name__ == '__main__':
    DATASET_VERSION = 3

    tianchi_train = pd.read_csv('../item_share_train_info_sorted.csv', sep=',', header=0,
                                names=['inviter_id', 'item_id', 'voter_id', 'timestamp'])
    tianchi_train['timestamp'] = pd.to_datetime(tianchi_train['timestamp'])

    tianchi_test = pd.read_csv('../item_share_test_info_sorted.csv', sep=',', header=0,
                               names=['inviter_id', 'item_id', 'voter_id', 'timestamp'])

    if DATASET_VERSION == 1:
        kg_train = tianchi_train.iloc[:int(len(tianchi_train) * 0.8)]
        kg_test = tianchi_train[~tianchi_train.index.isin(kg_train.index)]
    elif DATASET_VERSION == 2:
        kg_train = tianchi_train
        kg_test = tianchi_test
    else:
        kg_train = tianchi_train
        kg_test = []

    user_info = pd.read_csv('../user_info.csv', sep=',', header=0,
                            names=['user_id', 'user_gender', 'user_age', 'user_level'])

    item_info = pd.read_csv('../item_info.csv', sep=',', header=0,
                            names=['item_id', 'cate_id', 'cate_level1_id', 'brand_id', 'shop_id'])
    item_info['cate_id'] = item_info['cate_id'].apply(lambda x: 'cate_' + str(x))
    item_info['cate_level1_id'] = item_info['cate_level1_id'].apply(lambda x: 'cate_level1_' + str(x))
    item_info['brand_id'] = item_info['brand_id'].apply(lambda x: 'brand_' + str(x))
    item_info['shop_id'] = item_info['shop_id'].apply(lambda x: 'shop_' + str(x))

    entity_remap = pd.DataFrame()
    entity_remap['raw_entity_id'] = pd.concat([item_info['item_id'],
                                               item_info['cate_id'].drop_duplicates(),
                                               item_info['cate_level1_id'].drop_duplicates(),
                                               item_info['brand_id'].drop_duplicates(),
                                               item_info['shop_id'].drop_duplicates()], ignore_index=True)
    entity_remap.reset_index(names='remap_entity_id', inplace=True)
    entity_remap.to_csv('./entity_remap.csv', sep=',', index=False)

    user_remap = user_info[['user_id']].reset_index().rename(
        columns={'index': 'remap_user_id', 'user_id': 'raw_user_id'})
    user_remap.to_csv('./user_remap.csv', sep=',', index=False)
    print('KG remap file done.')

    remap([kg_train, kg_test], user_remap, left_on='inviter_id', right_on='raw_user_id', remap_field='remap_user_id')
    remap([kg_train, kg_test], user_remap, left_on='voter_id', right_on='raw_user_id', remap_field='remap_user_id')
    remap([kg_train, kg_test], entity_remap, left_on='item_id')
    remap([item_info], entity_remap)

    kg_final = pd.DataFrame(columns=['h', 'r', 't'])
    kg_final = pd.concat([kg_final,
                          create_triples(item_info['item_id'], 0, item_info['cate_id']),
                          create_triples(item_info['item_id'], 1, item_info['cate_level1_id']),
                          create_triples(item_info['item_id'], 2, item_info['brand_id']),
                          create_triples(item_info['item_id'], 3, item_info['shop_id'])],
                         axis=0, ignore_index=True)
    kg_final.to_csv('./kg_final.txt', sep=' ', index=False, header=False)
    print('KG triple file done.')

    cf = pd.concat([kg_train[['inviter_id', 'item_id', 'timestamp']].rename(columns={'inviter_id': 'user_id'}),
                    kg_train[['voter_id', 'item_id', 'timestamp']].rename(columns={'voter_id': 'user_id'})],
                   ignore_index=True).drop_duplicates(ignore_index=True).sort_values('timestamp', ignore_index=True)
    cf = cf[['user_id', 'item_id']].groupby('user_id').agg({'item_id': list}).reset_index()
    cf = dict(zip(cf['user_id'], cf['item_id']))

    with open('./train.txt', 'w', encoding='utf-8') as file:
        no_inter_user = 0
        for user in user_remap['remap_user_id']:
            if user not in cf.keys():
                no_inter_user += 1
                continue
            if len(cf[user]) < 5:
                continue
            inters = [str(x) for x in cf[user]]

            if DATASET_VERSION == 3:
                kg_test.append(f"{user} {inters[-1]}\n")
                file.write(f"{user} {' '.join(inters[:-1])}\n")
            else:
                file.write(f"{user} {' '.join(inters)}\n")
        print(f'{no_inter_user} users do not have interactions.')

    with open('./test.txt', 'w', encoding='utf-8') as file:
        if DATASET_VERSION == 3:
            file.writelines(kg_test)
        else:
            kg_test = kg_test[['item_id', 'voter_id']].drop_duplicates(ignore_index=True)
            for _, row in kg_test.iterrows():
                file.write(f"{row['item_id']} {row['voter_id']}\n")
    print('Train&Test files done.')