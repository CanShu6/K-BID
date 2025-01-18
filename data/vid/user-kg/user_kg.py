import pandas as pd


def create_triples(h, r, t):
    triples = pd.DataFrame(columns=['h', 'r', 't'])
    triples['h'] = h
    triples['t'] = t
    triples['r'] = r
    return triples


def remap(raw_dfs, remap_df, raw_field='raw_entity_id', remap_field='remap_entity_id'):
    for raw_df in raw_dfs:
        for column in raw_df.columns:
            raw_df[column] = raw_df.merge(right=remap_df, left_on=column, right_on=raw_field,
                                          how='left')[remap_field]


if __name__ == '__main__':
    DATASET_VERSION = 2

    # 训练集
    tianchi_train = pd.read_csv('../item_share_train_info_sorted.csv', sep=',', header=0,
                                names=['inviter_id', 'item_id', 'voter_id', 'timestamp'])
    # 测试集
    tianchi_test = pd.read_csv('../item_share_test_info_sorted.csv', sep=',', header=0,
                               names=['inviter_id', 'item_id', 'voter_id', 'timestamp'])

    if DATASET_VERSION == 1:
        kg_train = tianchi_train.iloc[:int(len(tianchi_train) * 0.8)]
        kg_test = tianchi_train[~tianchi_train.index.isin(kg_train.index)]
    else:
        kg_train = tianchi_train
        kg_test = tianchi_test

    # 用户信息
    user_info = pd.read_csv('../user_info.csv', sep=',', header=0,
                            names=['user_id', 'user_gender', 'user_age', 'user_level'])
    user_info['user_gender'] = user_info['user_gender'].apply(lambda x: 'gender_' + str(x))
    user_info['user_age'] = user_info['user_age'].apply(lambda x: 'age_' + str(x))
    user_info['user_level'] = user_info['user_level'].apply(lambda x: 'level_' + str(x))
    # 物品信息
    item_info = pd.read_csv('../item_info.csv', sep=',', header=0,
                            names=['item_id', 'cate_id', 'cate_level1_id', 'brand_id', 'shop_id'])

    # ========> 创建知识图谱的实体重映射文件
    entity_remap = pd.DataFrame()
    entity_remap['raw_entity_id'] = pd.concat([user_info['user_id'],
                                               item_info['item_id'],
                                               user_info['user_gender'].drop_duplicates(),
                                               user_info['user_age'].drop_duplicates(),
                                               user_info['user_level'].drop_duplicates()], ignore_index=True)
    entity_remap.reset_index(names='remap_entity_id', inplace=True)
    entity_remap.to_csv('./entity_remap.csv', sep=',', index=False)
    print('KG remap file done.')

    remap([kg_train, kg_test, user_info], entity_remap)

    # ========> 创建知识图谱的三元组文件
    # 邀请者-物品交互
    share = kg_train[['inviter_id', 'item_id']].drop_duplicates(ignore_index=True)
    # 回流者-物品交互
    click = kg_train[['voter_id', 'item_id']].drop_duplicates(ignore_index=True)
    # 用户-用户交互
    social = kg_train[['inviter_id', 'voter_id']].drop_duplicates(ignore_index=True)

    kg_final = pd.DataFrame(columns=['h', 'r', 't'])
    kg_final = pd.concat([kg_final,
                          create_triples(share['inviter_id'], 0, share['item_id']),
                          create_triples(click['voter_id'], 1, click['item_id']),
                          create_triples(social['inviter_id'], 2, social['voter_id']),
                          create_triples(user_info['user_id'], 3, user_info['user_gender']),
                          create_triples(user_info['user_id'], 4, user_info['user_age']),
                          create_triples(user_info['user_id'], 5, user_info['user_level'])],
                         axis=0, ignore_index=True)
    kg_final.to_csv('./kg_final.txt', sep=' ', index=False, header=False)
    print('KG triple file done.')

    # ========> 创建训练和测试文件
    social = social.groupby('inviter_id').agg({'voter_id': list}).reset_index()
    social = dict(zip(social['inviter_id'], social['voter_id']))

    with open('./train.txt', 'w', encoding='utf-8') as file:
        no_inter_user = 0
        for user in user_info['user_id']:
            if user not in social.keys():
                no_inter_user += 1
                continue
            # 5 core setting
            # if len(social[user]) < 5:
            #     continue
            inters = [str(x) for x in social[user]]
            file.write(f"{user} {' '.join(inters)}\n")
        print(f'{no_inter_user} users do not have interactions.')

    with open('./test.txt', 'w', encoding='utf-8') as file:
        kg_test = kg_test[['inviter_id', 'voter_id']].drop_duplicates(ignore_index=True)
        for _, row in kg_test.iterrows():
            file.write(f"{row['inviter_id']} {row['voter_id']}\n")
    print('Train&Test files done.')
