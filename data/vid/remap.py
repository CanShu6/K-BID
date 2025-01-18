import pandas as pd

if __name__ == '__main__':
    user_df = pd.read_csv('user_info.csv', header=0, sep=',',
                          names=['raw_user_id', 'raw_user_gender', 'raw_user_age', 'raw_user_level'])
    user_df.reset_index(inplace=True)
    user_df.rename(columns={'index': 'user_id'}, inplace=True)
    user_df['user_id'] += 1
    user_df[['raw_user_id', 'user_id']].sort_values('user_id').to_csv('./remap_user_id.csv', index=False)

    item_df = pd.read_csv('item_info.csv', header=0, sep=',',
                          names=['raw_item_id', 'raw_cate_id', 'raw_cate_level1_id', 'raw_brand_id', 'raw_shop_id'])
    item_df.reset_index(inplace=True)
    item_df.rename(columns={'index': 'item_id'}, inplace=True)
    item_df['item_id'] += 1
    item_df[['raw_item_id', 'item_id']].sort_values('item_id').to_csv('./remap_item_id.csv', index=False)
