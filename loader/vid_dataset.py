import os.path
import pickle

import pandas as pd
from keras.utils import pad_sequences
from torch.utils.data import Dataset
from tqdm import tqdm


class BaseDataset:
    def __init__(self, args):
        self.args = args
        self.data_path = os.path.join(args.data_path, args.dataset)
        user_remap = pd.read_csv(os.path.join(self.data_path, 'remap_user_id.csv'))
        item_remap = pd.read_csv(os.path.join(self.data_path, 'remap_item_id.csv'))
        self.feature_nunique = {'user_id': user_remap['user_id'].max() + 1,
                                'item_id': item_remap['item_id'].max() + 1}
        self.max_seq_len = args.max_seq_len
        self.gen_dataset()

    def gen_dataset(self):
        """
        Construct the training and test sets

        """
        train_output_path = os.path.join(self.data_path, f'vid-train-len{self.args.max_seq_len}.pkl')
        test_output_path = os.path.join(self.data_path, f'vid-test-len{self.args.max_seq_len}.pkl')
        recall_output_path = os.path.join(self.data_path, 'vid-recall.pkl')

        def create_train_sample(invite, share, vote, item_seq, label):
            item_seq = [] if item_seq == '' else item_seq.split(' ')
            item_seq_len = len(item_seq)
            item_seq = pad_sequences([item_seq], maxlen=self.max_seq_len, padding='post', truncating='post', value=0)
            item_seq = item_seq[0]
            # [Inviter ID, shared item ID, voter ID, historical item sequence length of voter, historical item sequence of voter, tag]
            return [invite, share, vote, item_seq_len, item_seq, float(label)]

        def to_list(x):
            if pd.isna(x):
                return []
            else:
                return [int(y) for y in x.strip().split(' ')]

        def create_test_sample(invite, share, votes: list, item_seqs: list, label):
            item_seq_lens = [len(x) for x in item_seqs]
            item_seqs = pad_sequences(item_seqs, maxlen=self.max_seq_len, padding='post', truncating='post', value=0)
            # [Inviter ID, shared item ID, recall returner list, historical item sequence length of voter, historical item sequence of voter, target voter]
            return [[invite] * len(votes), [share] * len(votes), votes, item_seq_lens, item_seqs, label]

        # ===================> Loading the training set
        if os.path.exists(train_output_path):
            self.train_dataset = pickle.load(open(train_output_path, 'rb'))
            print('train dataset loaded.')
        else:
            train_inter = pd.read_csv(os.path.join(self.data_path, 'vid-train.inter'), header=0, sep=',',
                                      names=['inviter_id', 'share_item', 'voter_id', 'item_seq', 'label'])
            self.train_dataset = []
            for _, row in tqdm(train_inter.iterrows(), total=len(train_inter)):
                sample = create_train_sample(row['inviter_id'], row['share_item'],
                                             row['voter_id'], row['item_seq'], row['label'])
                self.train_dataset.append(sample)

            pickle.dump(self.train_dataset, open(train_output_path, 'wb'))
            print('train dataset constructed.')

        # ===================> Loading the recall set
        if os.path.exists(recall_output_path):
            recall_dataset = pickle.load(open(recall_output_path, 'rb'))
            print('recall dataset loaded.')
        else:
            recall_inter = pd.read_csv(os.path.join(self.data_path, 'id-recall.inter'), header=0, sep=',',
                                       names=['id', 'recall_voter', 'recall_item_seq'])
            recall_inter['recall_item_seq'] = recall_inter['recall_item_seq'].apply(to_list)
            recall_inter = recall_inter.groupby('id') \
                .agg({'recall_voter': list, 'recall_item_seq': list}).reset_index()
            recall_voters = dict(zip(recall_inter['id'], recall_inter['recall_voter']))
            recall_item_seqs = dict(zip(recall_inter['id'], recall_inter['recall_item_seq']))
            recall_dataset = {'recall_voters': recall_voters, 'recall_item_seqs': recall_item_seqs}

            pickle.dump(recall_dataset, open(recall_output_path, 'wb'))
            print('recall dataset constructed.')

        # ===================> Loading the test set
        if os.path.exists(test_output_path):
            self.test_dataset = pickle.load(open(test_output_path, 'rb'))
            print('test dataset loaded.')
        else:
            test_inter = pd.read_csv(os.path.join(self.data_path, 'vid-test.inter'), header=0, sep=',',
                                     names=['id', 'inviter_id', 'share_item', 'voter_id', 'timestamp'])
            self.test_dataset = []
            for _, row in tqdm(test_inter.iterrows(), total=len(test_inter)):
                sample = create_test_sample(row['inviter_id'], row['share_item'],
                                            recall_dataset['recall_voters'][row['id']],
                                            recall_dataset['recall_item_seqs'][row['id']],
                                            row['voter_id'])
                self.test_dataset.append(sample)

            pickle.dump(self.test_dataset, open(test_output_path, 'wb'))
            print('test dataset constructed.')


class TrainDataset(Dataset):
    def __init__(self, dataset):
        self.train_dataset = dataset

    def __getitem__(self, index):
        train_sample = self.train_dataset[index]
        # [Inviter ID, shared item ID, voter ID, voter's historical item sequence length, voter's historical item sequence], tag
        return train_sample[:-1], train_sample[-1]

    def __len__(self):
        return len(self.train_dataset)


class TestDataset(Dataset):
    def __init__(self, dataset):
        self.test_dataset = dataset

    def __getitem__(self, index):
        test_sample = self.test_dataset[index]
        # [Inviter ID, shared item ID, recall voter list, historical item sequence length of voter, historical item sequence of voter], target voter
        return test_sample[:-1], test_sample[-1]

    def __len__(self):
        return len(self.test_dataset)
