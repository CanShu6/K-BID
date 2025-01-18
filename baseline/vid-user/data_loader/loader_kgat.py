import collections
import os.path
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from tqdm import tqdm

from loader_base import DataLoaderBase


class DataLoaderKGAT(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args)
        self.args = args
        self.social_batch_size = args.social_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size

        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data)
        self.print_info(logging)

        self.laplacian_type = args.laplacian_type
        self.create_adjacency_dict()
        self.create_laplacian_dict()

    def construct_data(self, kg_data):
        # add inverse kg data
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations
        self.kg_train_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)

        self.n_relations = max(self.kg_train_data['r']) + 1
        self.n_entities = max(max(self.kg_train_data['h']), max(self.kg_train_data['t'])) + 1
        self.n_kg_train = len(self.kg_train_data)

        # construct kg dict
        kg_dict_path = os.path.join(self.data_dir, 'kg_dict.pkl')
        if os.path.exists(kg_dict_path):
            kg_dict = pickle.load(open(kg_dict_path, 'rb'))
            self.h_list = torch.LongTensor(kg_dict['h_list'])
            self.t_list = torch.LongTensor(kg_dict['t_list'])
            self.r_list = torch.LongTensor(kg_dict['r_list'])
            self.train_kg_dict = kg_dict['train_kg_dict']
            self.train_relation_dict = kg_dict['train_relation_dict']
            print('kg data loaded.')
        else:
            h_list = []
            t_list = []
            r_list = []

            self.train_kg_dict = collections.defaultdict(list)
            self.train_relation_dict = collections.defaultdict(list)

            for row in tqdm(self.kg_train_data.iterrows(), total=len(self.kg_train_data)):
                h, r, t = row[1]
                h_list.append(h)
                t_list.append(t)
                r_list.append(r)

                self.train_kg_dict[h].append((t, r))
                self.train_relation_dict[r].append((h, t))

            self.h_list = torch.LongTensor(h_list)
            self.t_list = torch.LongTensor(t_list)
            self.r_list = torch.LongTensor(r_list)

            kg_dict = {
                'h_list': h_list, 't_list': t_list, 'r_list': r_list,
                'train_kg_dict': self.train_kg_dict, 'train_relation_dict': self.train_relation_dict
            }
            pickle.dump(kg_dict, open(kg_dict_path, 'wb'))
            print('kg data constructed.')

    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [e[0] for e in ht_list]
            cols = [e[1] for e in ht_list]
            vals = [1] * len(rows)
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_entities, self.n_entities))
            self.adjacency_dict[r] = adj

    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == 'symmetric':
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            self.laplacian_dict[r] = norm_lap_func(adj)

        A_in = sum(self.laplacian_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo())

    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def print_info(self, logging):
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_relations:       %d' % self.n_relations)

        logging.info('n_h_list:          %d' % len(self.h_list))
        logging.info('n_t_list:          %d' % len(self.t_list))
        logging.info('n_r_list:          %d' % len(self.r_list))

        logging.info('n_social_train:        %d' % self.n_social_train)
        logging.info('n_social_test:         %d' % self.n_social_test)

        logging.info('n_kg_train:        %d' % self.n_kg_train)


