import collections
import os
import pickle

import pandas as pd
from tqdm import tqdm

from loader_base import DataLoaderBase
class DataLoaderCKE(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args)

        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size

        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data)
        self.print_info(logging)


    def construct_data(self, kg_data):
        # add inverse kg data
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations
        self.kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)

        self.n_relations = max(self.kg_data['r']) + 1
        self.n_entities = max(max(self.kg_data['h']), max(kg_data['t'])) + 1
        self.n_kg_data = len(self.kg_data)

        # construct kg dict
        kg_dict_path = os.path.join(self.data_dir, 'cke_kg_dict.pkl')
        if os.path.exists(kg_dict_path):
            kg_dict = pickle.load(open(kg_dict_path, 'rb'))
            self.kg_dict = kg_dict['kg_dict']
            self.relation_dict = kg_dict['relation_dict']
            print('kg data loaded.')
        else:
            self.kg_dict = collections.defaultdict(list)
            self.relation_dict = collections.defaultdict(list)
            for row in tqdm(self.kg_data.iterrows(), total=len(self.kg_data)):
                h, r, t = row[1]
                self.kg_dict[h].append((t, r))
                self.relation_dict[r].append((h, t))

            kg_dict = {
                'kg_dict': self.kg_dict, 'relation_dict': self.relation_dict
            }
            pickle.dump(kg_dict, open(kg_dict_path, 'wb'))
            print('kg data constructed.')


    def print_info(self, logging):
        logging.info('n_users:      %d' % self.n_users)
        logging.info('n_items:      %d' % self.n_items)
        logging.info('n_entities:   %d' % self.n_entities)
        logging.info('n_relations:  %d' % self.n_relations)

        logging.info('n_cf_train:   %d' % self.n_cf_train)
        logging.info('n_cf_test:    %d' % self.n_cf_test)

        logging.info('n_kg_data:    %d' % self.n_kg_data)


