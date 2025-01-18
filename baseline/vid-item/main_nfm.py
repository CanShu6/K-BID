import logging
import random
import sys
from time import time, sleep

import numpy as np
import torch
import pandas as pd
import scipy.sparse as sp
import torch.multiprocessing as mp
import torch.optim as optim
import wandb
from tqdm import tqdm

from data_loader.loader_nfm import DataLoaderNFM
from parsers.parser_nfm import parse_nfm_args
from model import NFM
from utils.log_helper import create_log_id, logging_config
from utils.metrics import calc_metrics_at_k
from utils.model_helper import early_stopping, save_model, load_model, check_mem, occupy_mem

def evaluate(model, dataloader, Ks, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]

    n_users = len(user_ids)
    n_items = dataloader.n_items
    item_ids = list(range(n_items))

    metric_names = ['precision', 'mrr', 'auc']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            feature_values = dataloader.generate_test_batch(batch_user_ids)
            feature_values = feature_values.to(device)

            with torch.no_grad():
                batch_scores = model(feature_values, is_train=False)            # (batch_size)

            user_idx_map = dict(zip(batch_user_ids, range(n_users)))
            rows = [user_idx_map[u] for u in np.repeat(batch_user_ids, n_items).tolist()]
            cols = item_ids * len(batch_user_ids)
            batch_scores = batch_scores.cpu()
            cf_score_matrix = torch.Tensor(sp.coo_matrix((batch_scores, (rows, cols)),
                                                         shape=(len(batch_user_ids), n_items)).todense())

            batch_metrics = calc_metrics_at_k(cf_score_matrix, train_user_dict, test_user_dict,
                                              batch_user_ids, item_ids, Ks)
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)

    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.mean(metrics_dict[k][m])
    return metrics_dict


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # load data
    data = DataLoaderNFM(args, logging)
    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    # construct model & optimizer
    model = NFM(args, data.n_users, data.n_items, data.n_entities, user_pre_embed, item_pre_embed)
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # initialize metrics
    best_epoch = -1

    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    epoch_list = []
    metrics_name = ['precision', 'mrr', 'auc']
    metrics_list = {k: {m: [] for m in metrics_name} for k in Ks}

    # train model
    for epoch in range(0, args.n_epoch):
        model.train()

        # train cf
        time1 = time()
        total_loss = 0
        n_batch = data.n_cf_train // data.train_batch_size + 1

        for iter in tqdm(range(1, n_batch + 1), desc='CF Training'):
            pos_feature_values, neg_feature_values = data.generate_train_batch(data.train_user_dict)
            pos_feature_values = pos_feature_values.to(device)
            neg_feature_values = neg_feature_values.to(device)
            batch_loss = model(pos_feature_values, neg_feature_values, is_train=True)

            if np.isnan(batch_loss.cpu().detach().numpy()):
                logging.info('ERROR: Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_batch))
                sys.exit()

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss.item()

        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(
            epoch, n_batch, time() - time1, total_loss))
        wandb.log({'total loss': total_loss})

        # evaluate cf
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time3 = time()
            metrics_dict = evaluate(model, data, Ks, device)
            logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | '
                         'Precision [{:.4f}, {:.4f}], MRR [{:.4f}, {:.4f}], AUC [{:.4f}, {:.4f}]' \
                         .format(epoch, time() - time3,
                                 metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'],
                                 metrics_dict[k_min]['mrr'], metrics_dict[k_max]['mrr'],
                                 metrics_dict[k_min]['auc'], metrics_dict[k_max]['auc']))

            epoch_list.append(epoch)
            for k in Ks:
                for m in metrics_name:
                    metrics_list[k][m].append(metrics_dict[k][m])
                    wandb.log({'{}@{}'.format(m, k): metrics_dict[k][m]})
            best_recall, should_stop = early_stopping(metrics_list[k_min]['mrr'], args.stopping_steps)

            if should_stop:
                break

            if metrics_list[k_min]['mrr'].index(best_recall) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

    # save metrics
    metrics_df = [epoch_list]
    metrics_cols = ['epoch_idx']
    for k in Ks:
        for m in metrics_name:
            metrics_df.append(metrics_list[k][m])
            metrics_cols.append('{}@{}'.format(m, k))
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)

    # print best metrics
    best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
    logging.info('Best CF Evaluation: Epoch {:04d} | '
                 'Precision [{:.4f}, {:.4f}], MRR [{:.4f}, {:.4f}], AUC [{:.4f}, {:.4f}]'
                 .format(int(best_metrics['epoch_idx']),
                         best_metrics['precision@{}'.format(k_min)], best_metrics['precision@{}'.format(k_max)],
                         best_metrics['mrr@{}'.format(k_min)], best_metrics['mrr@{}'.format(k_max)],
                         best_metrics['auc@{}'.format(k_min)], best_metrics['auc@{}'.format(k_max)]))



if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    args = parse_nfm_args()

    if args.use_polling == 1:
        is_occupied = False
        cuda_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        while not is_occupied:
            for cuda in cuda_list:
                total_mem, used_mem = check_mem(cuda)
                if int(total_mem) - int(used_mem) >= 15000:
                    args.cuda = cuda
                    is_occupied = True
                    break
            if not is_occupied:
                sleep(5)
    occupy_mem(args.cuda, args.occupy_mem_ratio)
    print(f'cuda:{args.cuda} is occupied!')

    wandb.init(
        project='NFM-vid-item',
        name=f'lr{args.lr}_dim{args.embed_dim}',
        config=args
    )

    train(args)
