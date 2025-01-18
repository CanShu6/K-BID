import logging
import os
import random
import sys
from time import time, sleep

import torch
import pandas as pd
import torch.optim as optim
import wandb
from tqdm import tqdm
import numpy as np

from data_loader.loader_cke import DataLoaderCKE
from model import CKE
from parsers.parser_cke import parse_cke_args
from utils.log_helper import create_log_id, logging_config
from utils.metrics import calc_metrics_at_k
from utils.model_helper import early_stopping, save_model, load_model, check_mem, occupy_mem

def evaluate(model, dataloader, Ks, device):
    test_batch_size = dataloader.test_batch_size
    test_user_dict = dataloader.test_user_dict
    train_user_dict = dataloader.train_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    metric_names = ['precision', 'mrr', 'auc']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)

            with torch.no_grad():
                batch_scores = model(batch_user_ids, item_ids, is_train=False)       # (n_batch_users, n_items)

            batch_metrics = calc_metrics_at_k(batch_scores.cpu(), train_user_dict, test_user_dict,
                                              batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks)

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
    data = DataLoaderCKE(args, logging)
    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    # construct model & optimizer
    model = CKE(args, data.n_users, data.n_items, data.n_entities, data.n_relations, user_pre_embed, item_pre_embed)
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
    for epoch in range(args.n_epoch):
        model.train()

        # train kg & cf
        time1 = time()
        total_loss = 0
        n_batch = data.n_cf_train // data.cf_batch_size + 1

        for iter in tqdm(range(1, n_batch + 1), desc='Training'):
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(data.kg_dict, data.kg_batch_size, data.n_entities)

            cf_batch_user = cf_batch_user.to(device)
            cf_batch_pos_item = cf_batch_pos_item.to(device)
            cf_batch_neg_item = cf_batch_neg_item.to(device)

            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            batch_loss = model(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail, is_train=True)

            if np.isnan(batch_loss.cpu().detach().numpy()):
                logging.info('ERROR: Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_batch))
                sys.exit()

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss.item()

        # log
        logging.info('KG & CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Loss {:.4f}'\
                     .format(epoch, n_batch, time() - time1, total_loss))
        wandb.log({'total_loss': total_loss})

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
            best_mrr, should_stop = early_stopping(metrics_list[k_min]['mrr'], args.stopping_steps)

            if should_stop:
                break

            if metrics_list[k_min]['mrr'].index(best_mrr) == len(epoch_list) - 1:
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
    args = parse_cke_args()
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
        project='CKE-vid-item',
        name=f'lr{args.lr}_dim{args.embed_dim}',
        config=args
    )
    train(args)
