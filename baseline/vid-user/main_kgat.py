import logging
import os
import random
import sys
from time import time

import pandas as pd
import torch.optim as optim
import wandb
from tqdm import tqdm
import torch
import numpy as np

from parsers.parser_kgat import parse_kgat_args
from utils.log_helper import create_log_id, logging_config
from utils.metrics import calc_metrics_at_k
from utils.model_helper import early_stopping, save_model, load_model
from data_loader.loader_kgat import DataLoaderKGAT
from model import KGAT


def evaluate(model, dataloader, Ks, device):
    test_batch_size = dataloader.test_batch_size
    inviters, label_voters = dataloader.social_test_data

    model.eval()

    inviter_batches = [inviters[i: i + test_batch_size] for i in range(0, len(inviters), test_batch_size)]
    label_voter_batches = [label_voters[i: i + test_batch_size] for i in range(0, len(label_voters), test_batch_size)]
    inviter_batches = [torch.LongTensor(d) for d in inviter_batches]

    n_users = dataloader.n_users
    user_ids = torch.arange(n_users, dtype=torch.long).to(device)

    metric_names = ['precision', 'mrr', 'auc']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(inviter_batches), desc='Evaluating Iteration') as pbar:
        for batch_id, batch_inviters in enumerate(inviter_batches):
            batch_inviters = batch_inviters.to(device)

            with torch.no_grad():
                batch_scores = model(batch_inviters, user_ids, mode='predict')       # (n_batch_users, n_items)

            batch_scores = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(batch_scores, label_voter_batches[batch_id],
                                              batch_inviters.cpu().numpy(), user_ids.cpu().numpy(), Ks)

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
    data = DataLoaderKGAT(args, logging)
    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    # construct model & optimizer
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations, data.A_in, user_pre_embed, item_pre_embed)
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)
    logging.info(model)

    social_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
        time0 = time()
        model.train()

        # train social
        social_total_loss = 0
        n_social_batch = data.n_social_train // data.social_batch_size + 1

        for iter in tqdm(range(1, n_social_batch + 1), desc='CF Training'):
            batch_inviters, batch_pos_voters, batch_neg_voters = data.generate_social_batch(data.train_inviter_dict, data.social_batch_size)
            batch_inviters = batch_inviters.to(device)
            batch_pos_voters = batch_pos_voters.to(device)
            batch_neg_voters = batch_neg_voters.to(device)

            social_batch_loss = model(batch_inviters, batch_pos_voters, batch_neg_voters, mode='train_social')

            if np.isnan(social_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (Social Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_social_batch))
                sys.exit()

            social_batch_loss.backward()
            social_optimizer.step()
            social_optimizer.zero_grad()
            social_total_loss += social_batch_loss.item()

        # train kg
        kg_total_loss = 0
        n_kg_batch = data.n_kg_train // data.kg_batch_size + 1

        for iter in tqdm(range(1, n_kg_batch + 1), desc='KG Training'):
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(data.train_kg_dict, data.kg_batch_size, data.n_entities)
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            kg_batch_loss = model(kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail, mode='train_kg')

            if np.isnan(kg_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (KG Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_kg_batch))
                sys.exit()

            kg_batch_loss.backward()
            kg_optimizer.step()
            kg_optimizer.zero_grad()
            kg_total_loss += kg_batch_loss.item()

        # update attention
        h_list = data.h_list.to(device)
        t_list = data.t_list.to(device)
        r_list = data.r_list.to(device)
        relations = list(data.laplacian_dict.keys())
        model(h_list, t_list, r_list, relations, mode='update_att')

        logging.info('Social + KG Training: Epoch {:04d} | Total Time {:.1f}s | Total Loss {:.2f}'\
                     .format(epoch, time() - time0, social_total_loss + kg_total_loss))
        wandb.log({'social_loss': social_total_loss, 'kg_loss': kg_total_loss,
                   'total_loss': social_total_loss + kg_total_loss})

        # evaluate
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch - 1:
            time6 = time()
            metrics_dict = evaluate(model, data, Ks, device)
            logging.info('Social Evaluation: Epoch {:04d} | Total Time {:.1f}s | '
                         'Precision [{:.4f}, {:.4f}], MRR [{:.4f}, {:.4f}], AUC [{:.4f}, {:.4f}]'\
                         .format(epoch, time() - time6,
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
    logging.info('Best Social Evaluation: Epoch {:04d} | '
                 'Precision [{:.4f}, {:.4f}], MRR [{:.4f}, {:.4f}], AUC [{:.4f}, {:.4f}]'
                 .format(int(best_metrics['epoch_idx']),
                         best_metrics['precision@{}'.format(k_min)], best_metrics['precision@{}'.format(k_max)],
                         best_metrics['mrr@{}'.format(k_min)], best_metrics['mrr@{}'.format(k_max)],
                         best_metrics['auc@{}'.format(k_min)], best_metrics['auc@{}'.format(k_max)]))


def predict(args):
    # GPU / CPU
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # load data
    data = DataLoaderKGAT(args, logging)

    # load model
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    # predict
    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    metrics_dict = evaluate(model, data, Ks, device)
    metrics_cols = []
    metrics_list = []
    metrics_name = ['precision', 'mrr', 'auc']
    for k in Ks:
        for m in metrics_name:
            metrics_cols.append('{}@{}'.format(m, k))
            metrics_list.append(metrics_dict[k][m])
    pd.DataFrame({'metrics': metrics_cols, 'value': metrics_list})\
        .to_csv(os.path.join(args.save_dir, 'final_scores.tsv'), index=False, sep='\t')

    print('CF Evaluation: Precision [{:.4f}, {:.4f}], MRR [{:.4f}, {:.4f}], AUC [{:.4f}, {:.4f}]'.format(
        metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'],
        metrics_dict[k_min]['mrr'], metrics_dict[k_max]['mrr'],
        metrics_dict[k_min]['auc'], metrics_dict[k_max]['auc']))


def save(args):
    # load data
    data = DataLoaderKGAT(args, logging)

    # load model
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations)
    model = load_model(model, args.pretrain_model_path)

    # export embeddings
    user_embeddings = model.entity_embed.weight[:data.n_users, :]
    item_embeddings = model.entity_embed.weight[data.n_users:data.n_items, :]

    folder = os.path.join(args.pretrain_embedding_dir, args.data_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.save(os.path.join(folder, 'user_embed.npy'), user_embeddings.detach().numpy())
    np.save(os.path.join(folder, 'item_embed.npy'), item_embeddings.detach().numpy())
    print('embeddings saved.')


if __name__ == '__main__':
    args = parse_kgat_args()
    if args.mode == 0:
        wandb.init(
            project='vid-user',
            name=f'lr{args.lr}_dim{args.embed_dim}',
            config=args
        )
        train(args)
    elif args.mode == 1:
        predict(args)
    elif args.mode == 2:
        save(args)


