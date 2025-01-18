import logging
import os
import sys
from time import time

import numpy as np
import pandas as pd
import torch
import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from loader.vid_dataset import BaseDataset, TrainDataset, TestDataset
from model.vid import VID
from parsers.parse_vid import parse_vid_args
from utils.basic import create_log_id, logging_config
from utils.metrics import calc_metrics_at_k
from utils.model_helper import init_seed, early_stopping, save_model, load_model


def evaluate():
    metrics_dict = {k: {m: [] for m in metrics_name} for k in Ks}
    model.eval()
    for sample, label in tqdm(test_dl, total=len(test_dl)):
        inviter = torch.LongTensor(sample[0]).to(device)
        share_item = torch.LongTensor(sample[1]).to(device)
        recall_voters = torch.LongTensor(sample[2]).to(device)
        recall_item_lens = torch.LongTensor(sample[3]).to(device)
        recall_item_seqs = sample[4].squeeze().to(device)

        with torch.no_grad():
            scores = model.predict(inviter, share_item, recall_voters, recall_item_lens, recall_item_seqs)
        # Calculating metrics
        metrics_of_single_user = calc_metrics_at_k(int(label), scores.tolist(), recall_voters.tolist(), Ks)
        for k in Ks:
            for m in metrics_name:
                metrics_dict[k][m].append(metrics_of_single_user[k][m])
    # Calculate the mean of the metrics
    for k in Ks:
        for m in metrics_name:
            metrics_dict[k][m] = np.mean(metrics_dict[k][m])
    return metrics_dict


def train():
    best_epoch = -1
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.n_epoch):
        time0 = time()
        total_loss = 0

        # ===================> train
        model.train()
        for batch, (sample, label) in tqdm(enumerate(train_dl), total=len(train_dl)):
            sample = [x.to(device) for x in sample]
            label = label.to(device)
            loss = model(sample[0], sample[1], sample[2], sample[3], sample[4], label)

            if np.isnan(loss.cpu().detach().numpy()):
                logging.error('ERROR: Epoch {:04d} batch {:04d} Loss is nan.'.format(epoch, batch))
                sys.exit()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if batch % args.log_batch_every == 0 or batch == len(train_dl) - 1:
            #     wandb.log({'batch_loss': loss})
            total_loss += loss

        wandb.log({'epoch_loss': total_loss})
        logging.info('Training: Epoch {:04d} | Total Time {:.1f}s | Loss {:.2f}'.format(epoch, time() - time0, total_loss))

        # ===================> evaluate
        if epoch % args.evaluate_every == 0 or epoch == args.n_epoch - 1:
            time1 = time()
            metrics_dict = evaluate()
            logging.info('Evaluation: Epoch {:04d} | Total Time {:.1f}s | '
                         'Precision [{:.4f}, {:.4f}], MRR [{:.4f}, {:.4f}], AUC [{:.4f}, {:.4f}]'\
                         .format(epoch, time() - time1,
                                 metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'],
                                 metrics_dict[k_min]['mrr'], metrics_dict[k_max]['mrr'],
                                 metrics_dict[k_min]['auc'], metrics_dict[k_max]['auc']))
            epoch_list.append(epoch)
            for k in Ks:
                for m in metrics_name:
                    metrics_list[k][m].append(metrics_dict[k][m])
                    wandb.log({'{}@{}'.format(m, k): metrics_dict[k][m]})

            best_mrr, should_stop = early_stopping(metrics_list[k_min]['mrr'], args.stopping_steps)
            # trigger early stop
            if should_stop:
                break
            # Save the optimal model
            if metrics_list[k_min]['mrr'].index(best_mrr) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

    # ===================> save metrics
    metrics_df = [epoch_list]
    metrics_cols = ['epoch_idx']
    for k in Ks:
        for m in metrics_name:
            metrics_df.append(metrics_list[k][m])
            metrics_cols.append('{}@{}'.format(m, k))
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)

    # ===================> print metrics
    best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
    logging.info('Best Evaluation: Epoch {:04d} | '
                 'Precision [{:.4f}, {:.4f}], MRR [{:.4f}, {:.4f}], AUC [{:.4f}, {:.4f}]'\
                 .format(int(best_metrics['epoch_idx']),
                         best_metrics['precision@{}'.format(k_min)],
                         best_metrics['precision@{}'.format(k_max)],
                         best_metrics['mrr@{}'.format(k_min)],
                         best_metrics['mrr@{}'.format(k_max)],
                         best_metrics['auc@{}'.format(k_min)],
                         best_metrics['auc@{}'.format(k_max)]))


def predict():
    pass


def save():
    pretrained_model = load_model(model, args.save_model_dir, device)

    voter_embed = pretrained_model.voter_emb.weight

    folder = os.path.join(args.save_emb_dir, f'embed_dim{voter_embed.size()[-1]}')
    if not os.path.exists(folder):
        os.makedirs(folder)

    np.save(os.path.join(folder, 'voter_embed.npy'), voter_embed.cpu().detach().numpy())
    print('voter embeddings saved.')


if __name__ == '__main__':
    """
    训练: python main_vid.py --mode 0 --pretrain 1 --gpu 5 --evaluate_every 1
    """
    args = parse_vid_args()

    # seed
    init_seed(args.seed)
    # logging
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(sys.argv)
    logging.info(args)

    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    epoch_list = []
    metrics_name = ['precision', 'mrr', 'auc']
    metrics_list = {k: {m: [] for m in metrics_name} for k in Ks}

    # load data
    base_dataset = BaseDataset(args)

    # load model
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = VID(args, base_dataset.feature_nunique, device).to(device)
    logging.info(model)

    if args.mode == 0:
        wandb.init(
            project='VID',
            name=f'lr{args.lr}_'
                 f'dim{args.emb_dim}_'
                 f'user{args.enable_user_kg}_'
                 f'item{args.enable_item_kg}_'
                 f'len{args.max_seq_len}_'
                 f'self-att{args.use_self_attention}_'
                 f'mlp{max(eval(args.mlp_hidden_size))}_'
                 f'short-pref{args.use_short_pref}',
            config=args
        )

        train_dl = DataLoader(TrainDataset(base_dataset.train_dataset), batch_size=args.batch_size, shuffle=True)
        test_dl = DataLoader(TestDataset(base_dataset.test_dataset), batch_size=1, shuffle=True)

        # load pretrained embedding
        if args.enable_user_kg == 1:
            # user kg
            user_kg = {
                'user_embed': np.load(os.path.join(args.pretrain_path, f'user-kg/embed_dim{args.emb_dim}/user_embed.npy')),
                'item_embed': np.load(os.path.join(args.pretrain_path, f'user-kg/embed_dim{args.emb_dim}/item_embed.npy'))
            }
            user_emb = torch.from_numpy(user_kg['user_embed']).to(device)
            model.inviter_emb.weight.data[1:] = user_emb
            model.voter_emb.weight.data[1:] = user_emb
            print('user embeddings loaded.')
        if args.enable_item_kg == 1:
            # item kg
            item_kg = {
                'user_embed': np.load(os.path.join(args.pretrain_path, f'item-kg/embed_dim{args.emb_dim}/user_embed.npy')),
                'item_embed': np.load(os.path.join(args.pretrain_path, f'item-kg/embed_dim{args.emb_dim}/item_embed.npy'))
            }
            item_emb = torch.from_numpy(item_kg['item_embed']).to(device)
            model.item_emb.weight.data[1:] = item_emb
            user_emb = torch.from_numpy(item_kg['user_embed']).to(device)
            if args.enable_user_kg == 1:
                model.inviter_emb.weight.data[1:] = model.inviter_emb.weight.data[1:] * 0.8 + user_emb * 0.2
                model.voter_emb.weight.data[1:] = model.voter_emb.weight.data[1:] * 0.8 + user_emb * 0.2
            else:
                model.inviter_emb.weight.data[1:] = user_emb
                model.voter_emb.weight.data[1:] = user_emb
            print('item embeddings loaded.')
        # train
        train()
    elif args.mode == 1:
        predict()
    elif args.mode == 2:
        save()
