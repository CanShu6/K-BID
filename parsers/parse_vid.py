import argparse


def parse_vid_args():
    parser = argparse.ArgumentParser(description='Run DIN.')

    parser.add_argument('--mode',               type=int,   default=0,
                        choices=[0, 1, 2],      help='0: train, 1: predict, 2: save')

    parser.add_argument('--enable_user_kg',     type=int,   default=0,
                        choices=[0, 1],         help='0: disable, 1: enable')
    parser.add_argument('--enable_item_kg',     type=int,   default=0,
                        choices=[0, 1],         help='0: disable, 1: enable')
    parser.add_argument('--pretrain_path',      type=str,   default='./pretrain')
    parser.add_argument('--save_emb_dir',       type=str,   default='data/embeddings/')
    parser.add_argument('--save_model_dir',     type=str,   default='trained_model/model.pth')

    parser.add_argument('--seed',               type=int,   default=2023)
    parser.add_argument('--gpu',                type=int,   default=0)
    parser.add_argument('--model',              type=str,   default='VID')
    parser.add_argument('--dataset',            type=str,   default='vid')
    parser.add_argument('--data_path',          type=str,   default='./data')

    parser.add_argument('--lr',                 type=float, default=0.01)
    parser.add_argument('--n_epoch',            type=int,   default=1000)
    parser.add_argument('--emb_dim',            type=int,   default=64)
    parser.add_argument('--batch_size',         type=int,   default=1024)
    parser.add_argument('--mlp_hidden_size',    type=str,   default='[256, 256]')
    parser.add_argument('--dropout_prob',       type=float, default=0.)
    parser.add_argument('--max_seq_len',        type=int,   default=20)
    parser.add_argument('--use_short_pref',     type=int,   default=1)
    parser.add_argument('--use_self_attention', type=int,   default=1)
    parser.add_argument('--head_num',           type=int,   default=4)

    parser.add_argument('--log_batch_every',    type=int,   default=10)
    parser.add_argument('--evaluate_every',     type=int,   default=10)
    parser.add_argument('--stopping_steps',     type=int,   default=10)
    parser.add_argument('--Ks',                 type=str,   default='[5,10,20]')

    args = parser.parse_args()

    save_dir = 'trained_model/VID/{}/emb-dim{}_lr{}_user-kg{}_item-kg{}_self-att{}_mlp{}_short-pref{}_len{}/'.format(
        args.dataset, args.emb_dim, args.lr, args.enable_user_kg, args.enable_item_kg, args.use_self_attention,
        '-'.join([str(x) for x in eval(args.mlp_hidden_size)]), args.use_short_pref, args.max_seq_len)
    args.save_dir = save_dir

    return args
