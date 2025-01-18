import os
import random

import numpy as np
import torch


def init_seed(seed, reproducibility=True):
    r"""init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def early_stopping(mrr_list, stopping_steps):
    best_mrr = max(mrr_list)
    best_step = mrr_list.index(best_mrr)
    if len(mrr_list) - best_step - 1 >= stopping_steps:
        should_stop = True
    else:
        should_stop = False
    return best_mrr, should_stop


def save_model(model, model_dir, current_epoch, last_best_epoch=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
        if os.path.exists(old_model_state_file):
            os.system('rm {}'.format(old_model_state_file))


def load_model(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
