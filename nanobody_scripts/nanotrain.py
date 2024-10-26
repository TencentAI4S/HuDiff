import os.path
import pickle
import numpy as np
import argparse
import yaml
from easydict import EasyDict
import shutil
import sys
current_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(current_dir)

import torch
import torch.utils.tensorboard
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

from dataset.oas_unpair_dataset_new import OasHeavyMaskCollater
from torch.utils.data import DataLoader
from utils.train_utils import model_selected, optimizer_selected, scheduler_selected
from utils.misc import seed_all, get_new_log_dir, get_logger, inf_iterator, count_parameters
from utils.loss import MaskedAccuracy, OasMaskedHeavyCrossEntropyLoss
from utils.train_utils import get_dataset


def convert_multi_gpu_checkpoint_to_single_gpu(checkpoint):
    if 'module' in list(checkpoint['model'].keys())[0]:
        new_state_dict = {}
        for key, value in checkpoint['model'].items():
            new_key = key.replace('module.', '')  # Remove 'module.' prefix
            new_state_dict[new_key] = value
        checkpoint['model'] = new_state_dict
    return checkpoint['model']


def freeze_parameters(block):
    for x in block:
        x.requires_grad = False

def unfreeze_parameters(block):
    for x in block:
        x.requires_grad = True

def train(it):

    H_sum_loss, H_sum_nll = 0., 0.
    H_sum_cdr_loss = 0.
    sum_loss = 0
    H_sum_acc_loss = 0.
    sum_roc_auc = 0.

    model.train()
    for _ in range(config.train.batch_acc):
        optimizer.zero_grad()

        (H_src, H_tgt, H_region, chain_type,
         H_masks, H_cdr_masks,
         H_timesteps) = next(train_iterator)
        H_src, H_tgt = H_src.to(device), H_tgt.to(device)
        H_region = H_region.to(device)
        chain_type = chain_type.to(device)
        H_masks, H_cdr_masks = H_masks.to(device), H_cdr_masks.to(device)
        H_timesteps = H_timesteps.to(device)

        H_pred = model(H_src, H_region, chain_type)

        H_loss, H_nll, H_cdr_loss = cross_loss(
            H_pred,
            H_tgt,
            H_masks,
            H_cdr_masks,
            H_timesteps
        )
        if args.train_loss == 'fr':
            loss = H_loss
        elif args.train_loss == 'all':
            loss = H_loss + H_cdr_loss
        else:
            loss = None
            print("Please set correct train loss type!")

        loss.mean()
        loss.backward()
        optimizer.step()

        sum_loss += loss
        H_sum_loss += H_loss
        H_sum_nll += H_nll
        H_sum_cdr_loss += H_cdr_loss

        # Those value indicate whether the pred equl to tgt. max may is 1.
        # Not backward.
        H_acc_loss, roc_auc = mask_acc_loss(H_pred, H_tgt, H_masks)
        H_sum_acc_loss += H_acc_loss
        sum_roc_auc += roc_auc


    mean_loss = sum_loss / config.train.batch_acc
    mean_H_loss = H_sum_loss / config.train.batch_acc
    mean_H_nll = H_sum_nll / config.train.batch_acc

    mean_H_cdr_loss = H_sum_cdr_loss / config.train.batch_acc

    # Not backward.
    mean_H_acc_loss = H_sum_acc_loss / config.train.batch_acc
    mean_roc_auc = sum_roc_auc / config.train.batch_acc

    logger.info('Training iter {}, Loss is: {:.6f} | H_loss: {:.6f} | H_nll: {:.6f} '
                '| H_cdr_loss: {:.6f} | H_acc: {:.6f} | ROC_AUC: {:.6f}'.
                format(it, mean_loss, mean_H_loss, mean_H_nll, mean_H_cdr_loss, mean_H_acc_loss, mean_roc_auc))
    writer.add_scalar('train/loss', mean_loss, it)
    writer.add_scalar('train/H_loss', mean_H_loss, it)
    writer.add_scalar('train/H_nll', mean_H_nll, it)
    writer.add_scalar('train/H_cdr_loss', mean_H_cdr_loss, it)
    writer.add_scalar('train/H_acc', mean_H_acc_loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/roc_auc', mean_roc_auc, it)
    writer.flush()



def valid(it, valid_type):
    H_sum_loss, H_sum_nll = 0., 0.
    H_sum_cdr_loss = 0.
    H_sum_acc_loss = 0.
    sum_valid_loss = 0.
    sum_roc_auc = 0.
    model.eval()


    val_sum = len(val_loader)
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Val', total=len(val_loader)):
            (H_src, H_tgt, H_region, chain_type,
             H_masks, H_cdr_masks,
             H_timesteps) = batch
            H_src, H_tgt = H_src.to(device), H_tgt.to(device)
            H_region = H_region.to(device)
            chain_type = chain_type.to(device)
            H_masks, H_cdr_masks = H_masks.to(device), H_cdr_masks.to(device)
            H_timesteps = H_timesteps.to(device)

            H_pred = model(H_src, H_region, chain_type)

            H_loss, H_nll, H_cdr_loss = cross_loss(
                H_pred,
                H_tgt,
                H_masks,
                H_cdr_masks,
                H_timesteps
            )

            if args.train_loss == 'fr':
                loss = H_loss
            elif args.train_loss == 'all':
                loss = H_loss + H_cdr_loss
            else:
                loss = None
                print("Please set correct train loss type!")

            sum_valid_loss += loss

            H_sum_loss += H_loss
            H_sum_nll += H_nll
            H_sum_cdr_loss += H_cdr_loss

            # Those value indicate whether the pred equl to tgt. max may is 1.
            H_acc_loss, roc_auc = mask_acc_loss(H_pred, H_tgt, H_masks)

            # Not backward.
            H_sum_acc_loss += H_acc_loss
            sum_roc_auc += roc_auc

    mean_loss = sum_valid_loss / val_sum
    mean_H_loss = H_sum_loss / val_sum
    mean_H_nll = H_sum_nll / val_sum
    mean_H_cdr_loss = H_sum_cdr_loss / val_sum

    # Not backward.
    mean_H_acc_loss = H_sum_acc_loss / val_sum
    mean_roc_auc = sum_roc_auc / val_sum

    scheduler.step(mean_loss)

    logger.info('Validation iter {}, Loss is: {:.6f} | H_loss: {:.6f} | H_nll: {:.6f} '
                '| H_cdr_loss: {:.6f} | H_acc: {:.6f} | ROC_AUC: {:.6f}'.
                format(it, mean_loss, mean_H_loss, mean_H_nll,
                       mean_H_cdr_loss, mean_H_acc_loss, mean_roc_auc))
    writer.add_scalar('val/loss', mean_loss, it)
    writer.add_scalar('val/H_loss', mean_H_loss, it)
    writer.add_scalar('val/H_nll', mean_H_nll, it)
    writer.add_scalar('val/H_cdr_loss', mean_H_cdr_loss, it)
    writer.add_scalar('val/H_acc', mean_H_acc_loss, it)
    writer.add_scalar('val/roc_auc', mean_roc_auc, it)
    writer.flush()

    return mean_loss


if __name__ == '__main__':
    # Required args.
    parser = argparse.ArgumentParser()
    parser.add_argument('--unpair_data_path', type=str,
                        default=None
                        )
    parser.add_argument('--data_name', type=str,
                        default='heavy')
    parser.add_argument('--train_model', type=str,
                        default='heavy')
    parser.add_argument('--data_version', type=str,
                        default='test')
    parser.add_argument('--train_loss', type=str,
                        default='all', choices=['fr', 'all'])
    parser.add_argument('--config_path', type=str,
                        default=None
                        )
    parser.add_argument('--log_path', type=str,
                        default=None
                        )
    parser.add_argument('--resume', type=bool,
                        default=False)
    parser.add_argument('--checkpoint', type=str,
                        default=None)

    args = parser.parse_args()

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Config parameters.
    if not args.resume:
        with open(args.config_path, 'r') as f:
            config = EasyDict(yaml.safe_load(f))
    else:
        assert args.checkpoint != '', "Need Specified Checkpoint."
        ckpt_path = args.checkpoint
        ckpt = torch.load(ckpt_path, map_location='cpu')
        config = ckpt['config']

    version = f'{args.data_name}_{args.train_model}_{args.data_version}_{args.train_loss}'
    # Create Log dir.
    log_dir = get_new_log_dir(
        root=args.log_path,
        prefix=version
    )

    # Checkpoints dir.
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # logger and writer
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)

    logger.info(args)
    logger.info(config)


    # Copy files for checking.
    shutil.copyfile(args.config_path, os.path.join(log_dir, os.path.basename(args.config_path)))
    shutil.copyfile('./nanobody_scripts/nanotrain.py', os.path.join(log_dir, 'nanotrain.py'))
    shutil.copytree('./model', os.path.join(log_dir, 'model'))


    # Fixed
    seed_all(config.train.seed)

    # Create dataloader.
    h_subsets = get_dataset(args.unpair_data_path, args.data_name, args.data_version)
    train_dataset, val_dataset = h_subsets['train'], h_subsets['val']
    collater = OasHeavyMaskCollater()

    # Only consider Heavy.
    train_iterator = inf_iterator(DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        shuffle=True,
        collate_fn=collater
    ))
    logger.info(f'Training: {len(train_dataset)} Validation: {len(val_dataset)}')
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        collate_fn=collater
    )
    logger.info('Dataloader has created!')

    # Build model.
    logger.info('Building model and initializing!')

    model = model_selected(config).to(device)
    if args.resume:
        ckpt_model = convert_multi_gpu_checkpoint_to_single_gpu(ckpt)
        model.load_state_dict(ckpt_model)

    # Build optimizer and scheduler.
    optimizer = optimizer_selected(config.train.optimizer, model)
    scheduler = scheduler_selected(config.train.scheduler, optimizer)

    # Config the type of loss.
    cross_loss = OasMaskedHeavyCrossEntropyLoss()
    mask_acc_loss = MaskedAccuracy()       # Do not be considered during backward, only make sure the correction of mask.

    if args.resume:
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        # """Do not use the ckpt optimizer, because other layer has freezed."""
        it_sum = ckpt['iteration']
        logger.info('The re iteration start from {}'.format(it_sum))

    logger.info(f'# trainable parameters: {count_parameters(model) / 1e6:.4f} M')
    logger.info('Training...')
    best_val_loss = torch.inf
    best_iter = 0
    for it in range(0, config.train.max_iter+1):
        train(it)
        if it % config.train.valid_step == 0 or it == config.train.max_iter:
            valid_loss = valid(it, valid_type=args.data_name)
            # valid_loss = 1
            if valid_loss < best_val_loss:
                best_val_loss, best_iter = valid_loss, it
                logger.info(f'Bset validate loss achieved: {best_val_loss:.6f}')
                ckpt_path = os.path.join(ckpt_dir, '%d.pt'%it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }, ckpt_path)
            else:
                logger.info(f'[Validate] Val loss is not improved. '
                            f'Best val loss: {best_val_loss:.6f} at iter {best_iter}')

