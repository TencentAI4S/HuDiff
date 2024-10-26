import os
import argparse
import yaml
from easydict import EasyDict
import shutil
import sys
current_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(current_dir)

import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from tqdm import tqdm
# from torch.nn.utils import clip_grad_norm_

from dataset.oas_pair_dataset_new import OasPairMaskCollater
from utils.train_utils import model_selected, optimizer_selected, scheduler_selected
from utils.misc import seed_all, get_new_log_dir, get_logger, inf_iterator, count_parameters
from utils.loss import OasMaskedCrossEntropyLoss, MaskedAccuracy, OasMaskedSplitCrossEntropyLoss
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
    if config.train.loss_type == 'split':
        H_sum_loss, H_sum_nll = 0., 0.
        H_sum_cdr_loss = 0.
        L_sum_loss, L_sum_nll = 0., 0.
        L_sum_cdr_loss = 0.
    elif config.train.loss_type == 'merge':
        H_L_sum_loss, H_L_sum_nll = 0., 0.
        H_L_sum_cdr_loss = 0.
    sum_loss = 0
    H_L_sum_acc_loss = 0.
    sum_roc_auc = 0.

    model.train()
    for _ in range(config.train.batch_acc):
        optimizer.zero_grad()

        (H_L_src, H_L_tgt, H_L_region, chain_type, batch,
            H_L_masks, H_L_cdr_masks,
            H_L_timesteps) = next(train_iterator)
        H_L_src, H_L_tgt = H_L_src.to(device), H_L_tgt.to(device)
        H_L_region = H_L_region.to(device)
        chain_type, batch = chain_type.to(device), batch.to(device)
        H_L_masks, H_L_cdr_masks = H_L_masks.to(device), H_L_cdr_masks.to(device)
        H_L_timesteps = H_L_timesteps.to(device)

        H_L_pred = model(H_L_src, H_L_region, chain_type)

        if config.train.loss_type == 'split':
            H_loss, H_nll, H_cdr_loss, L_loss, L_nll, L_cdr_loss = cross_loss(
                                                H_L_pred,
                                                H_L_tgt,
                                                H_L_masks,
                                                H_L_cdr_masks,
                                                H_L_timesteps
                                                )
            loss = H_loss + L_loss + H_cdr_loss + L_cdr_loss
            loss.mean()
            loss.backward()
            optimizer.step()

            sum_loss += loss
            H_sum_loss += H_loss
            H_sum_nll += H_nll
            H_sum_cdr_loss += H_cdr_loss
            L_sum_loss += L_loss
            L_sum_nll += L_nll
            L_sum_cdr_loss += L_cdr_loss

        elif config.train.loss_type == 'merge':
            H_L_loss, H_L_nll, H_L_cdr_loss = cross_loss(
                H_L_pred,
                H_L_tgt,
                H_L_masks,
                H_L_cdr_masks,
                H_L_timesteps
            )

            loss = H_L_loss + H_L_cdr_loss

            loss.mean()
            loss.backward()
            optimizer.step()

            sum_loss += loss
            H_L_sum_loss += H_L_loss
            H_L_sum_nll += H_L_nll
            H_L_sum_cdr_loss += H_L_cdr_loss

        else:
            print("Loss type is wrong.")
        # Those value indicate whether the pred equl to tgt. max may is 1.
        # Not backward.
        H_L_acc_loss, roc_auc = mask_acc_loss(H_L_pred, H_L_tgt, H_L_masks)
        H_L_sum_acc_loss += H_L_acc_loss
        sum_roc_auc += roc_auc

    if config.train.loss_type == 'split':
        mean_loss = sum_loss / config.train.batch_acc

        mean_H_loss = H_sum_loss / config.train.batch_acc
        mean_H_nll = H_sum_nll / config.train.batch_acc
        mean_H_cdr_loss = H_sum_cdr_loss / config.train.batch_acc

        mean_L_loss = L_sum_loss / config.train.batch_acc
        mean_L_nll = L_sum_nll / config.train.batch_acc
        mean_L_cdr_loss = L_sum_cdr_loss / config.train.batch_acc


        # Not backward.
        mean_H_L_acc_loss = H_L_sum_acc_loss / config.train.batch_acc
        mean_roc_auc = sum_roc_auc / config.train.batch_acc

        logger.info('Training iter {}, Loss is: {:.6f} | H_loss: {:.6f} | H_nll: {:.6f} '
                    '| H_cdr_loss: {:.6f} | L_loss: {:.6f} | L_nll: {:.6f} '
                    '| L_cdr_loss: {:.6f} | H_L_acc: {:.6f} | ROC_AUC: {:.6f}'.
                    format(it, mean_loss, mean_H_loss, mean_H_nll, mean_H_cdr_loss,
                           mean_L_loss, mean_L_nll, mean_L_cdr_loss,
                           mean_H_L_acc_loss, mean_roc_auc))

        writer.add_scalar('train/loss', mean_loss, it)
        writer.add_scalar('train/H_loss', mean_H_loss, it)
        writer.add_scalar('train/H_nll', mean_H_nll, it)
        writer.add_scalar('train/H_cdr_loss', mean_H_cdr_loss, it)
        writer.add_scalar('train/L_loss', mean_L_loss, it)
        writer.add_scalar('train/L_nll', mean_L_nll, it)
        writer.add_scalar('train/L_cdr_loss', mean_L_cdr_loss, it)
        writer.add_scalar('train/H_L_acc', mean_H_L_acc_loss, it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/roc_auc', mean_roc_auc, it)
        writer.flush()

    elif config.train.loss_type == 'merge':
        mean_loss = sum_loss / config.train.batch_acc
        mean_H_L_loss = H_L_sum_loss / config.train.batch_acc
        mean_H_L_nll = H_L_sum_nll / config.train.batch_acc

        mean_H_L_cdr_loss = H_L_sum_cdr_loss / config.train.batch_acc

        # Not backward.
        mean_H_L_acc_loss = H_L_sum_acc_loss / config.train.batch_acc
        mean_roc_auc = sum_roc_auc / config.train.batch_acc

        logger.info('Training iter {}, Loss is: {:.6f} | H_L_loss: {:.6f} | H_L_nll: {:.6f} '
                    '| H_L_cdr_loss: {:.6f} | H_L_acc: {:.6f} | ROC_AUC: {:.6f}'.
                    format(it, mean_loss, mean_H_L_loss, mean_H_L_nll, mean_H_L_cdr_loss, mean_H_L_acc_loss, mean_roc_auc))
        writer.add_scalar('train/loss', mean_loss, it)
        writer.add_scalar('train/H_L_loss', mean_H_L_loss, it)
        writer.add_scalar('train/H_L_nll', mean_H_L_nll, it)
        writer.add_scalar('train/H_L_cdr_loss', mean_H_L_cdr_loss, it)
        writer.add_scalar('train/H_L_acc', mean_H_L_acc_loss, it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/roc_auc', mean_roc_auc, it)
        writer.flush()
    else:
        print('Loss type is wrong.')


def valid(it):
    if config.train.loss_type == 'split':
        H_sum_loss, H_sum_nll = 0., 0.
        H_sum_cdr_loss = 0.
        L_sum_loss, L_sum_nll = 0., 0.
        L_sum_cdr_loss = 0.
    elif config.train.loss_type == 'merge':
        H_L_sum_loss, H_L_sum_nll = 0., 0.
        H_L_sum_cdr_loss = 0.

    H_L_sum_acc_loss = 0.
    sum_valid_loss = 0.
    sum_roc_auc = 0.
    model.eval()
    val_sum = len(val_loader)
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Val', total=len(val_loader)):
            (H_L_src, H_L_tgt, H_L_region, chain_type, batch,
                H_L_masks, H_L_cdr_masks,
                H_L_timesteps) = batch
            H_L_src, H_L_tgt = H_L_src.to(device), H_L_tgt.to(device)
            H_L_region = H_L_region.to(device)
            chain_type, batch = chain_type.to(device), batch.to(device)
            H_L_masks, H_L_cdr_masks = H_L_masks.to(device), H_L_cdr_masks.to(device)
            H_L_timesteps = H_L_timesteps.to(device)

            H_L_pred = model(H_L_src, H_L_region, chain_type)

            if config.train.loss_type == 'split':
                H_loss, H_nll, H_cdr_loss, L_loss, L_nll, L_cdr_loss = cross_loss(
                    H_L_pred,
                    H_L_tgt,
                    H_L_masks,
                    H_L_cdr_masks,
                    H_L_timesteps
                )
                loss = H_loss + L_loss + H_cdr_loss + L_cdr_loss
                sum_valid_loss += loss
                H_sum_loss += H_loss
                H_sum_nll += H_nll
                H_sum_cdr_loss += H_cdr_loss
                L_sum_loss += L_loss
                L_sum_nll += L_nll
                L_sum_cdr_loss += L_cdr_loss

            elif config.train.loss_type == 'merge':
                H_L_loss, H_L_nll, H_L_cdr_loss = cross_loss(
                    H_L_pred,
                    H_L_tgt,
                    H_L_masks,
                    H_L_cdr_masks,
                    H_L_timesteps
                )
                loss = H_L_loss + H_L_cdr_loss
                sum_valid_loss += loss

                H_L_sum_loss += H_L_loss
                H_L_sum_nll += H_L_nll
                H_L_sum_cdr_loss += H_L_cdr_loss

            else:
                print("Loss type is wrong.")
            # Those value indicate whether the pred equl to tgt. max may is 1.
            H_L_acc_loss, roc_auc = mask_acc_loss(H_L_pred, H_L_tgt, H_L_masks)

            # Not backward.
            H_L_sum_acc_loss += H_L_acc_loss
            sum_roc_auc += roc_auc

    if config.train.loss_type == 'split':
        mean_loss = sum_valid_loss / val_sum
        mean_H_loss = H_sum_loss / val_sum
        mean_H_nll = H_sum_nll / val_sum
        mean_H_cdr_loss = H_sum_cdr_loss / val_sum
        mean_L_loss = L_sum_loss / val_sum
        mean_L_nll = L_sum_nll / val_sum
        mean_L_cdr_loss = L_sum_cdr_loss / val_sum

        # Not backward.
        mean_H_L_acc_loss = H_L_sum_acc_loss / val_sum
        mean_roc_auc = sum_roc_auc / val_sum
        scheduler.step(mean_loss)

        logger.info('Validation iter {}, Loss is: {:.6f} | H_loss: {:.6f} | H_nll: {:.6f} '
                    '| H_cdr_loss: {:.6f} | L_loss: {:.6f} | L_nll: {:.6f} '
                    '| L_cdr_loss: {:.6f} | H_L_acc: {:.6f} | ROC_AUC: {:.6f}'.
                    format(it, mean_loss,
                           mean_H_loss, mean_H_nll, mean_H_cdr_loss,
                           mean_L_loss, mean_L_nll, mean_L_cdr_loss,
                           mean_H_L_acc_loss, mean_roc_auc))
        writer.add_scalar('val/loss', mean_loss, it)
        writer.add_scalar('val/H_loss', mean_H_loss, it)
        writer.add_scalar('val/H_nll', mean_H_nll, it)
        writer.add_scalar('val/H_cdr_loss', mean_H_cdr_loss, it)
        writer.add_scalar('val/L_loss', mean_L_loss, it)
        writer.add_scalar('val/L_nll', mean_L_nll, it)
        writer.add_scalar('val/L_cdr_loss', mean_L_cdr_loss, it)
        writer.add_scalar('val/H_L_acc', mean_H_L_acc_loss, it)
        writer.add_scalar('val/roc_auc', mean_roc_auc, it)
        writer.flush()

    elif config.train.loss_type == 'merge':
        mean_loss = sum_valid_loss / val_sum
        mean_H_L_loss = H_L_sum_loss / val_sum
        mean_H_L_nll = H_L_sum_nll / val_sum
        mean_H_L_cdr_loss = H_L_sum_cdr_loss / val_sum

        # Not backward.
        mean_H_L_acc_loss = H_L_sum_acc_loss / val_sum
        mean_roc_auc = sum_roc_auc /val_sum

        scheduler.step(mean_loss)

        logger.info('Validation iter {}, Loss is: {:.6f} | H_L_loss: {:.6f} | H_L_nll: {:.6f} '
                    '| H_L_cdr_loss: {:.6f} | H_L_acc: {:.6f} | ROC_AUC: {:.6f}'.
                    format(it, mean_loss, mean_H_L_loss, mean_H_L_nll,
                           mean_H_L_cdr_loss, mean_H_L_acc_loss, mean_roc_auc))
        writer.add_scalar('val/loss', mean_loss, it)
        writer.add_scalar('val/H_L_loss', mean_H_L_loss, it)
        writer.add_scalar('val/H_L_nll', mean_H_L_nll, it)
        writer.add_scalar('val/H_L_cdr_loss', mean_H_L_cdr_loss, it)
        writer.add_scalar('val/H_L_acc', mean_H_L_acc_loss, it)
        writer.add_scalar('val/roc_auc', mean_roc_auc, it)
        writer.flush()

    else:
        print("loss type is wrong.")

    return mean_loss


if __name__ == '__main__':
    # Required args.
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair_data_path', type=str,
                        default='/data/oas_pair_human_data'
                        )
    parser.add_argument('--data_version', type=str,
                        default='filter'
                        )
    parser.add_argument('--data_name', type=str,
                       default='pair'
                        )
    parser.add_argument('--config_path', type=str,
                        default='/configs/antibody_test.yml'
                        )
    parser.add_argument('--log_path', type=str,
                        default='/antibody_log'
                        )
    parser.add_argument('--resume', type=bool,
                        default=False)
    parser.add_argument('--checkpoint', type=str,
                        default=None
                        )

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
        config.train.batch_size = 16  # Deal with debug ood problem.

    version = f'antibody_{args.data_name}_{args.data_version}_{config.train.loss_type}'
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
    shutil.copyfile('./antibody_scripts/antibody_train.py', os.path.join(log_dir, 'antibody_train.py'))
    shutil.copytree('./model', os.path.join(log_dir, 'model'))


    # Reproduction.
    seed_all(config.train.seed)

    # Create dataloader.
    subsets = get_dataset(args.pair_data_path, args.data_name, args.data_version)
    train_dataset, val_dataset = subsets['train'], subsets['val']
    collater = OasPairMaskCollater(n_region=config.model.n_region)

    # Pair
    train_iterator = inf_iterator(DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        shuffle=True,
        collate_fn=collater
    ))
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        collate_fn=collater
    )

    logger.info('Dataloader has created!')

    # Build model.
    logger.info(f'Training: {len(train_dataset)} Validation: {len(val_dataset)}')
    logger.info('Building model and initializing!')

    model = model_selected(config).to(device)
    if args.resume:
        ckpt_model = convert_multi_gpu_checkpoint_to_single_gpu(ckpt)
        model.load_state_dict(ckpt_model)

    # Build optimizer and scheduler.
    optimizer = optimizer_selected(config.train.optimizer, model)
    scheduler = scheduler_selected(config.train.scheduler, optimizer)

    # Config the type of loss.
    if config.train.loss_type == 'split':
        cross_loss = OasMaskedSplitCrossEntropyLoss(l_weight=config.train.l_loss_weight)
    else:
        cross_loss = OasMaskedCrossEntropyLoss()
    mask_acc_loss = MaskedAccuracy()       # Not be considered during backward, only make sure the correction of mask.

    if args.resume:
        # optimizer.load_state_dict(ckpt['optimizer'])
        # scheduler.load_state_dict(ckpt['scheduler'])
        """Do not use the ckpt optimizer, because other layer has freezed."""
        it_sum = ckpt['iteration']
        logger.info('The re iteration start from {}'.format(it_sum))

    logger.info(f'# trainable parameters: {count_parameters(model) / 1e6:.4f} M')
    logger.info('Training...')
    best_val_loss = torch.inf
    best_iter = 0
    for it in range(0, config.train.max_iter+1):
        train(it)
        if it % config.train.valid_step == 0 or it == config.train.max_iter:
            valid_loss = valid(it)
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

