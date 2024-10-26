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

from dataset.oas_unpair_dataset_new import OasHeavyMaskCollater, OasCamelCollater
from torch.utils.data import DataLoader
from utils.train_utils import model_selected, optimizer_selected, scheduler_selected
from utils.misc import seed_all, get_new_log_dir, get_logger, inf_iterator, count_parameters
from utils.loss import OasMaskedNanoCrossEntropyLoss, OasMaskedHeavyCrossEntropyLoss
from utils.train_utils import get_dataset
from model.nanoencoder.model import NanoAntiTFNet
from utils.tokenizer import Tokenizer

# Finetune package
from model.nanoencoder.abnativ_model import AbNatiV_Model

def get_abnativ_parameters(ckpt):
    ckpt_dict = torch.load(ckpt, map_location=device)
    hparams = ckpt_dict['hyper_parameters']['hparams']
    return ckpt_dict, hparams


def get_infilling_parameters(ckpt):
    ckpt_dict = torch.load(ckpt, map_location=device)
    model_param = ckpt_dict['config'].model
    model_weight = convert_multi_gpu_checkpoint_to_single_gpu(ckpt_dict)
    return model_weight, model_param


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

def finetune(it):

    sum_loss = 0
    sum_ab_loss = 0
    sum_cdr_loss = 0
    sum_recon_loss = 0
    sum_ab_vh_loss = 0
    sum_ab_vhh_loss = 0

    sum_heavy_loss = 0
    sum_heavy_cdr_loss = 0
    for _ in range(config.finetune.batch_acc):
        optimizer.zero_grad()

        if args.cross_training and it % config.finetune.cross_interval == 0:
            (H_src, H_tgt, H_region, chain_type,
             H_masks, H_cdr_masks,
             H_timesteps) = next(heavy_train_iterator)
            H_src, H_tgt = H_src.to(device), H_tgt.to(device)
            H_region = H_region.to(device)
            chain_type = chain_type.to(device)
            H_masks, H_cdr_masks = H_masks.to(device), H_cdr_masks.to(device)
            H_timesteps = H_timesteps.to(device)

            H_pred = framework_model.infilling_pretrain(H_src, H_region, chain_type)
            H_loss, H_nll, H_cdr_loss = heavy_cross_loss(
                H_pred,
                H_tgt,
                H_masks,
                H_cdr_masks,
                H_timesteps
            )
            heavy_loss = H_loss + H_cdr_loss
            heavy_loss.mean()
            heavy_loss.backward()
            optimizer.step()

            sum_heavy_loss += heavy_loss
            sum_heavy_cdr_loss += H_cdr_loss

        (vhh_token_src, vhh_src_mask, vhh_token,
         vhh_region, vhh_cdr_index, vhh_cdr_mask,
         vhh_timesteps, vhh_ab_input) = next(vhh_train_iterator)

        vhh_token_src, vhh_src_mask, vhh_token = vhh_token_src.to(device), vhh_src_mask.to(device), vhh_token.to(device),
        vhh_region, vhh_cdr_index, vhh_cdr_mask = vhh_region.to(device), vhh_cdr_index.to(device), vhh_cdr_mask.to(device)
        vhh_timesteps = vhh_timesteps.to(device)
        vhh_ab_input = vhh_ab_input.to(device)


        ab_loss, pred_seq, vh_loss, vhh_loss = framework_model(
                                                        vhh_token_src,
                                                        vhh_src_mask,
                                                        vhh_token,
                                                        vhh_region,
                                                        vhh_ab_input
                                                    )

        if not config.model.part_reconstruct_vhh:
            cdr_loss = vhh_finetune_loss(
                pred_seq,
                vhh_token,
                vhh_cdr_mask,
                None,
                None
            )

            ab_loss = ab_loss.mean()

            loss = ab_loss + cdr_loss

        else:
            cdr_loss, reconstruct_loss = vhh_finetune_loss(
                pred_seq,
                vhh_token,
                vhh_cdr_mask,
                vhh_src_mask,
                vhh_timesteps,
                reconstruct=True
            )
            ab_loss = ab_loss.mean()
            recon_weight = config.finetune.reconstruct_loss_weight
            loss = ab_loss + cdr_loss + recon_weight * reconstruct_loss
        loss.backward()
        optimizer.step()

        sum_loss += loss
        sum_ab_loss += ab_loss
        sum_cdr_loss += cdr_loss
        sum_ab_vh_loss += vh_loss
        sum_ab_vhh_loss += vhh_loss

        if config.model.part_reconstruct_vhh:
            sum_recon_loss += reconstruct_loss

    mean_loss = sum_loss / config.finetune.batch_acc
    mean_ab_loss = sum_ab_loss / config.finetune.batch_acc
    mean_cdr_loss = sum_cdr_loss / config.finetune.batch_acc
    # Sum for cdr.
    mean_ab_vh_loss = sum_ab_vh_loss / config.finetune.batch_acc
    mean_ab_vhh_loss = sum_ab_vhh_loss / config.finetune.batch_acc

    # Must make sure the config.finetune.batch_acc equal 1.
    if args.cross_training:
        mean_heavy_loss = sum_heavy_loss / config.finetune.batch_acc
        mean_heavy_cdr_loss = sum_heavy_cdr_loss / config.finetune.batch_acc

    if config.model.part_reconstruct_vhh:
        mean_recon_loss = sum_recon_loss / config.finetune.batch_acc

    if not config.model.part_reconstruct_vhh:
        logger.info('Finetuning iter {}, Loss is: {:.6f} | Mean ab loss: {:.6f} | Mean cdr loss: {:.6f} | LR: {:.6f}'.
                format(it, mean_loss, mean_ab_loss, mean_cdr_loss, optimizer.param_groups[0]['lr']))
        logger.info('Judge the variation of VH score loss and VHH score loss iter {}, judge loss: '
                    'ab_vh_loss: {:.6f} | ab_vhh_loss: {:.6f}'.format(it, mean_ab_vh_loss, mean_ab_vhh_loss))
    else:
        logger.info('Finetuning iter {}, Loss is: {:.6f} | Mean ab loss: {:.6f} | '
                    'Mean cdr loss: {:.6f} | Mean recon loss: {:.6f} | LR: {:.6f}'.
                format(it, mean_loss, mean_ab_loss, mean_cdr_loss, mean_recon_loss, optimizer.param_groups[0]['lr']))

    if args.cross_training:
        logger.info('Heavy Finetuning iter{}, Heavy loss is: {:.6f} | Heavy cdr loss; {:.6f}'.format(
            it, mean_heavy_loss, mean_heavy_cdr_loss
        ) )

    writer.add_scalar('Finetuning/loss', mean_loss, it)
    writer.add_scalar('Finetuning/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('Finetuning/ab_loss', mean_ab_loss, it)
    writer.add_scalar('Finetuning/cdr_loss', mean_cdr_loss, it)
    writer.add_scalar('Finetuning/ab_vh_loss', mean_ab_vh_loss, it)
    writer.add_scalar('Finetuning/ab_vhh_loss', mean_ab_vhh_loss, it)
    if config.model.part_reconstruct_vhh:
        writer.add_scalar('Finetuning/recon_loss', mean_recon_loss, it)

    if args.cross_training:
        writer.add_scalar('Finetuning/heavy_loss', mean_heavy_loss, it)
        writer.add_scalar('Finetuning/heavy_cdr_loss', mean_heavy_cdr_loss, it)
    writer.flush()


def finetune_val(it):
    sum_valid_loss = 0
    sum_ab_loss = 0
    sum_cdr_loss = 0
    sum_recon_loss = 0
    sum_ab_vh_loss = 0
    sum_ab_vhh_loss = 0

    sum_heavy_loss = 0
    sum_heavy_cdr_loss = 0
    framework_model.eval()

    val_sum = len(vhh_val_loader)
    with torch.no_grad():
        if args.cross_training:
            for batch in tqdm(heavy_val_loader, desc='Val', total=len(heavy_val_loader)):
                (H_src, H_tgt, H_region, chain_type,
                 H_masks, H_cdr_masks,
                 H_timesteps) = batch
                H_src, H_tgt = H_src.to(device), H_tgt.to(device)
                H_region = H_region.to(device)
                chain_type = chain_type.to(device)
                H_masks, H_cdr_masks = H_masks.to(device), H_cdr_masks.to(device)
                H_timesteps = H_timesteps.to(device)

                H_pred = framework_model.infilling_pretrain(H_src, H_region, chain_type)
                H_loss, H_nll, H_cdr_loss = heavy_cross_loss(
                    H_pred,
                    H_tgt,
                    H_masks,
                    H_cdr_masks,
                    H_timesteps
                )
                heavy_loss = H_loss + H_cdr_loss
                heavy_loss.mean()

                sum_heavy_loss += heavy_loss
                sum_heavy_cdr_loss += H_cdr_loss

        for i, batch in enumerate(tqdm(vhh_val_loader, desc='Val', total=len(vhh_val_loader))):
            (vhh_token_src, vhh_src_mask, vhh_token,
             vhh_region, vhh_cdr_index, vhh_cdr_mask,
             vhh_timesteps, vhh_ab_input) = batch
            vhh_token_src, vhh_src_mask, vhh_token = vhh_token_src.to(device), vhh_src_mask.to(device), vhh_token.to(
                device),
            vhh_region, vhh_cdr_index, vhh_cdr_mask = vhh_region.to(device), vhh_cdr_index.to(device), vhh_cdr_mask.to(
                device)
            vhh_timesteps = vhh_timesteps.to(device)
            vhh_ab_input = vhh_ab_input.to(device)

            ab_loss, pred_seq, vh_loss, vhh_loss = framework_model(vhh_token_src, vhh_src_mask, vhh_token, vhh_region, vhh_ab_input)
            if not config.model.part_reconstruct_vhh:

                cdr_loss = vhh_finetune_loss(
                    pred_seq,
                    vhh_token,
                    vhh_cdr_mask,
                    None,
                    None
                )

                ab_loss = ab_loss.mean()
                val_loss = ab_loss + cdr_loss
            else:
                cdr_loss, reconstruct_loss = vhh_finetune_loss(
                    pred_seq,
                    vhh_token,
                    vhh_cdr_mask,
                    vhh_src_mask,
                    vhh_timesteps,
                    reconstruct=True
                )
                ab_loss = ab_loss.mean()
                recon_weight = config.finetune.reconstruct_loss_weight
                val_loss = ab_loss + cdr_loss + recon_weight * reconstruct_loss
            sum_valid_loss += val_loss
            sum_ab_loss += ab_loss
            sum_cdr_loss += cdr_loss
            sum_ab_vh_loss += vh_loss
            sum_ab_vhh_loss += vhh_loss

            if config.model.part_reconstruct_vhh:
                sum_recon_loss += reconstruct_loss


    mean_loss = sum_valid_loss / val_sum
    mean_ab_loss = sum_ab_loss / val_sum
    mean_cdr_loss = sum_cdr_loss / val_sum
    mean_ab_vh_loss = sum_ab_vh_loss / val_sum
    mean_ab_vhh_loss = sum_ab_vhh_loss / val_sum

    if args.cross_training:
        mean_heavy_loss = sum_heavy_loss / len(heavy_val_loader)
        mean_heavy_cdr_loss = sum_heavy_cdr_loss / len(heavy_val_loader)

    if config.model.part_reconstruct_vhh:
        mean_recon_loss = sum_recon_loss / val_sum

    if config.finetune.scheduler.type == 'warm_up':
        scheduler.step()
    elif config.finetune.scheduler.type == 'plateau':
        scheduler.step(val_loss)
    else:
        print(f'Scheduler {scheduler.finetune.scheduler.type} not provid.')

    if not config.model.part_reconstruct_vhh:
        logger.info('Validation iter {}, Loss is: {:.6f} | Val ab loss: {:.6f} | Val cdr loss: {:.6f}'.
                    format(it, mean_loss, mean_ab_loss, mean_cdr_loss))
        logger.info('Judge the variation of VH score loss and VHH score loss iter {}, judge loss: '
                    'ab_vh_loss: {:.6f} | ab_vhh_loss: {:.6f}'.
                    format(it, mean_ab_vh_loss, mean_ab_vhh_loss))
    else:
        logger.info('Validation iter {}, Loss is: {:.6f} | Val ab loss: {:.6f} | '
                    'Val cdr loss: {:.6f} | Val Recon loss: {:.6f}'.
                    format(it, mean_loss, mean_ab_loss, mean_cdr_loss, mean_recon_loss))

    if args.cross_training:
        logger.info('Heavy Validation iter{}, Heavy loss is: {:.6f} | Heavy cdr loss; {:.6f}'.format(
            it, mean_heavy_loss, mean_heavy_cdr_loss
        ) )

    writer.add_scalar('val/loss', mean_loss, it)
    writer.add_scalar('val/ab_loss', mean_ab_loss, it)
    writer.add_scalar('val/cdr_loss', mean_cdr_loss, it)
    writer.add_scalar('val/ab_vh_loss', mean_ab_vh_loss, it)
    writer.add_scalar('val/ab_vhh_loss', mean_ab_vhh_loss, it)
    if config.model.part_reconstruct_vhh:
        writer.add_scalar('val/recon_loss', mean_recon_loss, it)

    if args.cross_training:
        writer.add_scalar('val/heavy_loss', mean_heavy_loss, it)
        writer.add_scalar('val/heavy_cdr_loss', mean_heavy_cdr_loss, it)

    writer.flush()
    return mean_loss


if __name__ == '__main__':
    # Required args.
    parser = argparse.ArgumentParser()
    parser.add_argument('--vhh_data_fpath', type=str,
                        default=None
                        )
    parser.add_argument('--data_name', type=str,
                        default='vhh')
    parser.add_argument('--train_stage', type=str,
                        default='finetune_no_equal_weight')
    parser.add_argument('--data_version', type=str,
                        default='test')
    parser.add_argument('--config_path', type=str,
                        default=None
                        )
    parser.add_argument('--log_path', type=str,
                        default='tmp/'
                        )
    parser.add_argument('--unpari_data_path', type=str,
                        default=None
                        )
    parser.add_argument('--unpair_data_name', type=str,
                        default='heavy'
                        )
    parser.add_argument('--cross_training', type=eval,
                        default=False
                        )
    parser.add_argument('--resume', type=eval,
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

    # Reconstruct and VHH nativeness exist conflict: though purpose same, but training objects are different.
    version = f'{args.data_name}_{args.train_stage}_{args.data_version}' \
              f'_reconvhh_{config.model.part_reconstruct_vhh}_vhh_{config.model.vhh_nativeness}'
    print(f'Whether Considering the vhh score: {config.model.vhh_nativeness}')
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
    shutil.copyfile('./nanobody_scripts/nanofinetune.py', os.path.join(log_dir, 'nanofinetune.py'))
    shutil.copytree('./model', os.path.join(log_dir, 'model'))

    # Fixed
    seed_all(config.finetune.seed)

    # Create dataloader.
    # Here first we need to consider the new dataset is VHH dataset.
    # Then we need to reload the Abnativ socre model, and freezing its parameters.
    # Finally, need to consider the tfold structure (Thinking...).
    vhh_subsets = get_dataset(args.vhh_data_fpath, args.data_name, args.data_version)
    vhh_train_dataset, vhh_val_dataset = vhh_subsets['train'], vhh_subsets['val']
    vhh_collater = OasCamelCollater()
    vhh_train_iterator = inf_iterator(
        DataLoader(
            vhh_train_dataset,
            shuffle=True,
            batch_size=config.finetune.batch_size,
            num_workers=config.finetune.num_workers,
            collate_fn=vhh_collater
        )
    )

    logger.info(f'VHH Training: {len(vhh_train_dataset)} Validation: {len(vhh_val_dataset)}')
    vhh_val_loader = DataLoader(
        vhh_val_dataset,
        batch_size=config.finetune.batch_size,
        num_workers=config.finetune.num_workers,
        collate_fn=vhh_collater
    )
    logger.info(f'{args.train_stage} Dataloader has created!')

    # if cross-training, we need another human dataloader.
    if args.cross_training:
        heavy_subsets = get_dataset(args.unpair_data_path, args.unpair_data_name, args.data_version)
        heavy_train_dataset, heavy_val_dataset = heavy_subsets['train'], heavy_subsets['val']
        heavy_collater = OasHeavyMaskCollater()

        # Only consider Heavy.
        heavy_train_iterator = inf_iterator(DataLoader(
            heavy_train_dataset,
            batch_size=config.finetune.batch_size,
            num_workers=config.finetune.num_workers,
            shuffle=True,
            collate_fn=heavy_collater
        ))
        logger.info(f'VH Training: {len(heavy_train_dataset)} Validation: {len(heavy_val_dataset)}')
        heavy_val_loader = DataLoader(
            heavy_val_dataset,
            batch_size=config.finetune.batch_size,
            num_workers=config.finetune.num_workers,
            collate_fn=heavy_collater
        )
        logger.info('VH Dataloader has created!')


    # Abnativ Humanness Model Construct.
    ckpt, hparams = get_abnativ_parameters(config.finetune.model.abnativ_humanness_ckpt_fpath)
    abnativ_model = AbNatiV_Model(hparams)
    abnativ_model.load_state_dict(ckpt['state_dict'])
    abnativ_model.to(device)

    # Abnativ VHH nativeness Model Construct.
    if config.model.vhh_nativeness:
        vhh_ckpt, vhh_hparams = get_abnativ_parameters(config.finetune.model.abnativ_vhh_ckpt_fpath)
        vhh_abnativ_model = AbNatiV_Model(vhh_hparams)
        vhh_abnativ_model.load_state_dict(vhh_ckpt['state_dict'])
        vhh_abnativ_model.to(device)

    # Change the model state as eval.
    # Freezing the model parameters.
    abnativ_model.eval()
    for ab_parame in abnativ_model.parameters():
        ab_parame.requires_grad = False
    if config.model.vhh_nativeness:
        vhh_abnativ_model.eval()
        for vhh_ab_parame in vhh_abnativ_model.parameters():
            vhh_ab_parame.requires_grad = False

    # Infilling Model Construct.
    infilling_ckpt, infilling_parames = get_infilling_parameters(config.finetune.model.infilling_ckpt_fpath)
    infilling_model = NanoAntiTFNet(**infilling_parames)
    infilling_model.load_state_dict(infilling_ckpt)
    infilling_model.to(device)

    target_infilling_model = NanoAntiTFNet(**infilling_parames)
    target_infilling_model.load_state_dict(infilling_ckpt)
    target_infilling_model.to(device)
    for target_parame in target_infilling_model.parameters():
        target_parame.requires_grad = False

    # Construct the whole Framework Model. order [ab, tfold, infilling] or [ab, infilling]
    model_dict = {
        'abnativ': abnativ_model,
        'tfold_ab': None,
        'infilling': infilling_model,
        'target_infilling': target_infilling_model
    }
    if config.model.vhh_nativeness:
        model_dict['vhh_abnativ'] = vhh_abnativ_model

    tokenizer = Tokenizer()
    framework_model = model_selected(config, pretrained_model=model_dict, tokenizer=tokenizer)
    vhh_finetune_loss = OasMaskedNanoCrossEntropyLoss()
    heavy_cross_loss = OasMaskedHeavyCrossEntropyLoss()

    # Build optimizer and scheduler.
    optimizer = optimizer_selected(config.finetune.optimizer, framework_model)
    scheduler = scheduler_selected(config.finetune.scheduler, optimizer)

    logger.info(f'# finetune parameters: {count_parameters(framework_model) / 1e6:.4f} M')
    logger.info(f'Need to check whether the finetuen parameters are equal to the infilling parameters or not.')
    assert count_parameters(framework_model) == count_parameters(infilling_model), print('Need to check!')
    logger.info('Training...')

    best_val_loss = torch.inf
    best_iter = 0
    for it in range(0, config.finetune.max_iter + 1):
        finetune(it)
        if it % config.finetune.valid_step == 0 and it!= 0 or it == config.finetune.max_iter:
            valid_loss = finetune_val(it)
            if valid_loss < best_val_loss:
                best_val_loss, best_iter = valid_loss, it
                logger.info(f'Bset validate loss achieved: {best_val_loss:.6f}')
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': framework_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'abnativ_params': hparams,
                    'infilling_params': infilling_parames,
                    'iteration': it,
                }, ckpt_path)
            else:
                logger.info(f'[Validate] Val loss is not improved. '
                            f'Best val loss: {best_val_loss:.6f} at iter {best_iter}')

