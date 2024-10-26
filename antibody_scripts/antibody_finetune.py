import os
import yaml
import argparse
import shutil
import sys
from easydict import EasyDict
from tqdm import tqdm

import torch
import torch.utils
import torch.utils.tensorboard
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(current_dir)


from utils.misc import (get_new_log_dir, get_logger, seed_all,
                        inf_iterator, count_parameters
                        )
from utils.train_utils import get_dataset, model_selected, optimizer_selected, scheduler_selected
from utils.tokenizer import Tokenizer
from utils.loss import OasMaskedSplitCrossEntropyLoss
from dataset.oas_pair_dataset_new import OasPairMaskCollater
from model.nanoencoder.abnativ_model import AbNatiV_Model

def get_abnativ_parameters(ckpt):
    ckpt_dict = torch.load(ckpt, map_location='cpu')
    hparams = ckpt_dict['hyper_parameters']['hparams']
    return ckpt_dict, hparams


def finetune():
    all_score_loss = 0.
    sum_loss = 0.
    all_h_cdr_loss = 0.
    all_l_cdr_loss = 0.
    all_h_loss = 0.
    all_l_loss = 0.
    mouse_h_ratio = finetune_config.model.mouse_resi_h_ratio
    mouse_l_ratio = finetune_config.model.mouse_resi_l_ratio
    for _ in range(finetune_config.finetune.batch_acc):
        optimizer.zero_grad()
        (H_L_src, H_L_tgt, H_L_region, chain_type, batch,
            H_L_masks, H_L_cdr_masks,
            H_L_timesteps, aho_h_seq, aho_l_seq) = next(train_iterator)
        H_L_src, H_L_tgt = H_L_src.to(device), H_L_tgt.to(device)
        H_L_region = H_L_region.to(device)
        chain_type, batch = chain_type.to(device), batch.to(device)
        H_L_masks, H_L_cdr_masks = H_L_masks.to(device), H_L_cdr_masks.to(device)
        H_L_timesteps = H_L_timesteps.to(device)
        aho_h_seq, aho_l_seq = aho_h_seq.to(device), aho_l_seq.to(device)

        human_score, H_L_pred, _, _ = anti_finetune_framework(
            H_L_src, 
            H_L_region, 
            chain_type, 
            H_L_masks,
            H_L_tgt,
            aho_h_seq,
            aho_l_seq,
            device
            )
        H_loss, _, H_cdr_loss, L_loss, _, L_cdr_loss = cdr_loss(
            H_L_pred,
            H_L_tgt,
            H_L_masks,
            H_L_cdr_masks,
            H_L_timesteps
            )
        loss = human_score + H_cdr_loss + L_cdr_loss + mouse_h_ratio * H_loss + mouse_l_ratio * L_loss
        loss.mean()
        loss.backward()
        optimizer.step()

        sum_loss += loss
        all_score_loss += human_score
        all_h_cdr_loss += H_cdr_loss
        all_l_cdr_loss += L_cdr_loss
        all_h_loss += H_loss
        all_l_loss += L_loss
    mean_loss = sum_loss / finetune_config.finetune.batch_acc
    mean_score_loss = all_score_loss / finetune_config.finetune.batch_acc
    mean_h_cdr_loss = all_h_cdr_loss / finetune_config.finetune.batch_acc
    mean_l_cdr_loss = all_l_cdr_loss / finetune_config.finetune.batch_acc
    mean_h_loss = all_h_loss / finetune_config.finetune.batch_acc
    mean_l_loss = all_l_loss / finetune_config.finetune.batch_acc
    logger.info('Finetuning iter {}, Loss is : {:.6f} | score Loss: {:.6f} | '
                'H cdr loss: {:.6f} | L cdr loss {:.6f} | H loss: {:.6f} | L loss: {:.6f}'.format(
                    it, mean_loss, mean_score_loss, mean_h_cdr_loss, mean_l_cdr_loss, mean_h_loss, mean_l_loss
                ))
    writer.add_scalar('finetune/loss', mean_loss, it)
    writer.add_scalar('finetune/Score_loss', mean_score_loss, it)
    writer.add_scalar('finetune/H_cdr_loss', mean_h_cdr_loss, it)
    writer.add_scalar('finetune/L_cdr_loss', mean_l_cdr_loss, it)
    writer.add_scalar('finetune/H_loss', mean_h_loss, it)
    writer.add_scalar('finetune/L_loss', mean_l_loss, it)
    writer.add_scalar('finetune/lr', optimizer.param_groups[0]['lr'], it)
    writer.flush()


def valid():
    all_score_loss = 0.
    sum_loss = 0.
    all_h_cdr_loss = 0.
    all_l_cdr_loss = 0.
    all_h_loss = 0.
    all_l_loss = 0.
    mouse_h_ratio = finetune_config.model.mouse_resi_h_ratio
    mouse_l_ratio = finetune_config.model.mouse_resi_l_ratio
    anti_finetune_framework.eval()
    val_sum = len(val_loader)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, desc='Val', total=len(val_loader))):   
            (H_L_src, H_L_tgt, H_L_region, chain_type, batch,
                H_L_masks, H_L_cdr_masks,
                H_L_timesteps, aho_h_seq, aho_l_seq) = batch
            H_L_src, H_L_tgt = H_L_src.to(device), H_L_tgt.to(device)
            H_L_region = H_L_region.to(device)
            chain_type, batch = chain_type.to(device), batch.to(device)
            H_L_masks, H_L_cdr_masks = H_L_masks.to(device), H_L_cdr_masks.to(device)
            H_L_timesteps = H_L_timesteps.to(device)
            aho_h_seq, aho_l_seq = aho_h_seq.to(device), aho_l_seq.to(device)

            human_score, H_L_pred, _, _  = anti_finetune_framework(
                H_L_src, 
                H_L_region, 
                chain_type, 
                H_L_masks, 
                H_L_tgt, 
                aho_h_seq,
                aho_l_seq,
                device
                )
            H_loss, _, H_cdr_loss, L_loss, _, L_cdr_loss = cdr_loss(
                H_L_pred,
                H_L_tgt,
                H_L_masks,
                H_L_cdr_masks,
                H_L_timesteps
                )
            loss = human_score + H_cdr_loss + L_cdr_loss + mouse_h_ratio * H_loss + mouse_l_ratio * L_loss
            loss.mean()

            sum_loss += loss
            all_score_loss += human_score
            all_h_cdr_loss += H_cdr_loss
            all_l_cdr_loss += L_cdr_loss
            all_h_loss += H_loss
            all_l_loss += L_loss
    mean_loss = sum_loss / val_sum
    mean_score_loss = all_score_loss / val_sum
    mean_h_cdr_loss = all_h_cdr_loss / val_sum
    mean_l_cdr_loss = all_l_cdr_loss / val_sum
    mean_h_loss = all_h_loss / val_sum
    mean_l_loss = all_l_loss / val_sum

    scheduler.step(mean_loss)

    logger.info('Validation iter {}, Loss is : {:.6f} | score Loss: {:.6f} | '
                'H cdr loss: {:.6f} | L cdr loss {:.6f} | H loss: {:.6f} | L loss: {:.6f}'.format(
                    it, mean_loss, mean_score_loss, mean_h_cdr_loss, mean_l_cdr_loss,
                    mean_h_loss, mean_l_loss
                ))
    writer.add_scalar('val/loss', mean_loss, it)
    writer.add_scalar('val/Score_loss', mean_score_loss, it)
    writer.add_scalar('val/H_cdr_loss', mean_h_cdr_loss, it)
    writer.add_scalar('val/L_cdr_loss', mean_l_cdr_loss, it)
    writer.add_scalar('val/H_loss', mean_h_loss, it)
    writer.add_scalar('val/L_loss', mean_l_loss, it)
    writer.flush()

    return mean_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This progress is designed' \
                                      'for finetuning the antibody humaization model.')
    parser.add_argument('--pair_mouse_data_path', type=str,
                        default='/data/oas_pair_mouse_data'
                        )
    parser.add_argument('--data_version', type=str,
                        default='filter'
                        )
    parser.add_argument('--data_name', type=str,
                        default='mouse'
                        )
    parser.add_argument('--config_path', type=str,
                        default='/configs/antibody_finetune.yml'
                        )
    parser.add_argument('--log_path', type=str,
                        default='/antibody_finetune_log/'
                        )
    parser.add_argument('--ckpt_path', type=str,
                        default='/checkpoints/antibody/antibody_diff.pt',
                        help='Specific checkpoint'
                        )
    parser.add_argument('--consider_mouse', type=bool,
                        default=True
                        )
    parser.add_argument('--resume', type=bool,
                        default=False,
                        )
    parser.add_argument('--resume_ckpt_path', type=str,
                        default=None
                        )
    parser.add_argument('--max_norm', type=int,
                        default=1
                        )
    args = parser.parse_args()


    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Checkpoint parameters.
    if not args.resume:
        pretrained_ckpt = torch.load(args.ckpt_path, map_location='cpu')
        pretrained_config = pretrained_ckpt['config']
    else:
        pretrained_ckpt = torch.load(args.resume_ckpt_path, map_location='cpu')
        pretrained_config = pretrained_ckpt['pretrain_config']

    with open(args.config_path, 'r') as f:
        finetune_config = EasyDict(yaml.safe_load(f))

    mouse_h_ratio = finetune_config.model.mouse_resi_h_ratio
    mouse_l_ratio = finetune_config.model.mouse_resi_l_ratio
    version = f'fintune_kabat_{args.data_version}_{args.resume}_ratio_{mouse_h_ratio}-{mouse_l_ratio}'
    log_dir = get_new_log_dir(
        root=args.log_path,
        prefix=version
    )
    # Checkpoints dir.
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # logger and writer
    logger = get_logger('finetune', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(f'Pretrained parameters: {pretrained_config}')
    logger.info('-----------')
    logger.info(f'Finetune parameters: {finetune_config}')
    
    # Copy files for checking.
    shutil.copyfile(args.config_path, os.path.join(log_dir, os.path.basename(args.config_path)))
    shutil.copyfile('./antibody_scripts/antibody_finetune.py', os.path.join(log_dir, 'antibody_finetune.py'))
    shutil.copytree('./model', os.path.join(log_dir, 'model'))
    # Reproduction.
    seed_all(pretrained_config.train.seed)

    # Create dataloader.
    subsets = get_dataset(args.pair_mouse_data_path, args.data_name, args.data_version)
    train_dataset, val_dataset = subsets['train'], subsets['val']
    collater = OasPairMaskCollater(n_region=pretrained_config.model.n_region, consider_mouse=args.consider_mouse)

    train_iterator = inf_iterator(
        DataLoader(
            train_dataset,
            batch_size=finetune_config.finetune.batch_size,
            num_workers=finetune_config.finetune.num_workers,
            shuffle=True,
            collate_fn=collater
        )
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=finetune_config.finetune.batch_size,
        collate_fn=collater
    )

    logger.info(f'Training: {len(train_dataset)} Validation: {len(val_dataset)}')
    logger.info('Dataloader has created!')

    # Building model and create the fine-tuning framework.
    logger.info('Building model and initializing!')
    # Introduce the antibody pretrained model.
    pretrain_model = model_selected(pretrained_config)
    pretrain_model.load_state_dict(pretrained_ckpt['model'])

    # Introduce the tfold prediction module. (may not useful.)
    # tfold_predictor = AbPredictor(
    #     ppi_path=f'{args.tfold_model_path}/esm_ppi_650m.pth', 
    #     ab_path=f'{args.tfold_model_path}/tfold_ab.pth',
    #     device=device
    #     )

    # Abnativ Humanness Model Construct.
    # heavy.
    vh_ckpt, hparams = get_abnativ_parameters(finetune_config.preckpt.ab_vh_ckpt)
    ab_vh_model = AbNatiV_Model(hparams)
    ab_vh_model.load_state_dict(vh_ckpt['state_dict'])
    for param in ab_vh_model.parameters():
        param.requires_grad = False
    # light kappa.
    vlk_ckpt, lkparams = get_abnativ_parameters(finetune_config.preckpt.ab_vlk_ckpt)
    ab_vlk_model = AbNatiV_Model(lkparams)
    ab_vlk_model.load_state_dict(vlk_ckpt['state_dict'])
    for param in ab_vlk_model.parameters():
        param.requires_grad = False
    # light lambda.
    vll_ckpt, llparmas = get_abnativ_parameters(finetune_config.preckpt.ab_vll_ckpt)
    ab_vll_model = AbNatiV_Model(llparmas)
    ab_vll_model.load_state_dict(vll_ckpt['state_dict'])
    for param in ab_vll_model.parameters():
        param.requires_grad = False


    pretrain_model_dict = {
        'antibody_pretrained': pretrain_model,
        # 'tfold_ab': tfold_predictor
        'ab_vh_model': ab_vh_model,
        'ab_vlk_model': ab_vlk_model,
        'ab_vll_model': ab_vll_model
    }
    
    tokenizer = Tokenizer()
    # Construct the finetuning framework.
    anti_finetune_framework = model_selected(finetune_config, pretrain_model_dict, tokenizer)
    anti_finetune_framework.to(device)

    # Build optimizer and scheduler.
    optimizer = optimizer_selected(
                        finetune_config.finetune.optimizer, 
                        anti_finetune_framework
                        )
    scheduler = scheduler_selected(
                        finetune_config.finetune.scheduler, 
                        optimizer
                        )
    
    # Cdr loss.
    cdr_loss = OasMaskedSplitCrossEntropyLoss()
    
    logger.info(f'# trainable parameters: {count_parameters(anti_finetune_framework) / 1e6:.4f} M')
    logger.info('Fine-tuning...')
    best_val_loss = torch.inf
    best_iter = 0
    for it in range(0, finetune_config.finetune.max_iter+1):
        finetune()
        if it % finetune_config.finetune.valid_step == 0 and it !=0 or it == finetune_config.finetune.max_iter:
            valid_loss = valid()
            if valid_loss < best_val_loss:
                best_val_loss, best_iter = valid_loss, it
                logger.info(f'Bset validate loss achieved: {best_val_loss:.6f}')
                ckpt_path = os.path.join(ckpt_dir, '%d.pt'%it)
                torch.save({
                    'fineconfig': finetune_config,
                    'pretrain_config': pretrained_config,
                    'model': anti_finetune_framework.anti_infilling.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }, ckpt_path)
            else:
                logger.info(f'[Validate] Val loss is not improved. '
                            f'Best val loss: {best_val_loss:.6f} at iter {best_iter}')


