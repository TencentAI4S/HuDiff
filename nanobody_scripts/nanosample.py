""" This script only consider for the nanobody. """
import os.path
import sys
current_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(current_dir)


import numpy as np
import torch
from tqdm import tqdm
import argparse
import pandas as pd
from abnumber import Chain
from anarci import anarci, number
from copy import deepcopy
import re
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from utils.anti_numbering import get_regions
from dataset.preprocess import HEAVY_POSITIONS_dict, HEAVY_CDR_INDEX, INPAINT_HEAVY_CDR_INDEX
from dataset.oas_pair_dataset_new import light_pad_cdr, HEAVY_REGION_INDEX, LIGHT_REGION_INDEX
from utils.tokenizer import Tokenizer
from utils.train_utils import model_selected
from utils.misc import get_new_log_dir, get_logger, seed_all

# Finetune package
from model.nanoencoder.abnativ_model import AbNatiV_Model
from model.nanoencoder.model import NanoAntiTFNet

REGION_LENGTH = (26, 12, 17, 10, 38, 30, 11)

def save_nano(heavy_chains, path):
    with open(path, 'w') as f:
        for heavy in heavy_chains:
            Chain.to_fasta(heavy, f, description='VH')


def seqs_to_fasta(df, save_path, version=None):
    assert version is not None, print('Need to given specific version.')
    seq_list = []
    seq_description_list = []
    for i in df.index:
        hseq = df.loc[i]['hseq']
        desci = 'VH' + f'{version}_{i}'
        seq_list.append(hseq)
        seq_description_list.append(desci)
    seqs_records = [SeqRecord(Seq(seq), id=seq_descrip) for seq, seq_descrip in zip(seq_list, seq_description_list)]
    with open(save_path, 'w') as f:
        SeqIO.write(seqs_records, f, 'fasta')


def compare_length(length_list):
    small = True
    for i, lg in enumerate(length_list):
        if lg <= REGION_LENGTH[i]:
            continue
        else:
            small = False
    return small

def get_diff_region_aa_seq(raw_seq, length_list):
    split_aa_seq_list = []
    start_lg = 0
    for lg in length_list:
        end_lg = start_lg + lg
        aa_seq = raw_seq[start_lg:end_lg]
        split_aa_seq_list.append(aa_seq)
        start_lg = end_lg
    assert ''.join(split_aa_seq_list) == raw_seq, 'Split length has wrong.'
    return split_aa_seq_list


def get_pad_seq(aa_seq):
    """
    :param aa_seq: AA seqs.
    :return: the pading AA seqs.
    """
    seq_dict = {}
    results = number(aa_seq, scheme='imgt')

    for key, value in results[0]:
        # print(key[0], key[1].strip())
        str_key = str(key[0]) + key[1].strip()
        seq_dict[str_key] = value
    # print(seq_dict)
    return seq_dict


def get_input_element(nano_aa):
    """
    :param mouse_h:
    :param mouse_l:
    :return:
    """
    # 1. Make sure the length of the sequence;
    # 2. Get the index of different region;
    # 3. Padding the sequence;
    # 4. Mask the sequence and get mask index (need shuffle);
    # 5. sample aa by aa.

    h_seq_dict = get_pad_seq(nano_aa)
    h_cdr_region = deepcopy(HEAVY_REGION_INDEX)
    h_pad_region = torch.tensor(h_cdr_region)

    nano_pad_initial_seq = ['-'] * len(HEAVY_CDR_INDEX)
    for key, value in h_seq_dict.items():
        try:
            pos_idx = HEAVY_POSITIONS_dict[key]
            nano_pad_initial_seq[pos_idx] = value
        except KeyError:
            nkey = re.findall(r'\d+', key)
            nkey = int(nkey[0])
            if (27 <= nkey <= 38) or (56 <= nkey <= 65) or ( 105 <= nkey <= 117):
                print("Heavy CDR has problem.")
            else:
                print('H Position {} is not in predefine dict, which can be ignored.'.format(key))
    # print(h_pad_region)
    # print(nano_pad_initial_seq)
    return h_pad_region, nano_pad_initial_seq


def batch_input_element(nano_sq, inpaint_sample=False, batch_size=10):

    nano_pad_region, nano_pad_initial_seq = get_input_element(nano_sq)

    # Get mask. Do not change CDR region.
    if not inpaint_sample:
        nano_heavy_index = HEAVY_CDR_INDEX
    else:
        nano_heavy_index = INPAINT_HEAVY_CDR_INDEX
    nano_mask = torch.tensor(nano_heavy_index) == 0

    # initial mask.
    ms_tokenizer = Tokenizer()
    nano_pad_seq_tokenize = ms_tokenizer.seq2idx(nano_pad_initial_seq)
    fram_h = ~(torch.tensor(nano_heavy_index) != 0) * nano_pad_seq_tokenize
    fram_pad_mask = (fram_h != 21)   # Because during finetune, which pad including in the framework do not consider training.
    nano_mask = fram_pad_mask * nano_mask
    nano_pad_seq_tokenize[nano_mask] = ms_tokenizer.idx_msk

    nano_pad_region = nano_pad_region.unsqueeze(0).expand(batch_size, -1).clone()
    nano_pad_seq_tokenize = nano_pad_seq_tokenize.unsqueeze(0).expand(batch_size, -1).clone()

    nano_loc = np.arange(len(nano_heavy_index))
    nano_loc = nano_loc[nano_mask]

    return nano_pad_seq_tokenize, nano_pad_region, nano_loc, ms_tokenizer


def get_nano_line(fpath):
    df_vhh = pd.read_csv(fpath)
    return df_vhh

# Def a read function. only output the sample human result.
def out_humanization_df(path):
    sample_df = pd.read_csv(path)
    human_df = sample_df[sample_df['Specific'] == 'humanization'].reset_index()
    return human_df


def save_seq_to_fasta(save_dir, save_df, species):
    for idx, line in save_df.iterrows():
        hseq=line['hseq']
        name = f'{idx}_{species}_H'
        save_fpath = os.path.join(save_dir, f'{idx}_{species}.fasta')
        with open(save_fpath, 'w') as f:
            seq_record = SeqRecord(Seq(hseq), id=name)
            SeqIO.write(seq_record, f, "fasta")


def split_fasta_for_save(fpath):
    sample_human_df = out_humanization_df(fpath)

    # Create file for save fa and pdb
    fa_fpath = os.path.join(os.path.dirname(fpath), 'sample_human_fa')
    pdb_fpath = os.path.join(os.path.dirname(fpath), 'sample_human_pdb')
    os.makedirs(fa_fpath, exist_ok=True)
    os.makedirs(pdb_fpath, exist_ok=True)

    save_seq_to_fasta(fa_fpath, sample_human_df, 'human')


def get_multi_model_state(ckpt):
    abnativ_state_dict = {
        k.partition('eval_abnativ_model.')[2]: v for k, v in ckpt['model'].items() if k.startswith('eval_abnativ_model.')
    }
    infilling_state_dict = {
        k.partition('infilling_pretrain.')[2]: v for k, v in ckpt['model'].items() if
        k.startswith('infilling_pretrain.')
    }
    return abnativ_state_dict, None, infilling_state_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        default='nanofinetune.pt'
                    )
    parser.add_argument('--data_fpath', type=str,
                        default='abnativ_select_vhh.csv'
                    )
    parser.add_argument('--batch_size', type=int,
                        default=1
                    )
    parser.add_argument('--sample_number', type=int,
                        default=1
                        )
    parser.add_argument('--try_number', type=int,
                        default=10
                        )
    parser.add_argument('--seed', type=int,
                        default=2023
                        )
    parser.add_argument('--sample_order', type=str,
                        default='shuffle')
    parser.add_argument('--sample_method', type=str,
                        default='gen', choices=['gen', 'rl_gen'])
    parser.add_argument('--length_limit', type=str,
                        default='not_equal')
    parser.add_argument('--model', type=str,
                        default='finetune_vh', choices=['pretrain', 'finetune_vh'])
    parser.add_argument('--fa_version', type=str,
                        default='v_nano')
    parser.add_argument('--inpaint_sample', type=eval,
                        default=False)
    parser.add_argument('--structure', type=eval,
                        default=False)
    args = parser.parse_args()

    print(args.inpaint_sample)
    batch_size = args.batch_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_all(args.seed)

    if 'filter' in args.data_fpath:
        data_sample = 'abnativ_select'
    else:
        data_sample = 'nanobert'

    # Make sure the name of sample log.
    sample_tag = f'{args.seed}_{args.sample_order}_{data_sample}_{args.sample_method}_{args.length_limit}_{args.model}'

    # log dir
    log_path = os.path.dirname(os.path.dirname(args.ckpt))
    log_dir = get_new_log_dir(
        root=log_path,
        prefix=sample_tag
    )
    logger = get_logger('test', log_dir)

    if args.model == 'pretrain':
        ckpt = torch.load(args.ckpt, map_location=device)
        config = ckpt['config']
        model = model_selected(config).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()

    elif args.model == 'finetune_vh':
        ckpt = torch.load(args.ckpt, map_location=device)
        config = ckpt['config']
        abnativ_state, _, infilling_state = get_multi_model_state(ckpt)
        # Abnativ model.
        hparams = ckpt['abnativ_params']
        abnativ_model = AbNatiV_Model(hparams)
        abnativ_model.load_state_dict(abnativ_state)
        abnativ_model.to(device)
        infilling_params = ckpt['infilling_params']
        infilling_model = NanoAntiTFNet(**infilling_params)
        infilling_model.load_state_dict(infilling_state)
        infilling_model.to(device)

        model_dict = {
            'abnativ': abnativ_model,
            'infilling': infilling_model,
            'target_infilling': infilling_model,
        }

        # Carefull!!! tmp.                                                                                                                 │|                               |                      |                  N/A |
        config.model['equal_weight'] = True
        config.model['vhh_nativeness'] = False
        config.model['human_threshold'] = None
        config.model['human_all_seq'] = False
        config.model['temperature'] = False

        framework_model = model_selected(config, pretrained_model=model_dict, tokenizer=Tokenizer())
        model = framework_model.infilling_pretrain
        model.eval()

    logger.info(args.ckpt)
    logger.info(args.seed)

    # save path
    save_fpath = os.path.join(log_dir, 'sample_humanization_result.csv')
    with open(save_fpath, 'a', encoding='UTF-8') as f:
        f.write('Specific,name,hseq,\n')

    wrong_idx_list = []
    length_not_equal_list = []
    nano_df = get_nano_line(args.data_fpath)
    for idx, nano_line in tqdm(enumerate(nano_df.itertuples()), total=len(nano_df.index)):
        sample_number = args.sample_number
        try_num = args.try_number
        nano_vhh = nano_line.vhhseq
        nano_pad_token, nano_pad_region, nano_loc, ms_tokenizer = batch_input_element(nano_vhh,
                                                                                      inpaint_sample=args.inpaint_sample,
                                                                                      batch_size=batch_size
                                                                                    )
        origin = 'nano'
        name = idx
        with open(save_fpath, 'a', encoding='UTF-8') as f:
            f.write(f'{origin},{name},{nano_vhh}\n')

        if args.sample_order == 'shuffle':
            np.random.shuffle(nano_loc)
        while sample_number > 0 and try_num > 0:
            all_token = ms_tokenizer.toks
            with torch.no_grad():
                for i in tqdm(nano_loc, total=len(nano_loc)):
                    nano_prediction = model(
                        nano_pad_token.to(device),
                        nano_pad_region.to(device),
                        H_chn_type=None
                    )

                    nano_pred = nano_prediction[:, i, :len(all_token)-1]
                    nano_soft = torch.nn.functional.softmax(nano_pred, dim=1)
                    nano_sample = torch.multinomial(nano_soft, num_samples=1)
                    nano_pad_token[:, i] = nano_sample.squeeze()

            nano_untokenized = [ms_tokenizer.idx2seq(s) for s in nano_pad_token]
            for _, g_h in enumerate(nano_untokenized):
                if sample_number == 0:
                    break

                with open(save_fpath, 'a', encoding='UTF-8') as f:
                    logger.info(g_h)
                    try:
                        sample_origin = 'humanization'
                        sample_name = str(name) + 'human_sample'
                        # Make sure that the sample seq can be detected by the Chain.
                        test_chain = Chain(g_h, scheme='imgt')

                        f.write(f'{sample_origin},{sample_name},{g_h}\n')

                        sample_number -= 1
                    except:
                        if try_num == 1:
                            sample_origin = 'humanization'
                            sample_name = str(name) + 'human_sample'
                            f.write(f'{sample_origin},{sample_name},{g_h}\n')
                        logger.info('Need to re sample again.')
                    try_num -= 1

    # Save as fasta for biophi oasis.
    fasta_save_fpath = os.path.join(log_dir, 'sample_identity.fa')
    logger.info('Save fasta fpath: {}'.format(fasta_save_fpath))
    sample_df = pd.read_csv(save_fpath)
    sample_human_df = sample_df[sample_df['Specific'] == 'humanization'].reset_index(drop=True)
    seqs_to_fasta(sample_human_df, fasta_save_fpath, version=args.fa_version)

    # Split save as fasta for structure prediction.
    if args.structure:
        split_fasta_for_save(save_fpath)


    logger.info('Length did not equal list: {}'.format(length_not_equal_list))
    logger.info('Wrong idx: {}'.format(wrong_idx_list))




