"""Batch sample for validation datasets."""
import os.path
import sys
from copy import deepcopy
import re
import argparse
current_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(current_dir)

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from abnumber import Chain
from anarci import number


# from utils.anti_numbering import get_regions
from patent_eval import cal_all_preservation
from utils.tokenizer import Tokenizer
from utils.train_utils import model_selected
from utils.misc import get_new_log_dir, get_logger, seed_all
from dataset.oas_pair_dataset_new import HEAVY_REGION_INDEX, LIGHT_REGION_INDEX
from dataset.preprocess import (
                            HEAVY_POSITIONS_dict, LIGHT_POSITIONS_dict,
                            HEAVY_CDR_INDEX, LIGHT_CDR_INDEX,
                            HEAVY_CDR_KABAT_NO_VERNIER, LIGHT_CDR_KABAT_NO_VERNIER
                            )

REGION_LENGTH = (26, 12, 17, 10, 38, 30, 11)

def save_pairs(heavy_chains, light_chains, path):
    """
    Save the sequences as fasta file.
    """
    assert len(heavy_chains) == len(light_chains)
    with open(path, 'w') as f:
        for heavy, light in zip(heavy_chains, light_chains):
            Chain.to_fasta(heavy, f, description='VH')
            Chain.to_fasta(light, f, description='VL')


def trans_to_chain(df, save_path, version=None):
    H_chain_list = [Chain(df.iloc[i]['hseq'], scheme='imgt') for i in df.index]
    L_chain_list = [Chain(df.iloc[i]['lseq'], scheme='imgt') for i in df.index]

    assert version is not None, print('Need to given specific version.')
    for i, (h_chain, l_chain) in tqdm(enumerate(zip(H_chain_list, L_chain_list)), 
                                      total=len(H_chain_list)):
        name = version + 'human' + f'{i}'
        h_chain.name = name
        l_chain.name = name

    save_pairs(H_chain_list, L_chain_list, save_path)


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
        str_key = str(key[0]) + key[1].strip()
        seq_dict[str_key] = value
    seq_chain_type = Chain(aa_seq, scheme='imgt').chain_type
    return seq_dict, seq_chain_type



def get_input_element(mouse_aa_h, mouse_aa_l, pad_region=0):

    h_seq_dict, h_chain_type = get_pad_seq(mouse_aa_h)
    l_seq_dict, l_chain_type = get_pad_seq(mouse_aa_l)

    h_cdr_region = deepcopy(HEAVY_REGION_INDEX)
    l_cdr_region = deepcopy(LIGHT_REGION_INDEX)

    h_pad_region = torch.tensor(h_cdr_region)
    l_pad_region = torch.tensor(l_cdr_region) + pad_region
    h_l_pad_region = torch.cat((h_pad_region, l_pad_region), dim=0)

    h_pad_initial_seq = ['-'] * len(HEAVY_CDR_INDEX)
    for key, value in h_seq_dict.items():
        try:
            pos_idx = HEAVY_POSITIONS_dict[key]
            h_pad_initial_seq[pos_idx] = value
        except KeyError:
            nkey = re.findall(r'\d+', key)
            nkey = int(nkey[0])
            if (27 <= nkey <= 38) or (56 <= nkey <= 65) or ( 105 <= nkey <= 117):
                print("Heavy CDR has problem.")
            else:
                print('H Position {} is not in predefine dict, which can be ignored.'.format(key))

    l_pad_initial_seq = ['-'] * len(LIGHT_CDR_INDEX)
    for key, value in l_seq_dict.items():
        try:
            pos_idx = LIGHT_POSITIONS_dict[key]
            l_pad_initial_seq[pos_idx] = value
        except KeyError:
            nkey = re.findall(r'\d+', key)
            nkey = int(nkey[0])
            if (27 <= nkey <= 38) or (56 <= nkey <= 65) or ( 105 <= nkey <= 117):
                print("Light CDR has problem.")
            else:
                print('L Position {} is not in predefine dict, which can be ignored.'.format(key))

    chain_type = [h_chain_type, l_chain_type]
    # batch.
    h_l_ms_batch = torch.tensor([0, 1])
    h_l_pad_initial_seq = h_pad_initial_seq + l_pad_initial_seq

    return (h_l_pad_region,
            h_l_pad_initial_seq,
            chain_type, h_l_ms_batch)


def batch_input_element(mouse_sq_h, mouse_sq_l, batch_size=10, pad_region=0, finetune=False):
    (h_l_pad_region,
     h_l_pad_initial_seq,
     chain_type, h_l_ms_batch) = get_input_element(mouse_sq_h, mouse_sq_l, pad_region)

    # Get mask. Do not change CDR region.
    if not finetune:
        h_l_mask = torch.tensor(HEAVY_CDR_INDEX+LIGHT_CDR_INDEX) == 0
        h_l_loc = np.arange(len(HEAVY_CDR_INDEX)+len(LIGHT_CDR_INDEX))
        print(len(h_l_loc))
    else:
        # h_l_mask = torch.tensor(HEAVY_CDR_KABAT_VERNIER+LIGHT_CDR_KABAT_VERNIER) == 0
        h_l_mask = torch.tensor(HEAVY_CDR_KABAT_NO_VERNIER+LIGHT_CDR_KABAT_NO_VERNIER) == 0
        h_l_loc = np.arange(len(HEAVY_CDR_INDEX)+len(LIGHT_CDR_INDEX))
    # h_l_loc = h_l_loc[h_l_mask]

    # initial mask.
    ms_tokenizer = Tokenizer()
    h_l_ms_pad_seq_tokenize = ms_tokenizer.seq2idx(h_l_pad_initial_seq)

    # Here is noly consider the finetune situation.
    if finetune:
        h_l_x_pad_mask = h_l_ms_pad_seq_tokenize == ms_tokenizer.idx_pad
        h_l_x_pad_mask = h_l_x_pad_mask * h_l_mask
        h_l_mask = h_l_mask.to(torch.int) - h_l_x_pad_mask.to(torch.int)
        h_l_mask = h_l_mask.bool()

    h_l_ms_pad_seq_tokenize[h_l_mask] = ms_tokenizer.idx_msk
    h_l_loc = h_l_loc[h_l_mask]

    h_l_ms_pad_region = h_l_pad_region.unsqueeze(0).expand(batch_size, -1).clone()
    h_l_ms_pad_seq_tokenize = h_l_ms_pad_seq_tokenize.unsqueeze(0).expand(batch_size, -1).clone()
    chain_type = torch.tensor([ms_tokenizer.chain_type_idx(c) for c in chain_type])
    chain_type = chain_type.view(-1, 1).repeat(1, batch_size).view(-1)
    h_l_ms_batch = h_l_ms_batch.view(-1, 1).repeat(1, batch_size).view(-1)

    return h_l_ms_pad_seq_tokenize, h_l_ms_pad_region, chain_type, \
        h_l_ms_batch, h_l_loc, ms_tokenizer


def batch_equal_input_element(mouse_sq_h, mouse_sq_l, batch_size=10, pad_region=0):
    (h_l_pad_region,
     h_l_pad_initial_seq,
     chain_type, h_l_ms_batch) = get_input_element(mouse_sq_h, mouse_sq_l, pad_region)

    # Get mask. Do not change CDR region.
    h_l_mask = torch.tensor(HEAVY_CDR_INDEX+LIGHT_CDR_INDEX) == 0
    # initial mask.
    ms_tokenizer = Tokenizer()
    h_l_ms_pad_seq_tokenize = ms_tokenizer.seq2idx(h_l_pad_initial_seq)
    no_pad_h_l_mask = h_l_ms_pad_seq_tokenize != ms_tokenizer.idx_pad
    h_l_mask = h_l_mask * no_pad_h_l_mask

    h_l_loc = np.arange(len(HEAVY_CDR_INDEX)+len(LIGHT_CDR_INDEX))
    h_l_loc = h_l_loc[h_l_mask]
    h_l_ms_pad_seq_tokenize[h_l_mask] = ms_tokenizer.idx_msk

    h_l_ms_pad_region = h_l_pad_region.unsqueeze(0).expand(batch_size, -1).clone()
    h_l_ms_pad_seq_tokenize = h_l_ms_pad_seq_tokenize.unsqueeze(0).expand(batch_size, -1).clone()
    chain_type = torch.tensor([ms_tokenizer.chain_type_idx(c) for c in chain_type])
    chain_type = chain_type.view(-1, 1).repeat(1, batch_size).view(-1)
    h_l_ms_batch = h_l_ms_batch.view(-1, 1).repeat(1, batch_size).view(-1)

    return h_l_ms_pad_seq_tokenize, h_l_ms_pad_region, chain_type, \
        h_l_ms_batch, h_l_loc, ms_tokenizer


def graft_chain(seq):
    """
    Need to graft CDR region to a similar human germline, and return the seq dict.
    :param seq:
    :return:
    """
    seq_chain = Chain(seq, scheme='imgt')
    graft_chain = seq_chain.graft_cdrs_onto_human_germline()
    align = seq_chain.align(graft_chain)
    identity_pos_list = []
    for pos in align.positions:
        if not pos.is_in_cdr():
            a1, a2 = align[pos]
            if a1 == a2:
                identity_pos_list.append(str(pos)[1:])
        else:
            identity_pos_list.append(str(pos)[1:])
    seq_dict, chain_type = get_pad_seq(graft_chain.seq)
    return seq_dict, identity_pos_list, chain_type


def get_inpaint_input(mouse_aa_h, mouse_aa_l, pad_region=0):
    """
    :param mouse_aa_h: AA seq.
    :param mouse_aa_l: AA seq.
    :param pad_region: Region index need to add for Light chain.
    :return:
    """
    graft_hseq_dict, identity_h_list, h_chain_type = graft_chain(mouse_aa_h)
    graft_lseq_dict, identity_l_list, l_chain_type = graft_chain(mouse_aa_l)

    h_cdr_region = deepcopy(HEAVY_REGION_INDEX)
    l_cdr_region = deepcopy(LIGHT_REGION_INDEX)

    h_pad_region = torch.tensor(h_cdr_region)
    l_pad_region = torch.tensor(l_cdr_region) + pad_region
    h_l_pad_region = torch.cat((h_pad_region, l_pad_region), dim=0)

    h_pad_initial_seq = ['-'] * len(HEAVY_CDR_INDEX)
    for key, value in graft_hseq_dict.items():
        try:
            if key in identity_h_list:
                pos_idx = HEAVY_POSITIONS_dict[key]
                h_pad_initial_seq[pos_idx] = value
        except KeyError:
            nkey = re.findall(r'\d+', key)
            nkey = int(nkey[0])
            if (27 <= nkey <= 38) or (56 <= nkey <= 65) or (105 <= nkey <= 117):
                logger.info("Heavy CDR has problem.")
            else:
                logger.info('H Position {} is not in predefine dict,' \
                            'which can be ignored.'.format(key))

    l_pad_initial_seq = ['-'] * len(LIGHT_CDR_INDEX)
    for key, value in graft_lseq_dict.items():
        try:
            if key in identity_l_list:
                pos_idx = LIGHT_POSITIONS_dict[key]
                l_pad_initial_seq[pos_idx] = value
        except KeyError:
            nkey = re.findall(r'\d+', key)
            nkey = int(nkey[0])
            if (27 <= nkey <= 38) or (56 <= nkey <= 65) or ( 105 <= nkey <= 117):
                logger.info("Light CDR has problem.")
            else:
                logger.info('L Position {} is not in predefine dict,' \
                            'which can be ignored.'.format(key))

    chain_type = [h_chain_type, l_chain_type]
    # batch.
    graft_h_l_batch = torch.tensor([0, 1])
    h_l_pad_initial_seq = h_pad_initial_seq + l_pad_initial_seq
    return (h_l_pad_region,
            h_l_pad_initial_seq,
            chain_type, graft_h_l_batch)


def batch_inpaint_input_element(mouse_sq_h, mouse_sq_l, batch_size=1, pad_region=0):
    (h_l_pad_region,
     graft_h_l_pad_initial_seq,
     chain_type, graft_h_l_batch) = get_inpaint_input(mouse_sq_h, mouse_sq_l, pad_region)

    # Get mask. Do not change CDR region and germline identity region.
    h_l_mask = torch.tensor(HEAVY_CDR_INDEX+LIGHT_CDR_INDEX) == 0

    ms_tokenizer = Tokenizer()
    graft_h_l_pad_seq_tokenize = ms_tokenizer.seq2idx(graft_h_l_pad_initial_seq)
    graft_pad_h_l_mask = graft_h_l_pad_seq_tokenize == ms_tokenizer.idx_pad
    # Only consider those positions not identity.
    h_l_mask = h_l_mask * graft_pad_h_l_mask
    h_l_loc = np.arange(len(HEAVY_CDR_INDEX) + len(LIGHT_CDR_INDEX))
    h_l_loc = h_l_loc[h_l_mask]
    graft_h_l_pad_seq_tokenize[h_l_mask] = ms_tokenizer.idx_msk

    h_l_graft_pad_region = h_l_pad_region.unsqueeze(0).expand(batch_size, -1).clone()
    graft_h_l_pad_seq_tokenize = graft_h_l_pad_seq_tokenize.unsqueeze(0).expand(batch_size, -1).clone()
    chain_type = torch.tensor([ms_tokenizer.chain_type_idx(c) for c in chain_type])
    chain_type = chain_type.view(-1, 1).repeat(1, batch_size).view(-1)
    graft_h_l_batch = graft_h_l_batch.view(-1, 1).repeat(1, batch_size).view(-1)

    return graft_h_l_pad_seq_tokenize, h_l_graft_pad_region, chain_type, \
        graft_h_l_batch, h_l_loc, ms_tokenizer



def get_mouse_line(fpath):
    df_humanization = pd.read_csv(fpath)
    mouse_df = df_humanization[df_humanization['type'] == 'mouse']
    return mouse_df

# Def a read function. only output the sample human result.
def out_humanization_df(path):
    sample_df = pd.read_csv(path)
    human_df = sample_df[sample_df['Specific'] == 'humanization'].reset_index()
    return human_df


def save_seq_to_fasta(save_dir, save_df, species):
    for idx, line in save_df.iterrows():
        hseq=line['hseq']
        lseq=line['lseq']
        h_chain = Chain(hseq, scheme='imgt')
        l_chain = Chain(lseq, scheme='imgt')
        h_chain.name = f'{idx}_{species}_H'
        l_chain.name = f'{idx}_{species}_L'
        save_fpath = os.path.join(save_dir, f'{idx}_{species}.fasta')
        with open(save_fpath, 'w') as f:
            Chain.to_fasta(h_chain, f, description='VH')
            Chain.to_fasta(l_chain, f, description='VL')


def split_fasta_for_save(fpath):
    sample_human_df = out_humanization_df(fpath)

    # Create file for save fa and pdb
    fa_fpath = os.path.join(os.path.dirname(fpath), 'sample_human_fa')
    pdb_fpath = os.path.join(os.path.dirname(fpath), 'sample_human_pdb')
    os.makedirs(fa_fpath, exist_ok=True)
    os.makedirs(pdb_fpath, exist_ok=True)

    save_seq_to_fasta(fa_fpath, sample_human_df, 'human')


def select_the_most_similarity_seq(ref_h_seq, ref_l_seq, h_untokenized, l_untokenized):
    identity_list = []
    ref_h_chain = Chain(ref_h_seq, scheme='imgt')
    ref_l_chain = Chain(ref_l_seq, scheme='imgt')

    for hseq, lseq in zip(h_untokenized, l_untokenized):
        test_h_chain, test_l_chain = Chain(hseq, scheme='imgt'), Chain(lseq, scheme='imgt')
        sample_h_identity = cal_all_preservation(ref_h_chain, test_h_chain)
        sample_l_identity = cal_all_preservation(ref_l_chain, test_l_chain)
        mean_identity = (sample_h_identity + sample_l_identity) / 2
        identity_list.append(mean_identity)
    identity_max_value = max(identity_list)
    seq_idx = identity_list.index(identity_max_value)
    identity_h_seq = h_untokenized[seq_idx]
    identity_l_seq = l_untokenized[seq_idx]
    return identity_h_seq, identity_l_seq


def cdr_pair_grafting(mouse_h_seq, mouse_l_seq, back_mutation=False, scheme='kabat'):
    mouse_h_chain = Chain(mouse_h_seq, scheme=scheme)
    mouse_l_chain = Chain(mouse_l_seq, scheme=scheme)

    graft_h_chain = mouse_h_chain.graft_cdrs_onto_human_germline(backmutate_vernier=back_mutation)
    graft_l_chain = mouse_l_chain.graft_cdrs_onto_human_germline(backmutate_vernier=back_mutation)
    return graft_h_chain.seq, graft_l_chain.seq


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        default='checkpoints/antibody/hudiffab.pt'
                    )
    parser.add_argument('--ckpt_version', type=str,
                        default='finetune', choices=['pretrain', 'finetune']
                        )
    parser.add_argument('--data_fpath', type=str,
                        default ='humanization_pair_data_filter.csv'
                    )
    parser.add_argument('--batch_size', type=int,
                        default=1
                    )
    parser.add_argument('--sample_number', type=int,
                        default=1
                        )
    parser.add_argument('--try_number', type=int,
                        default=1
                        )
    parser.add_argument('--seed', type=int,
                        default=2023
                        )
    parser.add_argument('--sample_order', type=str,
                        default='shuffle')
    parser.add_argument('--sample_method', type=str,
                        default='FR', choices=['FR', 'inpaint'])
    parser.add_argument('--similarity_search', type=bool, default=True)
    parser.add_argument('--length_limit', type=str,
                        default='not_equal')
    parser.add_argument('--sample_type', type=str,
                        default='pair')
    parser.add_argument('--fa_version', type=str,
                        default='v007')
    parser.add_argument('--structure', type=eval,
                        default=False)
    parser.add_argument('--traditional_method', type=bool,
                        default=False)
    parser.add_argument('--back_mutation', type=bool,
                        default=True)
    args = parser.parse_args()

    batch_size = args.batch_size
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    seed_all(args.seed)

    # Make sure the name of sample log.
    if 'humab' in args.data_fpath:
        data_sample = 'humab'
    elif 'putative' in args.data_fpath:
        data_sample = 'putative'
    else:
        data_sample = 'lab'
    sample_tag = f'{args.seed}_{args.sample_order}_{data_sample}_{args.ckpt_version}_search_simi_{args.similarity_search}'

    if not args.traditional_method:
        
        # log dir
        log_path = os.path.dirname(os.path.dirname(args.ckpt))
        # log_path = os.path.dirname(args.ckpt)
        log_dir = get_new_log_dir(
            root=log_path,
            prefix=sample_tag
        )
        logger = get_logger('test', log_dir)
        # load model check point.

        ckpt = torch.load(args.ckpt, map_location='cpu')
        if args.ckpt_version == 'pretrain':
            config = ckpt['config']
            finetune = False
        elif args.ckpt_version == 'finetune':
            config = ckpt['pretrain_config']
            finetune = True
        else:
            print(f'ckpt version has not existed.')

        model = model_selected(config).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        logger.info(args.ckpt)
        logger.info(args.seed)

        if config.model.n_region > 7:
            pad_region = 7
        else:
            pad_region = 0

        # save path
        save_fpath = os.path.join(log_dir, 'sample_humanization_result.csv')
        with open(save_fpath, 'a', encoding='UTF-8') as f:
            f.write('Specific,name,hseq,lseq,\n')

        wrong_idx_list = []
        length_not_equal_list = []
        mouse_df = get_mouse_line(args.data_fpath)
        for idx, mouse_line in tqdm(enumerate(mouse_df.itertuples()), total=len(mouse_df.index)):
            sample_number = args.sample_number
            try_num = args.try_number
            mouse_aa_h = mouse_line.h_seq
            mouse_aa_l = mouse_line.l_seq
            if args.sample_method == 'FR':
                (h_l_pad_seq_sample, h_l_pad_seq_region,
                chain_type, h_l_ms_batch, h_l_loc, ms_tokenizer) = batch_input_element(mouse_aa_h, mouse_aa_l,
                                                                                        batch_size, pad_region, finetune=finetune)
            elif args.sample_method == 'inpaint':
                (h_l_pad_seq_sample, h_l_pad_seq_region,
                chain_type, h_l_ms_batch, h_l_loc, ms_tokenizer) = batch_inpaint_input_element(mouse_aa_h, mouse_aa_l,
                                                                                            batch_size, pad_region)
            else:
                print('Sample Method has problem.')

            # Adding the raw mouse line.
            origin = 'mouse'
            name = mouse_line.name
            with open(save_fpath, 'a', encoding='UTF-8') as f:
                f.write(f'{origin},{name},{mouse_aa_h},{mouse_aa_l}\n')

            if args.sample_order == 'shuffle':
                np.random.shuffle(h_l_loc)
            while sample_number > 0 and try_num > 0:
                all_token = ms_tokenizer.toks
                with torch.no_grad():
                    for i in tqdm(h_l_loc, total=len(h_l_loc), desc='Antibody Humanization Process'):
                        h_l_prediction = model(
                            h_l_pad_seq_sample.to(device),
                            h_l_pad_seq_region.to(device),
                            chain_type.to(device),
                            # h_l_ms_batch.to(device)
                        )

                        h_l_pred = h_l_prediction[:, i, :len(all_token)-1]
                        h_l_soft = torch.nn.functional.softmax(h_l_pred, dim=1)
                        h_l_sample = torch.multinomial(h_l_soft, num_samples=1)
                        h_l_pad_seq_sample[:, i] = h_l_sample.squeeze()

                h_pad_seq_sample = h_l_pad_seq_sample[:, :152]
                l_pad_seq_sample = h_l_pad_seq_sample[:, 152:]
                h_untokenized = [ms_tokenizer.idx2seq(s) for s in h_pad_seq_sample]
                l_untokenized = [ms_tokenizer.idx2seq(s) for s in l_pad_seq_sample]

                # Write the sample result.
                sample_origin = 'humanization'
                sample_name = str(name) + 'human_sample'
                if args.similarity_search:
                    if sample_number == 0:
                        break
                    g_h, g_l = select_the_most_similarity_seq(mouse_aa_h, mouse_aa_l, h_untokenized, l_untokenized)
                    with open(save_fpath, 'a', encoding='UTF-8') as f:
                        f.write(f'{sample_origin},{sample_name},{g_h},{g_l}\n')

                        sample_number = 0
                else:
                    for _, (g_h, g_l) in enumerate(zip(h_untokenized, l_untokenized)):
                        if sample_number == 0:
                            break
                        with open(save_fpath, 'a', encoding='UTF-8') as f:
                            f.write(f'{sample_origin},{sample_name},{g_h},{g_l}\n')

                            sample_number -= 1
        logger.info('Length did not equal list: {}'.format(length_not_equal_list))
        logger.info('Wrong idx: {}'.format(wrong_idx_list))
    else:
        # log dir
        log_dpath = os.path.dirname(args.data_fpath)
        # log_path = os.path.dirname(args.ckpt)
        new_tag = f'{data_sample}_cdr_graft_back_mutation_{args.back_mutation}'
        log_dir = get_new_log_dir(
            root=log_dpath,
            prefix=new_tag
        )

        save_fpath = os.path.join(log_dir, 'sample_humanization_result.csv')
        with open(save_fpath, 'a', encoding='UTF-8') as f:
            f.write('Specific,name,hseq,lseq,\n')

        logger = get_logger('test', log_dir)
        mouse_df = get_mouse_line(args.data_fpath)
        for idx, mouse_line in tqdm(enumerate(mouse_df.itertuples()), total=len(mouse_df.index)):
            name = mouse_line.name
            mouse_aa_h = mouse_line.h_seq
            mouse_aa_l = mouse_line.l_seq

            if not args.back_mutation:
                graft_h_seq, graft_l_seq = cdr_pair_grafting(
                    mouse_h_seq=mouse_aa_h,
                    mouse_l_seq=mouse_aa_l,
                )
            else:
                graft_h_seq, graft_l_seq = cdr_pair_grafting(
                    mouse_h_seq=mouse_aa_h,
                    mouse_l_seq=mouse_aa_l,
                    back_mutation=args.back_mutation
                )

            with open(save_fpath, 'a', encoding='UTF-8') as f:
                sample_origin = 'humanization'
                sample_name = str(name) + 'human_sample'
                f.write(f'{sample_origin},{sample_name},{graft_h_seq},{graft_l_seq}\n')


    # Save as fasta for biophi oasis.
    fasta_save_fpath = os.path.join(log_dir, 'sample_identity.fa')
    sample_df = pd.read_csv(save_fpath)
    sample_human_df = sample_df[sample_df['Specific'] == 'humanization'].reset_index(drop=True)
    trans_to_chain(sample_human_df, fasta_save_fpath, version=args.fa_version)

    # Split save as fasta for structure prediction.
    if args.structure:
        split_fasta_for_save(save_fpath)