import os.path

import numpy as np
import torch
from tqdm import tqdm
import argparse
import pandas as pd
from abnumber import Chain
from anarci import anarci, number
from copy import deepcopy
import re
from Bio import SeqIO
import sys
current_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(current_dir)


from dataset.build_human_pair_oas_new import (region_padding_fix,
                                              HEAVY_POSITIONS_dict, LIGHT_POSITIONS_dict,
                                              HEAVY_CDR_INDEX, LIGHT_CDR_INDEX)
from dataset.oas_pair_dataset_new import light_pad_cdr, HEAVY_REGION_INDEX, LIGHT_REGION_INDEX
from sample import (get_pad_seq, get_input_element, batch_input_element)   
from utils.tokenizer import Tokenizer
from utils.train_utils import model_selected
from utils.misc import get_new_log_dir, get_logger, seed_all


REGION_LENGTH = (26, 12, 17, 10, 38, 30, 11)

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


def get_h_l_seq_from_fasta(fpath):
    """
    Split the heavy and light chain from the raw fasta file.
    :param fpath: the raw fasta file path.
    :return: heavy sequence, light sequence.
    """
    heavy_chain = None
    light_chain = None
    sequences = SeqIO.parse(fpath, 'fasta')
    for seq in sequences:
        if 'heavy chain' in seq.description:
            heavy_chain = str(seq.seq)
        elif 'light chain' in seq.description:
            light_chain = str(seq.seq)
        else:
            continue
    assert heavy_chain is not None and light_chain is not None, print("Reading the fasta has problem.")
    return heavy_chain, light_chain


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        # required=True,
                        default='./checkpoints/antibody/antibody_diff.pt'
                    )
    parser.add_argument('--anti_complex_fasta', type=str,
                        default=None
                    )
    parser.add_argument('--heavy_seq', type=str,
                        help='heavy chain sequence of antibody'
                    )
    parser.add_argument('--light_seq', type=str,
                        help='light chain sequence of antibody'
                    )
    parser.add_argument('--log_dirpath', type=str,
                        default='./tmp/'
                    )
    parser.add_argument('--batch_size', type=int,
                        default=100
                    )
    parser.add_argument('--seed', type=int,
                        default=2023
                        )
    parser.add_argument('--sample_number', type=int,
                        default=1000,
                        )
    parser.add_argument('--sample_order', type=str,
                        default='shuffle')
    parser.add_argument('--sample_type', type=str,
                        default='pair')
    args = parser.parse_args()

    batch_size = args.batch_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_all(args.seed)

    # Read the fasta file or seq.
    if args.anti_complex_fasta is not None:
        mouse_heavy, mouse_light = get_h_l_seq_from_fasta(args.anti_complex_fasta)
        pdb_name = os.path.basename(args.anti_complex_fasta).split('.')[0]
    else:
        mouse_heavy = args.heavy_seq
        mouse_light = args.light_seq
        pdb_name = 'Unkown'
    

    sample_tag = f'{pdb_name}_{args.sample_order}_{args.sample_type}'
    # log dir
    if args.log_dirpath is not None:
        log_path = args.log_dirpath
    else:
        log_path = os.path.dirname(args.anti_complex_fasta)
    log_dir = get_new_log_dir(
        root=log_path,
        prefix=sample_tag
    )
    logger = get_logger('test', log_dir)

    # load model check point.
    ckpt = torch.load(args.ckpt)
    config = ckpt['config']
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
    sample_number = args.sample_number
    mouse_aa_h = Chain(mouse_heavy, scheme='imgt').seq
    mouse_aa_l = Chain(mouse_light, scheme='imgt').seq

    origin = 'mouse'
    name = pdb_name
    with open(save_fpath, 'a', encoding='UTF-8') as f:
        f.write(f'{origin},{name},{mouse_aa_h},{mouse_aa_l}\n')

    try:
        (h_l_pad_seq_sample, h_l_pad_seq_region,
        chain_type, h_l_ms_batch, h_l_loc, ms_tokenizer) = batch_input_element(mouse_aa_h, mouse_aa_l, batch_size, pad_region)
    except:
        logger.info('This antibody encoding may have problem, please check!')

    if args.sample_order == 'shuffle':
        np.random.shuffle(h_l_loc)

    while sample_number > 0:
        all_token = ms_tokenizer.toks
        with torch.no_grad():
            for i in tqdm(h_l_loc, total=len(h_l_loc), desc='Humanization process'):
                h_l_prediction = model(
                    h_l_pad_seq_sample.to(device),
                    h_l_pad_seq_region.to(device),
                    chain_type.to(device),
                    h_l_ms_batch.to(device),
                )

                h_l_pred = h_l_prediction[:, i, :len(all_token)-1]
                h_l_soft = torch.nn.functional.softmax(h_l_pred, dim=1)
                h_l_sample = torch.multinomial(h_l_soft, num_samples=1)
                h_l_pad_seq_sample[:, i] = h_l_sample.squeeze()

        h_pad_seq_sample = h_l_pad_seq_sample[:, :152]
        l_pad_seq_sample = h_l_pad_seq_sample[:, 152:]
        h_untokenized = [ms_tokenizer.idx2seq(s) for s in h_pad_seq_sample]
        l_untokenized = [ms_tokenizer.idx2seq(s) for s in l_pad_seq_sample]

        for _, (g_h, g_l) in enumerate(zip(h_untokenized, l_untokenized)):

            if sample_number == 0:
                break

            with open(save_fpath, 'a', encoding='UTF-8') as f:
                sample_origin = 'humanization'
                sample_name = str(name) + 'human_sample'
                f.write(f'{sample_origin},{sample_name},{g_h},{g_l}\n')

                sample_number -= 1
                logger.info('Already Sample number {}'.format(args.sample_number-sample_number))
                logger.info('Sample Heavy Chain Seq: {}'.format(g_h))
                logger.info('Sample Light Chain Seq: {}'.format(g_l))

    logger.info('Length did not equal list: {}'.format(length_not_equal_list))
    logger.info('Wrong idx: {}'.format(wrong_idx_list))




