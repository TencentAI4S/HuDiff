""" This script only consider for the nanobody. """
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
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from nanosample import (batch_input_element, save_nano, seqs_to_fasta,
                        compare_length, get_diff_region_aa_seq, get_pad_seq,
                        get_input_element, get_nano_line, out_humanization_df,
                        save_seq_to_fasta, split_fasta_for_save, 
                        get_multi_model_state
                        )
from utils.tokenizer import Tokenizer
from utils.train_utils import model_selected
from utils.misc import get_new_log_dir, get_logger, seed_all

# Finetune package
from model.nanoencoder.abnativ_model import AbNatiV_Model
from model.nanoencoder.model import NanoAntiTFNet


def get_nano_seq_from_fasta(fpath):
    """
    Split the heavy and light chain from the raw fasta file.
    :param fpath: the raw fasta file path.
    :return: heavy sequence, light sequence.
    """
    nano_chain = None
    sequences = SeqIO.parse(fpath, 'fasta')
    for seq in sequences:
        if 'Nanobody' in seq.description:
            nano_chain = str(seq.seq)
        else:
            continue
    assert nano_chain is not None, print("Reading the fasta has problem.")
    return nano_chain


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This program is designed to humanize non-human nanobodies.")
    parser.add_argument('--ckpt', type=str,
                        default=None,
                        help='The ckpt path of the pretrained path.'
                    )
    parser.add_argument('--nano_complex_fasta', type=str,
                        default=None,
                        help='fasta file of the nanobody.'
                    )
    parser.add_argument('--batch_size', type=int,
                        default=10,
                        help='the batch size of sample.'
                    )
    parser.add_argument('--sample_number', type=int,
                        default=100,
                        help='The number of all sample.'
                        )
    parser.add_argument('--seed', type=int,
                        default=42
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
                        default=True)
    parser.add_argument('--structure', type=eval,
                        default=False)
    args = parser.parse_args()

    batch_size = args.batch_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # seed_all(args.seed)

    # Make sure the name of sample log.
    pdb_name = os.path.basename(args.nano_complex_fasta).split('.')[0]
    sample_tag = f'{pdb_name}_{args.model}_vhh'

    # log dir
    log_path = os.path.dirname(args.nano_complex_fasta)
    log_dir = get_new_log_dir(
        root=log_path,
        prefix=sample_tag
    )
    logger = get_logger('test', log_dir)

    # Here we specify the finetune model to generate the humanization seq.
    ckpt = torch.load(args.ckpt)
    config = ckpt['config']
    abnativ_state, _, infilling_state = get_multi_model_state(ckpt)
    # Abnativ model.
    hparams = ckpt['abnativ_params']
    abnativ_model = AbNatiV_Model(hparams)
    abnativ_model.load_state_dict(abnativ_state)
    abnativ_model.to(device)
    # infilling model.
    # infilling_params = config.model
    infilling_params = ckpt['infilling_params']
    infilling_model = NanoAntiTFNet(**infilling_params)
    infilling_model.load_state_dict(infilling_state)
    infilling_model.to(device)

    # Carefull!!! tmp
    config.model['equal_weight'] = True
    config.model['vhh_nativeness'] = False
    config.model['human_threshold'] = None
    config.model['human_all_seq'] = False
    config.model['temperature'] = False

    model_dict = {
        'abnativ': abnativ_model,
        'infilling': infilling_model,
        'target_infilling': infilling_model
    }
    framework_model = model_selected(config, pretrained_model=model_dict, tokenizer=Tokenizer())
    model = framework_model.infilling_pretrain
    model.eval()
    
    logger.info(args.ckpt)
    logger.info(args.seed)


    # Read the fasta file of nanobody.
    nano_chain = get_nano_seq_from_fasta(args.nano_complex_fasta)

    # save path
    save_fpath = os.path.join(log_dir, 'sample_humanization_result.csv')
    origin = 'Nano'
    with open(save_fpath, 'a', encoding='UTF-8') as f:
        f.write('Specific,name,hseq,\n')
        f.write(f'{origin},{pdb_name},{nano_chain}\n')

    wrong_idx_list = []
    length_not_equal_list = []
    sample_number = args.sample_number


    try:
        nano_pad_token, nano_pad_region, nano_loc, ms_tokenizer = batch_input_element(
                                                                                    nano_chain,
                                                                                    inpaint_sample=args.inpaint_sample,
                                                                                    batch_size=batch_size
                                                                                    )
    except:
        logger.info('This nanobody encoding may have problem, please check!')

    if args.sample_order == 'shuffle':
        np.random.shuffle(nano_loc)

    duplicated_set = set()

    while sample_number > 0:
        all_token = ms_tokenizer.toks
        with torch.no_grad():
            for i in tqdm(nano_loc, total=len(nano_loc), desc='Nanobody Humanization Process'):
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
                # try:
                sample_origin = 'humanization'
                sample_name = str(pdb_name)
                # Make sure that the sample seq can be detected by the Chain.
                # Duplicated.
                if g_h not in duplicated_set:
                    test_chain = Chain(g_h, scheme='imgt')
                    f.write(f'{sample_origin},{sample_name},{g_h}\n')
                    duplicated_set.add(g_h)
                    sample_number -= 1
                    logger.info('Already Sample number {}'.format(args.sample_number - sample_number))
                    logger.info('Sample Heavy Chain Seq: {}'.format(g_h))
                else:
                    sample_number -= 1

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




