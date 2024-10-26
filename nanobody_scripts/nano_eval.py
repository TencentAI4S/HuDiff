import os
import subprocess
from tqdm import tqdm
from abnumber import Chain
import numpy as np

import pandas as pd
from evaluation.T20_eval import frame_main as tframemain
from utils.misc import get_logger


def cal_fr_preservation(chain1, chain2):
    identity = 0
    fr_sum = 0
    align = chain1.align(chain2)
    for pos in align.positions:
        if not pos.is_in_cdr():
            a1, a2 = align[pos]
            if a1 == a2:
                identity += 1
            fr_sum += 1
    return identity / fr_sum


def cal_group_fr_germline_identity(df, scheme='imgt'):
    identity_fr_ratio_list = []
    for idx in tqdm(df.index):
        try:
            h_seq = df.iloc[idx]['h_seq']
            h_chain = Chain(h_seq, scheme=scheme)
            h_chain_graft = h_chain.graft_cdrs_onto_human_germline()
            fr_h_ratio = cal_fr_preservation(h_chain, h_chain_graft)
            identity_fr_ratio_list.append(fr_h_ratio)
        finally:
            continue

    return identity_fr_ratio_list


def cal_group_fr_germline_identity_for_sample(df, scheme='imgt'):
    identity_fr_ratio_list = []
    for idx in tqdm(df.index):
        try:
            h_seq = df.iloc[idx]['hseq']
            h_chain = Chain(h_seq, scheme=scheme)
            h_chain_graft = h_chain.graft_cdrs_onto_human_germline()
            fr_h_ratio = cal_fr_preservation(h_chain, h_chain_graft)
            identity_fr_ratio_list.append(fr_h_ratio)
        finally:
            continue

    return identity_fr_ratio_list


def cal_mean(vh_dir, vhh_dir):
    vh_fpath = os.path.join(vh_dir, 'sample_nano_vh_abnativ_seq_scores.csv')
    vhh_fpath = os.path.join(vhh_dir, 'sample_nano_vhh_abnativ_seq_scores.csv')

    sample_vh_df = pd.read_csv(vh_fpath)
    sample_vh_score = sample_vh_df['AbNatiV VH Score']

    sample_vhh_df = pd.read_csv(vhh_fpath)
    sample_vhh_score = sample_vhh_df['AbNatiV VHH Score']

    ref_vh_score = 0.7378085839359757
    ref_vhh_score = 0.9143594023426274

    dev_vh_score = sample_vh_score.mean() - ref_vh_score
    print('Raw sample result: {}'.format(sample_vh_score.mean()))

    return dev_vh_score, sample_vh_score.mean(), sample_vhh_score.mean()


def get_raw_frame_t20_score():
    raw_frame_t20_fpath = '/sample_frame_t20_score.csv'
    raw_frame_t20_df = pd.read_csv(raw_frame_t20_fpath)
    raw_frame_t20_mean = raw_frame_t20_df['h_score'].mean()
    return raw_frame_t20_df, raw_frame_t20_mean


def main(root_path=None):

    if root_path is None:
       
        root_path = 'sample_humanization_result.csv'


    logdir = os.path.dirname(root_path)
    logger = get_logger('sample', logdir, log_name='eval_log.txt')

    # abnativ path.
    exec_path = 'bin/abnativ'
    input_fa_path = os.path.join(os.path.dirname(root_path), 'sample_identity.fa')
    output_vh_dir = os.path.join(os.path.dirname(root_path), 'sample_nano_vh/')
    output_vhh_dir = os.path.join(os.path.dirname(root_path), 'sample_nano_vhh/')
    if not os.path.exists(output_vh_dir):
        print('Eval vh score ......')
        subprocess.Popen([exec_path, 'score', '-nat', 'VH', '-i', input_fa_path, '-odir', output_vh_dir, '-oid', 'sample_nano_vh', '-align'],
                         stderr=subprocess.PIPE, stdout=subprocess.PIPE).communicate()
    else:
        print('VH exists, Skip!')

    if not os.path.exists(output_vhh_dir):
        print('Eval vhh score ......')
        subprocess.Popen([exec_path, 'score', '-nat', 'VHH', '-i', input_fa_path, '-odir', output_vhh_dir, '-oid', 'sample_nano_vhh', '-align', '-isVHH'],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()
    else:
        print('VHH exists, Skip!')

    dev_vh, sample_ab_mean, sample_ab_vhh_mean = cal_mean(output_vh_dir, output_vhh_dir)

    # T20
    t20_frame_save_fpath = tframemain(root_path)
    sample_t20_frame_df = pd.read_csv(t20_frame_save_fpath)
    sample_frame_t20 = sample_t20_frame_df['h_score'].mean()

    raw_frame_t20_df, raw_frame_t20 = get_raw_frame_t20_score()

    # Identity
    sample_result_df = pd.read_csv(root_path)
    sample_human_df = sample_result_df[sample_result_df['Specific'] == 'humanization'].reset_index(drop=True)
    sample_identity = cal_group_fr_germline_identity_for_sample(sample_human_df, scheme='imgt')
    raw_identity = cal_group_fr_germline_identity(raw_frame_t20_df, scheme='imgt')


    logger.info('Raw Frame t20 score {}'.format(raw_frame_t20))
    logger.info('Sample Frame t20 score {}'.format(sample_frame_t20))
    logger.info('Improve Frame t20 score {}'.format(sample_frame_t20-raw_frame_t20))

    logger.info('Eval path is {}'.format(root_path))
    logger.info('VH score improve: {}'.format(dev_vh))
    logger.info('Sample VH score: {}'.format(sample_ab_mean))
    logger.info('Sample VHH score: {}'.format(sample_ab_vhh_mean))

    logger.info('Sample Germline identity(seq_num): {}({})'.format(np.array(sample_identity).mean(), len(sample_identity)))
    logger.info('Raw Germline identity(seq_num): {}({})'.format(np.array(raw_identity).mean(), len(raw_identity)))


if __name__ == '__main__':
    current_ld_library_path = os.getenv("LD_LIBRARY_PATH", "")
    os.environ['LD_LIBRARY_PATH'] = 'anaconda3/envs/abnativ/lib:' + current_ld_library_path

    main()