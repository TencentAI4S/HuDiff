import os.path
import subprocess
from tqdm import tqdm
from abnumber import Chain
import numpy as np
import sys
current_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(current_dir)

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# from ABLSTM_eval import main as abmain
# from humab_eval import main as hubmain
# from Zscore_eval import main as zmain
from evaluation.T20_eval import main as tmain
from utils.misc import get_logger


def cal_fr_mutation_precision(expchain, parental, test):
    share_precision = 0
    base_precision = 0
    only_precision = 0
    # mutation_sum = 0
    # share_sum = 0
    align = expchain.align(parental, test)
    for pos in align.positions:
        # if not pos.is_in_cdr():
        exp, mou, aa = align[pos]
        if exp != mou or aa != mou:
            if exp == aa:
                share_precision += 1
            elif exp != aa:
                if exp != mou:
                    base_precision += 1
                if aa != mou:
                    only_precision += 1

    if share_precision + only_precision == 0:
        return None
    else:
        return share_precision / (share_precision + only_precision)


def cal_group_fr_precision(exp_df, mou_df, sample_df, scheme='kabat'):
    fr_ratio_list_h = []
    fr_ratio_list_l = []
    for idx in tqdm(exp_df.index):
        exp_h_seq, exp_l_seq = exp_df.iloc[idx]['h_seq'], exp_df.iloc[idx]['l_seq']
        mou_h_seq, mou_l_seq = mou_df.iloc[idx]['h_seq'], mou_df.iloc[idx]['l_seq']
        sap_h_seq, sap_l_seq = sample_df.iloc[idx]['hseq'], sample_df.iloc[idx]['lseq']
        exp_h_chain, exp_l_chain = Chain(exp_h_seq, scheme=scheme), Chain(exp_l_seq, scheme=scheme)
        mou_h_chain, mou_l_chain = Chain(mou_h_seq, scheme=scheme), Chain(mou_l_seq, scheme=scheme)
        sap_h_chain, sap_l_chain = Chain(sap_h_seq, scheme=scheme), Chain(sap_l_seq, scheme=scheme)
        h_share_ratio = cal_fr_mutation_precision(exp_h_chain, mou_h_chain, sap_h_chain)
        l_share_ratio = cal_fr_mutation_precision(exp_l_chain, mou_l_chain, sap_l_chain)
        if h_share_ratio is not None:
            fr_ratio_list_h.append(h_share_ratio)
        if l_share_ratio is not None:
            fr_ratio_list_l.append(l_share_ratio)
        # vernier_ratio_list.append([h_share_ratio, l_share_ratio])
    return fr_ratio_list_h, fr_ratio_list_l


def cal_vernier_mutation_precision(expchain, parental, test):
    assert expchain.scheme == 'kabat', print('Only support the Kabat scheme.')
    share_precision = 0
    base_precision = 0
    only_precision = 0
    # mutation_sum = 0
    # share_sum = 0
    align = expchain.align(parental, test)
    for pos in align.positions:
        if pos.is_in_vernier():
            exp, mou, aa = align[pos]
            # print(exp, mou, aa)
            if exp != mou or aa != mou:
                if exp == aa:
                    share_precision += 1
                elif exp != aa:
                    if exp != mou:
                        base_precision += 1
                    if aa != mou:
                        only_precision += 1
    # print(share_precision)
    # print(only_precision)
    if share_precision + only_precision == 0:
        return None
    else:
        return share_precision / (share_precision + only_precision)


def cal_vernier_identity_precision(expchain, parental, test):
    assert expchain.scheme == 'kabat', print('Only support the Kabat scheme.')
    share_precision = 0
    base_precision = 0
    only_precision = 0
    # mutation_sum = 0
    # share_sum = 0
    align = expchain.align(parental, test)
    for pos in align.positions:
        if pos.is_in_vernier():
            exp, mou, aa = align[pos]
            print(exp, mou, aa)
            if exp == aa:
                share_precision += 1
            elif exp != aa:
                if exp != mou:
                    base_precision += 1
                if aa != mou:
                    only_precision += 1
    # print(share_precision)
    # print(only_precision)
    if share_precision + only_precision == 0:
        return None
    else:
        return share_precision / (share_precision + only_precision)


def cal_group_vernier_precision(exp_df, mou_df, sample_df, scheme='kabat'):
    vernier_ratio_list_h = []
    vernier_ratio_list_l = []
    for idx in tqdm(exp_df.index):
        exp_h_seq, exp_l_seq = exp_df.iloc[idx]['h_seq'], exp_df.iloc[idx]['l_seq']
        mou_h_seq, mou_l_seq = mou_df.iloc[idx]['h_seq'], mou_df.iloc[idx]['l_seq']
        sap_h_seq, sap_l_seq = sample_df.iloc[idx]['hseq'], sample_df.iloc[idx]['lseq']
        exp_h_chain, exp_l_chain = Chain(exp_h_seq, scheme=scheme), Chain(exp_l_seq, scheme=scheme)
        mou_h_chain, mou_l_chain = Chain(mou_h_seq, scheme=scheme), Chain(mou_l_seq, scheme=scheme)
        sap_h_chain, sap_l_chain = Chain(sap_h_seq, scheme=scheme), Chain(sap_l_seq, scheme=scheme)
        h_share_ratio = cal_vernier_mutation_precision(exp_h_chain, mou_h_chain, sap_h_chain)
        l_share_ratio = cal_vernier_mutation_precision(exp_l_chain, mou_l_chain, sap_l_chain)
        if h_share_ratio is not None:
            vernier_ratio_list_h.append(h_share_ratio)
        if l_share_ratio is not None:
            vernier_ratio_list_l.append(l_share_ratio)
        # vernier_ratio_list.append([h_share_ratio, l_share_ratio])
    return vernier_ratio_list_h, vernier_ratio_list_l


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


def cal_all_preservation(chain1, chain2):
    identity = 0
    fr_sum = 0
    align = chain1.align(chain2)
    for pos in align.positions:
        a1, a2 = align[pos]
        if a1 == a2:
            identity += 1
        fr_sum += 1
    return identity / fr_sum


def cal_vernier_preservation(chain1, chain2):
    identity = 0
    fr_sum = 0
    align = chain1.align(chain2)
    for pos in align.positions:
        if pos.is_in_vernier():
            a1, a2 = align[pos]
            if a1 == a2:
                identity += 1
            fr_sum += 1
    return identity / fr_sum


def cal_group_all_perservation(human_df, mouse_df, scheme='imgt', idx_type='lab'):
    if idx_type == 'lab':
        h_idx = 'h_seq'
        l_idx = 'l_seq'
    else:
        h_idx = 'hseq'
        l_idx = 'lseq'
    preservation_all_ratio_list = []
    preservation_vernier_ratio_list = []
    for idx in tqdm(human_df.index):
        human_h_seq, human_l_seq = human_df.iloc[idx][h_idx], human_df.iloc[idx][l_idx]
        human_h_chain, human_l_chain = Chain(human_h_seq, scheme=scheme), Chain(human_l_seq, scheme=scheme)

        mouse_h_seq, mouse_l_seq = mouse_df.iloc[idx]['h_seq'], mouse_df.iloc[idx]['l_seq']
        mouse_h_chain, mouse_l_chain = Chain(mouse_h_seq, scheme=scheme), Chain(mouse_l_seq, scheme=scheme)

        h_per_all_ratio = cal_all_preservation(human_h_chain, mouse_h_chain)
        l_per_all_ratio = cal_all_preservation(human_l_chain, mouse_l_chain)

        h_per_vernier_ratio = cal_vernier_preservation(human_h_chain, mouse_h_chain)
        l_per_vernier_ratio = cal_vernier_preservation(human_l_chain, mouse_l_chain)

        preservation_all_ratio_list.append([h_per_all_ratio, l_per_all_ratio])
        preservation_vernier_ratio_list.append([h_per_vernier_ratio, l_per_vernier_ratio])

    return preservation_all_ratio_list, preservation_vernier_ratio_list


def cal_group_fr_germline_identity(df, scheme='imgt'):
    identity_fr_ratio_list = []
    for idx in tqdm(df.index):
        h_seq, l_seq = df.iloc[idx]['h_seq'], df.iloc[idx]['l_seq']
        h_chain, l_chain = Chain(h_seq, scheme=scheme), Chain(l_seq, scheme=scheme)
        h_chain_graft = h_chain.graft_cdrs_onto_human_germline()
        l_chain_graft = l_chain.graft_cdrs_onto_human_germline()
        fr_h_ratio = cal_fr_preservation(h_chain, h_chain_graft)
        fr_l_ratio = cal_fr_preservation(l_chain, l_chain_graft)
        identity_fr_ratio_list.append([fr_h_ratio, fr_l_ratio])
    return identity_fr_ratio_list


def raw_mouse_t20():
    lab_mouse_t20_dirpath = '/sample_mouse_t20_score.csv'
    lab_mouse_t20_df = pd.read_csv(lab_mouse_t20_dirpath)
    mouse_t20_h_mean = lab_mouse_t20_df['h_score'].mean()
    mouse_t20_l_mean = lab_mouse_t20_df['l_score'].mean()
    return mouse_t20_h_mean, mouse_t20_l_mean


def exp_human_t20():
    lab_exper_t20_dirpath = '/sample_experimental_t20_score.csv'
    lab_t20_df = pd.read_csv(lab_exper_t20_dirpath)
    exp_t20_h_mean = lab_t20_df['h_score'].mean()
    exp_t20_l_mean = lab_t20_df['l_score'].mean()
    return exp_t20_h_mean, exp_t20_l_mean


def main(root_path):
    """
    Evaluating the metrics of all.
    :param root_path: sample file path.
    :return:
    """
    logdir = os.path.dirname(root_path)
    logger = get_logger('sample', logdir, log_name='eval_log.txt')


    mouse_h_t20, mouse_l_t20 = raw_mouse_t20()
    exp_h_t20, exp_l_t20 = exp_human_t20()

    t20_save_fpath = tmain(root_path)
    sample_t20_df = pd.read_csv(t20_save_fpath)
    sample_h_t20 = sample_t20_df['h_score'].mean()
    sample_l_t20 = sample_t20_df['l_score'].mean()

    logger.info('Experimental_H_score_improvement: {}'.format(exp_h_t20 - mouse_h_t20))
    logger.info('Experimental_L_score_improvement: {}'.format(exp_l_t20 - mouse_l_t20))

    logger.info('Sample_H_score_improvement: {}'.format(sample_h_t20 - mouse_h_t20))
    logger.info('Sample_L_score_improvement: {}'.format(sample_l_t20 - mouse_l_t20))

    # Oasis
    exec_path = 'bin/biophi'
    input_fa_path = os.path.join(os.path.dirname(root_path), 'sample_identity.fa')
    oasis_db = '/BioPhi/OASis_9mers_v1.db'
    out_put_oasis = os.path.join(os.path.dirname(root_path), 'sample_identity_oasis.xlsx')
    if not os.path.exists(out_put_oasis):
    # biophi oasis sample_identity.fa --oasis-db /data/home/waitma/antibody_proj/BioPhi-2021-publication/OASis_9mers_v1.db --output sample_identity_oasis.xlsx
        subprocess.Popen([exec_path, 'oasis', input_fa_path, '--oasis-db', oasis_db, '--output', out_put_oasis],
                            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()
    else:
        print('Oasis exists, Skip!')

    lab_mouse_dirpath = '/parental_oasis.xlsx'
    lab_exper_dirpath = '/experimental_osis.xlsx'

    mouse_oasis_curve = pd.read_excel(lab_mouse_dirpath, sheet_name='OASis Curves', index_col=0)
    labexp_oasis_curve = pd.read_excel(lab_exper_dirpath, sheet_name='OASis Curves', index_col=0)

    sample_oasis_curve = pd.read_excel(out_put_oasis, sheet_name='OASis Curves', index_col=0)

    logger.info('Lab oasis medium improve: {}'.format(labexp_oasis_curve['50%'].mean() - mouse_oasis_curve['50%'].mean()))
    logger.info('Sample oasis medium improve: {}'.format(sample_oasis_curve['50%'].mean() - mouse_oasis_curve['50%'].mean()))

    # # Identity.
    identity_ratio = cal_group_fr_germline_identity(sample_t20_df, scheme='imgt')
    logger.info('H germline identity: {}'.format(np.array(identity_ratio)[:, 0].mean()))
    logger.info('L germline identity: {}'.format(np.array(identity_ratio)[:, 1].mean()))

    # Preservation
    sample_df = pd.read_csv(root_path)
    sample_human_df = sample_df[sample_df['Specific'] == 'humanization'].reset_index(drop=True)

    humab_exp_fpath = '/experimental_humanized.csv'
    humab_mou_fpath = '/parental_mouse.csv'
    # lab_df = pd.read_csv(lab_human_fpath)
    lab_human_df = pd.read_csv(humab_exp_fpath)
    lab_mouse_df = pd.read_csv(humab_mou_fpath)

    exp_preservation_all_list, exp_preservation_vernier_list = cal_group_all_perservation(lab_human_df, lab_mouse_df, scheme='kabat', idx_type='lab')
    logger.info('H Experimental all Preservation: {}'.format(np.array(exp_preservation_all_list)[:, 0].mean()))
    logger.info('L Experimental all Preservation: {}'.format(np.array(exp_preservation_all_list)[:, 1].mean()))
    print('--------------------------------------------')
    logger.info('H Experimental vernier Preservation: {}'.format(np.array(exp_preservation_vernier_list)[:, 0].mean()))
    logger.info('L Experimental vernier Preservation: {}'.format(np.array(exp_preservation_vernier_list)[:, 1].mean()))

    sample_preservation_all_list, sample_preservation_vernier_list = cal_group_all_perservation(sample_human_df, lab_mouse_df, scheme='kabat', idx_type='sap')
    logger.info('H Sample all Preservation: {}'.format(np.array(sample_preservation_all_list)[:, 0].mean()))
    logger.info('L Sample all Preservation: {}'.format(np.array(sample_preservation_all_list)[:, 1].mean()))
    print('---------------------------------------------')
    logger.info('H Sample vernier Preservation: {}'.format(np.array(sample_preservation_vernier_list)[:, 0].mean()))
    logger.info('L Sample vernier Preservation: {}'.format(np.array(sample_preservation_vernier_list)[:, 1].mean()))

    #  Vernier mutation precision.
    vernier_ratio_list_h, vernier_ratio_list_l = cal_group_vernier_precision(lab_human_df, lab_mouse_df, sample_human_df, scheme='kabat')
    print(len(vernier_ratio_list_h))
    print(len(vernier_ratio_list_l))
    logger.info('H Vernier Mutation precision: {}'.format(np.array(vernier_ratio_list_h).mean()))
    logger.info('L Vernier Mutation precision: {}'.format(np.array(vernier_ratio_list_l).mean()))
    logger.info('Vernier Mutation precision Mean: {}'.format((np.array(vernier_ratio_list_h).mean()
                                                              + np.array(vernier_ratio_list_l).mean())/2))

    # Fr mutation precision.
    fr_ratio_list_h, fr_ratio_list_l = cal_group_fr_precision(lab_human_df, lab_mouse_df, sample_human_df, scheme='kabat')
    print(len(fr_ratio_list_h))
    print(len(fr_ratio_list_l))
    logger.info('H fr Mutation precision: {}'.format(np.array(fr_ratio_list_h).mean()))
    logger.info('L fr Mutation precision: {}'.format(np.array(fr_ratio_list_l).mean()))
    logger.info('Frmutation precision: {}'.format((np.array(fr_ratio_list_h).mean()
                                                   + np.array(fr_ratio_list_l).mean())/2))

if __name__ == '__main__':

    # Define a root path, which is the csv file of the sample results.
    root_path = None
    main(root_path)


