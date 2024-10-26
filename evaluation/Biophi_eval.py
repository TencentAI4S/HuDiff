import os
import sys

import pandas as pd
from abnumber import Chain
from tqdm import tqdm






def save_pairs(heavy_chains, light_chains, path):
    assert len(heavy_chains) == len(light_chains)
    with open(path, 'w') as f:
        for heavy, light in zip(heavy_chains, light_chains):
            Chain.to_fasta(heavy, f, description='VH')
            Chain.to_fasta(light, f, description='VL')

def trans_to_chain(df, save_path, version=None):
    H_chain_list = [Chain(df.iloc[i]['hseq'], scheme='imgt') for i in df.index]
    L_chain_list = [Chain(df.iloc[i]['lseq'], scheme='imgt') for i in df.index]

    assert version is not None, print('Need to given specific version.')
    for i, (h_chain, l_chain) in tqdm(enumerate(zip(H_chain_list, L_chain_list)), total=len(H_chain_list)):
        name = version + 'human' + f'{i}'
        h_chain.name = name
        l_chain.name = name

    save_pairs(H_chain_list, L_chain_list, save_path)


def main(sample_fpath=None):
    """
    Get the Biophi score.
    :return:
    """
    if sample_fpath is None:
        sample_fpath = '/apdcephfs/share_1364275/waitma/anti_proj/log_v6/' \
                   'v006_pair_not_pretrain_test_split_2024_01_12__20_37_03/2023_shuffle_lab_not_equal_2024_01_15__10_40_24/sample_humanization_result.csv'
    # sample_fpath = '/data/home/waitma/antibody_proj/antidiff/data/lab_data/graft_human/graft_germline_seq.csv'
    save_fpath = os.path.join(os.path.dirname(sample_fpath), 'sample_identity.fa')
    if os.path.exists(save_fpath):
        print('Fasta file Already exists. Skip!')
        return save_fpath

    sample_df = pd.read_csv(sample_fpath)
    sample_human_df = sample_df[sample_df['Specific'] == 'humanization'].reset_index(drop=True)
    # sample_human_df = sample_df[sample_df['type'] == 'mouse'].reset_index(drop=True)
    trans_to_chain(sample_human_df, save_fpath, version='exp')

if __name__ == '__main__':
    main()