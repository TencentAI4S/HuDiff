import os
import pickle

import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile

from ablstm import ModelLSTM

from anarci import anarci, number
import re
from abnumber import Chain

# Deal Heavy seq.
def seq_trans_to_aho(humanization_df):
    """
    Gather ABLSTM score.
    :param humanization_df: DataFrame
    :return:
    """
    h_seq_df = humanization_df['hseq']
    data = []
    for idx, a_seq in enumerate(h_seq_df):
        data.append((f'{idx}', a_seq))

    # single_data = [('1', single_h_seq), ('2', single_h_seq)]
    h_results = anarci(data, scheme='aho', output=False)
    h_aho_seq_list = []
    h_seq_results = h_results[0]
    for seq_list in h_seq_results:
        re_seq = seq_list[0][0]
        str_re_seq = str(re_seq)
        matches =  re.findall(r"'([A-Z\-])'", str_re_seq)
        aho_seq = '-' + ''.join(matches)
        if len(aho_seq) != 150:
            pad_count = 150 - len(aho_seq)
            aho_seq = aho_seq + '-' * pad_count
        h_aho_seq_list.append(aho_seq)

    return h_aho_seq_list

def model_eval(aho_txt_fpath):
    """
    Predicting the H-score by model.
    :param ach_txt:
    :return:
    """
    model_data_path = '/model.npy'
    pred_model = ModelLSTM(embedding_dim=64, hidden_dim=64, device='cuda', gapped=True, fixed_len=True)
    pred_model.load(fn=model_data_path)
    h_score = pred_model.eval(aho_txt_fpath)
    return h_score


def main(sample_fpath=None):
    """
    Get the Score.
    :return:
    """
    if sample_fpath is None:
        sample_fpath =  '/sample_humanization_result.csv'
    
    save_fpath = os.path.join(os.path.dirname(sample_fpath), 'sample_ablstm_score.pkl')

    sample_df = pd.read_csv(sample_fpath)
    sample_human_df = sample_df[sample_df['Specific'] == 'humanization'].reset_index(drop=True)
    aho_list = seq_trans_to_aho(sample_human_df)

    # Save h_aho_seq_list
    tmp = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
    tmp_fpath = tmp.name
    with open(tmp_fpath, 'w') as f:
        for seq in aho_list:
            f.write(seq + '\n')

    h_score = model_eval(tmp_fpath)
    os.remove(tmp_fpath)

    print(h_score)
    with open(save_fpath, 'wb') as f:
        pickle.dump(h_score, f)
        f.close()


if __name__ == '__main__':
    main()