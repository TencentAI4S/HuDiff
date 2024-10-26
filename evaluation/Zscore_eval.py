import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import time
from tqdm import tqdm
import re
from abnumber import Chain
import json
from urllib.parse import urlencode
import concurrent.futures

import seaborn as sns
import matplotlib.pyplot as plt

import os

SCORE_REGEX = re.compile('<h3>The Z-score value of the Query sequence is: (-?[0-9.]+)</h3>')
def get_z_score_online(seq):
    chain = Chain(seq, scheme='imgt')
    chain_type = 'human_heavy' if chain.chain_type == 'H' else ('human_lambda' if chain.chain_type == 'L' else 'human_kappa')
    html = None
    for retry in range(5):
        url = f'http://www.bioinf.org.uk/abs/shab/shab.cgi?aa_sequence={seq}&DB={chain_type}'
        request = requests.get(url)
        time.sleep(0.5 + retry * 5)
        if request.ok:
            html = request.text
            break
        else:
            print('Retry', retry+1)
    if not html:
        raise ValueError('Z-score server is not accessible')
    matches = SCORE_REGEX.findall(html)
    if not matches:
        print(html)
        # raise ValueError(f'Error calling url {url}')
        return None, None
    return float(matches[0]), chain_type

def get_pair_data_zscore(h_seq, l_seq):
    h_z_score, h_type = get_z_score_online(h_seq)
    l_z_score, l_type = get_z_score_online(l_seq)
    return [h_z_score, h_type, l_z_score, l_type, h_seq, l_seq]

def process_z_score_line(line):
    h_seq = line[1]['hseq']
    l_seq = line[1]['lseq']
    name = [line[1]['name']]
    for retry in range(10):
        try:
            data = get_pair_data_zscore(h_seq, l_seq)
            if len(data) > 2:
                break
        except:
            time.sleep(5)
            continue
    if len(data) != 2:
        new_data = name + data
        new_line_df = pd.DataFrame([new_data],
                                   columns=['Raw_name', 'h_score', 'h_gene', 'l_score', 'l_gene', 'h_seq', 'l_seq'])
        return new_line_df
    else:
        return None


def main(sample_fpath=None):
    """
    Gathering the Z score info for eval.
    sample_fpath: file path of the sample result.
    :return:
    """
    if sample_fpath is None:
        sample_fpath = '/sample_humanization_result.csv'

    print(sample_fpath)
    save_fpath = os.path.join(os.path.dirname(sample_fpath), 'sample_z_score.csv')

    sample_df = pd.read_csv(sample_fpath)
    sample_human_df = sample_df[sample_df['Specific'] == 'humanization'].reset_index(drop=True)

    save_z_df = pd.DataFrame(columns=['Raw_name', 'h_score', 'h_gene', 'l_score', 'l_gene', 'h_seq', 'l_seq'])

    results = []
    for line in sample_human_df.iterrows():
        results.append(process_z_score_line(line))

    save_z_df = pd.concat([result for result in results if result is not None], ignore_index=True)
    Not_successful_index = [i for i, result in enumerate(results) if result is None]

    print(Not_successful_index)
    save_z_df.to_csv(save_fpath, index=False)

if __name__ == '__main__':
    main()
