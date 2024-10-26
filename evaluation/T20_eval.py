import os

import requests
import sys
import pandas as pd
import time
from tqdm import tqdm
import re
from abnumber import Chain
import concurrent.futures



T20_REGEX = re.compile('<td>T20 Score:</td><td>([0-9.]+)</td>')
def get_t20_online(seq, region=1):
    if region == 1:
        chain = Chain(seq, scheme='imgt')
        chain_type = 'vh' if chain.chain_type == 'H' else ('vl' if chain.chain_type == 'L' else 'vk')
    elif region == 2:
        chain_type = 'vh'
    else:
        raise ValueError('Region type do not appropriate.')

    html = None
    for retry in range(5):
        url = f'https://sam.curiaglobal.com/t20/cgi-bin/blast.py?chain={chain_type}&region={region}&output=3&seqs={seq}'
        try:
            request = requests.get(url)
            if request.ok:
                html = request.text
                break
        except Exception as e:
            print(e)
        except:
            continue
        time.sleep(0.5 + retry * 5)
        print('Retry', retry+1)
    if not html:
        sys.exit(1)
    # print(html)
    matches = T20_REGEX.findall(html)
    time.sleep(1)
    if not matches:
        print(html)
        # raise ValueError(f'Error calling url {url}')
        return None, None
    return float(matches[0]), chain_type

def get_pair_data_t20(h_seq, l_seq, region=1):
    h_score, h_type = get_t20_online(h_seq, region)
    l_score, l_type = get_t20_online(l_seq, region)
    # print(h_score, l_score)
    return [h_score, h_type, l_score, l_type, h_seq, l_seq]


def get_one_chain_framework_t20(h_seq, region=2):
    h_score, h_type = get_t20_online(h_seq, region)
    return [h_score, h_type, h_seq]


def process_line(line):
    h_seq = line[1]['hseq']
    l_seq = line[1]['lseq']
    name = [line[1]['name']]
    data = []
    for retry in range(3):
        try:
            data = get_pair_data_t20(h_seq, l_seq)
            if len(data) > 2:
                break
        except:
            time.sleep(5)
            # continue
    # if data is not None:
    print(data)
    if len(data) > 2:
        new_data = name + data
        new_line_df = pd.DataFrame([new_data], columns=['Raw_name', 'h_score', 'h_gene', 'l_score', 'l_gene', 'h_seq', 'l_seq'])
        return new_line_df
    else:
        return None

def process_one_seq_and_frame_line(line):
    h_seq = line[1]['hseq']
    name = [line[1]['name']]
    # name = ['vhhseq' + str(line[0])]
    data = []
    for retry in range(3):
        try:
            data = get_one_chain_framework_t20(h_seq, region=2)
            if len(data) > 2:
                break
        except:
            time.sleep(5)
            continue
    # if data is not None:
    print(data)
    if len(data) > 2:
        new_data = name + data
        new_line_df = pd.DataFrame([new_data], columns=['Raw_name', 'h_score', 'h_gene', 'h_seq'])
        return new_line_df
    else:
        return None


def frame_main(sample_fpath=None):
    if sample_fpath is None:
        sample_fpath = '/sample_humanization_result.csv'

    
    print(sample_fpath)
    save_fpath = os.path.join(os.path.dirname(sample_fpath), 'sample_frame_t20_score.csv')
    if os.path.exists(save_fpath):
        return save_fpath

    sample_df = pd.read_csv(sample_fpath)

    sample_human_df = sample_df[sample_df['Specific'] == 'humanization'].reset_index(drop=True)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_one_seq_and_frame_line, sample_human_df.iterrows()), total=len(sample_human_df)))

    save_frame_t20_df = pd.concat([result for result in results if result is not None], ignore_index=True)
    Not_successful_index = [i for i, result in enumerate(results) if result is None]

    print(Not_successful_index)
    save_frame_t20_df.to_csv(save_fpath, index=False)
    return save_fpath


def main(sample_fpath=None):
    """
    Gather the T20 score from the website.
    :return:
    """
    if sample_fpath is None:
        sample_fpath = '/humanization_pair_data_filter.csv'
                   
    save_fpath = os.path.join(os.path.dirname(sample_fpath), 'sample_t20_score.csv')
    if os.path.exists(save_fpath):
        return save_fpath

    sample_df = pd.read_csv(sample_fpath)

    sample_human_df = sample_df[sample_df['Specific'] == 'humanization'].reset_index(drop=True)

    save_t20_df = pd.DataFrame(columns=['Raw_name', 'h_score', 'h_gene', 'l_score', 'l_gene', 'h_seq', 'l_seq'])
    results = []
    for line in sample_human_df.iterrows():
        results.append(process_line(line=line))

    save_t20_df = pd.concat([result for result in results if result is not None], ignore_index=True)
    Not_successful_index = [i for i, result in enumerate(results) if result is None]



    print(Not_successful_index)
    save_t20_df.to_csv(save_fpath, index=False)
    return save_fpath


if __name__ == '__main__':
    main()