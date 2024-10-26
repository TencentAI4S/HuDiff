import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import time
from tqdm import tqdm
import concurrent.futures
import os

from abnumber import Chain


# Define deal out-of-order table.
def regular_order_table(out_of_order_table):
    all_table_data = []
    for table in out_of_order_table:
        table_data = []
        for row in table.find_all('tr'):
            row_data = []
            for cell in row.find_all(['th', 'td']):
                row_data.append(cell.text)
            table_data.append(row_data)
        all_table_data.append(table_data)
    return all_table_data[:2]  # only the first two will be used, all is three.


# Define extract data. Only want to know wther the sequence can be viewed as human.
def extract_human_data(regular_table):
    extracted_data = []
    for table_data in regular_table:
        table_header = table_data[0]
        human_row = [None, None, None, None]
        for row in table_data:
            if row[-1] == 'HUMAN':
                human_row = row
        extracted_data.extend(human_row)
    return extracted_data


# Define request process.
def get_predict_result(job_name, h_seq, l_seq):
    # Url path
    humab_url = 'https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/humab'

    data = {
        'h_sequence_score': h_seq,
        'l_sequence_score': l_seq,
        'jobname_score': job_name,
        'humanise': True,
    }
    reponse = requests.post(humab_url, data=data)
    result_url = reponse.url
    print(result_url)

    # Need to wait a moment.
    time.sleep(15)

    # Get the result page.
    result_response = requests.get(result_url)

    if result_response.status_code == 200:
        soup = BeautifulSoup(result_response.text, 'html.parser')
        tables = soup.find_all('table', {'class': 'table table-results'})
        # print(tables)

        predict_table = regular_order_table(tables)
        print(predict_table)
        extract_data = extract_human_data(predict_table)
        print(extract_data)
    else:
        print('May be url has problem or need larger sleep time.')

    sequence_list = [h_seq, l_seq]
    return extract_data + sequence_list


def process_line(line):
    h_seq = line[1]['hseq']
    l_seq = line[1]['lseq']

    l_chain_type = Chain(l_seq, scheme='imgt').chain_type
    # if l_chain_type == 'L':
    #     return True

    name = [line[1]['name']]
    job_name = line[1]['Specific'] + '_' + str(line[0])
    for retry in range(50):
        try:
            data = get_predict_result(job_name, h_seq, l_seq)
            if len(data) > 2:
                break
        except:
            time.sleep(5)
            continue
    if len(data) != 2:
        new_data = name + data + [l_chain_type]
        new_line_df = pd.DataFrame([new_data],
                                   columns=['Raw_name', 'h_v_gene', 'h_score', 'h_threshold', 'h_classification',
                                 'l_v_gene', 'l_score', 'l_threshold', 'l_classification', 'h_seq', 'l_seq', 'l_chain_type'])
        return new_line_df
    else:
        return None


def main(sample_fpath=None):
    """
    To gather Hu-mab method score for eval.
    :return:
    """
    if sample_fpath is None:
        sample_fpath = 'sample_humanization_result.csv'
    save_fpath = os.path.join(os.path.dirname(sample_fpath), 'sample_humab_score.csv')

    sample_df = pd.read_csv(sample_fpath)
    sample_human_df = sample_df[sample_df['Specific'] == 'humanization'].reset_index(drop=True)
    # print(sample_human_df)

    save_humab_df = pd.DataFrame(columns=['Raw_name', 'h_v_gene', 'h_score', 'h_threshold', 'h_classification',
                                 'l_v_gene', 'l_score', 'l_threshold', 'l_classification', 'h_seq', 'l_seq'])

    results = []
    for line in sample_human_df.iterrows():
        results.append(process_line(line))

    save_humab_df = pd.concat([result for result in results if result is not None], ignore_index=True)
    Not_successful_index = [i for i, result in enumerate(results) if result is None]
    print(Not_successful_index)

    save_humab_df.to_csv(save_fpath, index=False)


if __name__ == '__main__':
    main()
