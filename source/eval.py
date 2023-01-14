# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      eval.py
@Author:    Rosenberg
@Date:      2022/12/10 9:17 
@Documentation: 
    ...
"""

import os
import re
import subprocess
from collections import OrderedDict

from jsonargparse import CLI
from scipy import stats
from tabulate import tabulate

from source import DEFAULT_CACHE_DIR

TREC_EVAL_SCRIPT_PATH = os.path.join(DEFAULT_CACHE_DIR, "eval", "trec_eval-9.0.7", "trec_eval")
SAMPLE_EVAL_SCRIPT_PATH = os.path.join(DEFAULT_CACHE_DIR, "eval", "sample_eval.pl")
GD_EVAL_SCRIPT_PATH = os.path.join(DEFAULT_CACHE_DIR, "eval", "gdeval.pl")


def get_qrels_file(collection_name):
    """
    Parameters
    ----------
    collection_name : str
        collection_name

    Returns
    -------
    path : str
        path of the qrels file
    """
    from pyserini.search._base import JRelevanceJudgments, qrels_mapping
    if collection_name in qrels_mapping:
        qrels = qrels_mapping[collection_name]

        target_path = os.path.join(DEFAULT_CACHE_DIR, qrels.path)
        if os.path.exists(target_path):
            return target_path

        target_dir = os.path.split(target_path)[0]
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        with open(target_path, 'w') as file:
            qrels_content = JRelevanceJudgments.getQrelsResource(qrels)
            file.write(qrels_content)
        return target_path

    raise FileNotFoundError(f'no qrels file for {collection_name}')


def run(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, _ = process.communicate()
    return str(output, encoding='utf-8')


def evaluate_trec(qrels, res, metrics):
    """ all_trecs, """
    command = [TREC_EVAL_SCRIPT_PATH, '-m', 'all_trec', '-M', '1000', qrels, res]
    output = run(command)
    return OrderedDict({
        metric: float(re.findall(r'{0}\s+all.+\d+'.format(metric), output)[0].split('\t')[2].strip())
        for metric in metrics
    })


def evaluate_sample_trec(qrels, res, metrics):
    command = [SAMPLE_EVAL_SCRIPT_PATH, qrels, res]
    output = run(command)

    metrics_val = []
    for metric in metrics:
        metrics_val.append(re.findall(r'{0}\s+all.+\d+'.format(metric), output)[0].split('\t')[4].strip())

    return OrderedDict(zip(metrics, metrics_val))


def evaluate_metrics(qrels, res, sample_qrels=None, metrics=None):
    normal_metrics = [met for met in metrics if not met.startswith('i')]
    infer_metrics = [met for met in metrics if met.startswith('i')]

    metrics_val_dict = OrderedDict()
    if len(normal_metrics) > 0:
        metrics_val_dict.update(evaluate_trec(qrels, res, metrics=normal_metrics))
    if len(infer_metrics) > 0:
        metrics_val_dict.update(evaluate_sample_trec(sample_qrels, res, metrics=infer_metrics))

    return metrics_val_dict


################################## perquery information ####################################
def evaluate_trec_perquery(qrels, res, metrics):
    ''' all_trecs, '''
    command = [TREC_EVAL_SCRIPT_PATH, '-m', 'all_trec', '-q', '-M', '1000', qrels, res]
    output = run(command)

    metrics_val = []
    for metric in metrics:
        curr_res = re.findall(r'{0}\s+\t\d+.+\d+'.format(metric), output)
        curr_res = list(map(lambda x: float(x.split('\t')[-1]), curr_res))
        metrics_val.append(curr_res)

    return OrderedDict(zip(metrics, metrics_val))


def evaluate_sample_trec_perquery(qrels, res, metrics):
    command = [SAMPLE_EVAL_SCRIPT_PATH, '-q', qrels, res]
    output = run(command)

    metrics_val = []
    for metric in metrics:
        curr_res = re.findall(r'{0}\s+\t\d+.+\d+'.format(metric), output)
        curr_res = map(lambda x: float(x.split('\t')[-1]), curr_res)
        metrics_val.append(curr_res)

    return OrderedDict(zip(metrics, metrics_val))


def evaluate_metrics_perquery(qrels, res, sample_qrels=None, metrics=None):
    normal_metrics = [met for met in metrics if not met.startswith('i')]
    infer_metrics = [met for met in metrics if met.startswith('i')]

    metrics_val_dict = OrderedDict()
    if len(normal_metrics) > 0:
        metrics_val_dict.update(evaluate_trec_perquery(qrels, res, metrics=normal_metrics))
    if len(infer_metrics) > 0:
        metrics_val_dict.update(evaluate_sample_trec_perquery(sample_qrels, res, metrics=infer_metrics))

    return metrics_val_dict


def tt_test(qrels, res1, res2, sample_qrels=None, metrics=None):
    met_dict1 = evaluate_metrics_perquery(qrels, res1, sample_qrels, metrics)
    met_dict2 = evaluate_metrics_perquery(qrels, res2, sample_qrels, metrics)

    avg_met_dict1 = evaluate_metrics(qrels, res1, sample_qrels, metrics)
    avg_met_dict2 = evaluate_metrics(qrels, res2, sample_qrels, metrics)
    print(avg_met_dict1)
    print(avg_met_dict2)

    test_dict = OrderedDict()
    for met in met_dict1.keys():
        p_value = stats.ttest_rel(met_dict1.get(met), met_dict2.get(met))[1]
        test_dict.update({met: p_value})

    return test_dict


def evaluate_trec_per_query(qrels, res):
    command = [TREC_EVAL_SCRIPT_PATH, '-q', '-m', 'ndcg_cut', qrels, res]
    output = run(command)
    # print(output)
    ndcg_lines = re.findall(r'ndcg_cut_10\s+\t\d+.+\d+', output)
    ndcg10 = float(re.findall(r'ndcg_cut_10\s+\tall+.+\d+', output)[0].split('\t')[2])
    # print(ndcg10)
    # print(ndcg_lines)
    NDCG10_all = 0.
    NDCG10 = {}
    for line in ndcg_lines:
        tokens = line.split('\t')
        # print(tokens)
        assert tokens[0].strip() == 'ndcg_cut_10'
        qid, ndcg = tokens[1].strip(), float(tokens[2].strip())
        NDCG10[qid] = ndcg
        NDCG10_all += ndcg

    NDCG10_all /= len(NDCG10)
    assert round(NDCG10_all, 4) == ndcg10
    # print('ndcg@10: ', NDCG10_all)
    # print(len(NDCG10))
    # gd_command = [gd_eval_script_path, '-k', '20', qrels, res] #+ " | awk -F',' '{print $3}'"
    # gd_output = run(gd_command, get_ouput=True)
    # gd_output = str(gd_output, encoding='utf-8')
    # print(gd_output)

    # NDCG10_set, ERR10_set = [], []
    # for line in gd_output.split('\n')[1: -2]:
    #   ndcg, err = line.split(',')[2: 4]
    #   NDCG10_set.append(float(ndcg))
    #   ERR10_set.append(float(err))

    # print len(NDCG20_set)
    # print NDCG20_set

    return NDCG10


ARG_DICT = {
    'msmarco-passage-dev-subset': ['-c', '-M', '10', '-m', 'recip_rank', '-m', 'recall.1000', 'msmarco-passage-dev-subset'],
    'dl19-passage': ['-c', '-l', '2', '-m', 'map', '-m', 'ndcg_cut.10', '-l', '2', '-m', 'recall.1000', 'dl19-passage'],
    'dl20': ['-c', '-l', '2', '-m', 'map', '-m', 'ndcg_cut.10', '-l', '2', '-m', 'recall.1000', 'dl20-passage'],
    'dl19-doc': ['-c', '-M', '100', '-m', 'map', '-m', 'ndcg_cut.10', '-m', 'recall.1000', 'dl19-doc'],
}


def evaluate(
        candidate_name: str = None,
        path_to_candidate: str = None,
        topic_name: str = None,
        print_result: bool = True,
):
    if candidate_name is not None:
        if path_to_candidate is not None:
            raise ValueError("Can not specify both candidate_name and path_to_candidate")
        else:
            run_path = os.path.join(DEFAULT_CACHE_DIR, "runs")
            path_to_candidate = os.path.join(run_path, candidate_name)
    elif path_to_candidate is None:
        raise ValueError("At least specify candidate_name or path_to_candidate")

    if not os.path.exists(topic_name):
        topic_name = get_qrels_file(topic_name)

    result = evaluate_trec(topic_name, path_to_candidate, ['map', 'ndcg_cut_10', 'recall_1000'])

    if print_result:
        print(tabulate({
            key: [value]
            for key, value in result.items()
        }, headers='keys', tablefmt='fancy_grid'))

    return result

    # argv = sys.argv
    # res1, res2 = argv[1], argv[2]
    # print(tt_test(qrels, res1, res2))
    # print(evaluate_trec(argv[1], argv[2], ['map', 'P_10']))
    # print(evaluate_sample_trec(argv[3], argv[4], ['infNDCG', 'infAP']))
    #
    # print(evaluate_metrics(argv[1], argv[2], argv[3], ['map', 'P_10', 'infNDCG']))
    # print(evaluate_trec_perquery(argv[1], argv[2], ['Rprec', 'P_10']))
    # print(evaluate_sample_trec_perquery(argv[3], argv[4], ['infNDCG']))
    # print(evaluate_trec_per_query(argv[1], argv[2]))
    # print(evaluate_metrics_perquery(argv[1], argv[2], argv[3], ['Rprec', 'P_10', 'infNDCG']))
    # print(tt_test(argv[1], argv[2], argv[3], argv[4], ['Rprec', 'P_10', 'infNDCG']))
    # print(evaluate_metrics(argv[1], argv[2], None, ['ndcg_cut_1', 'ndcg_cut_3', 'ndcg_cut_5']))
    # print(evaluate_metrics(argv[1], argv[2], None, ['recip_rank', 'ndcg_cut_10', 'map', 'recall_100', 'recall_500', 'recall_1000']))
    # print(evaluate_metrics(argv[1], argv[2], None, ['recip_rank', 'ndcg_cut_10', 'ndcg_cut_20', 'ndcg_cut_30', 'ndcg_cut_100', 'ndcg_cut_200', 'ndcg_cut_500', 'map']))
    # print(tt_test(argv[1], argv[2], argv[3], None, ['ndcg_cut_10', 'recall_1000']))
    # print(tt_test(argv[1], argv[2], argv[3], None, ['ndcg_cut_10', 'map']))


if __name__ == '__main__':
    CLI(evaluate)
