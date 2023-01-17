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
from typing import List, Mapping

import pandas as pd
from jsonargparse import CLI
from scipy import stats
from tabulate import tabulate

from source import DEFAULT_CACHE_DIR

TREC_EVAL_SCRIPT_PATH = os.path.join(DEFAULT_CACHE_DIR, "eval", "trec_eval-9.0.7", "trec_eval")


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
    command = [TREC_EVAL_SCRIPT_PATH, '-c', '-m', 'all_trec', '-M', '1000', qrels, res]
    output = run(command)
    return OrderedDict({
        metric: float(re.findall(r'{0}\s+all.+\d+'.format(metric), output)[0].split('\t')[2].strip())
        for metric in metrics
    })


def evaluate_trec_per_query(qrels, res, metrics):
    """ all_trecs, """
    command = [TREC_EVAL_SCRIPT_PATH, '-c', '-m', 'all_trec', '-q', '-M', '1000', qrels, res]
    output = run(command)

    metrics_val = []
    for metric in metrics:
        curr_res = re.findall(r'{0}\s+\t\d+.+\d+'.format(metric), output)
        curr_res = list(map(lambda x: float(x.split('\t')[-1]), curr_res))
        metrics_val.append(curr_res)

    return OrderedDict(zip(metrics, metrics_val))


def tt_test(qrels, res1, res2, metrics=None):
    met_dict1 = evaluate_trec_per_query(qrels, res1, metrics)
    met_dict2 = evaluate_trec_per_query(qrels, res2, metrics)

    avg_met_dict1 = evaluate_trec(qrels, res1, metrics)
    avg_met_dict2 = evaluate_trec(qrels, res2, metrics)

    test_dict = OrderedDict({
        metric: {
            "candidate": avg_met_dict1[metric],
            "reference": avg_met_dict2[metric],
            "p_value": stats.ttest_rel(met_dict1.get(metric), met_dict2.get(metric))[1]
        } for metric in metrics
    })
    return test_dict


def evaluate(
        candidate_name: str = None,
        path_to_candidate: str = None,
        reference_name: str = None,
        path_to_reference: str = None,
        topic_name: str = None,
        metrics: List[str] = None,
        print_result: bool = False,
) -> Mapping[str, float]:
    if metrics is None:
        metrics = ['recip_rank', 'ndcg_cut_10', 'map', 'recall_100', 'recall_500', 'recall_1000']

    if candidate_name is not None:
        if path_to_candidate is not None:
            raise ValueError("Can not specify both candidate_name and path_to_candidate")
        else:
            run_path = os.path.join(DEFAULT_CACHE_DIR, "runs")
            path_to_candidate = os.path.join(run_path, candidate_name)
    elif path_to_candidate is None:
        raise ValueError("At least specify candidate_name or path_to_candidate")

    if reference_name is not None:
        if path_to_reference is not None:
            raise ValueError("Can not specify both reference_name and path_to_reference")
        else:
            run_path = os.path.join(DEFAULT_CACHE_DIR, "runs")
            path_to_reference = os.path.join(run_path, reference_name)

    if not os.path.exists(topic_name):
        topic_name = get_qrels_file(topic_name)

    if path_to_reference is None:
        result = evaluate_trec(topic_name, path_to_candidate, metrics)
        if print_result:
            print(tabulate({
                key: [value]
                for key, value in result.items()
            }, headers='keys', tablefmt='fancy_grid'))
    else:
        result = tt_test(topic_name, path_to_candidate, path_to_reference, metrics)
        result = pd.DataFrame(result)
        if print_result:
            pd.set_option('display.max_columns', 100)
            pd.set_option('display.width', 100)
            print(result)

    return result


if __name__ == '__main__':
    CLI(evaluate)
