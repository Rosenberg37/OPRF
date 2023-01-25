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
from collections import OrderedDict
from typing import Mapping

from jsonargparse import CLI
from scipy import stats
from tabulate import tabulate

from source import DEFAULT_CACHE_DIR

TREC_EVAL_SCRIPT_PATH = os.path.join(DEFAULT_CACHE_DIR, "eval", "trec_eval-9.0.7", "trec_eval")
import os
import re
import subprocess
import platform
import pandas as pd
import tempfile

from pyserini.search import get_qrels_file
from pyserini.util import download_evaluation_script

EVAL_ARGS = {
    "msmarco-passage-dev-subset": {
        "recip_rank_10": ['-c', '-M', '10', '-m', 'recip_rank'],
        "recall.1000": ['-c', '-m', 'recall.1000'],
    },
    "dl19-passage": {
        "recip_rank": ['-c', '-l', '2', '-m', 'recip_rank'],
        "ndcg_cut.10": ['-c', '-m', 'ndcg_cut.10'],
        "map": ['-c', '-l', '2', '-m', 'map'],
        "recall.100": ['-c', '-l', '2', '-m', 'recall.100'],
        "recall.500": ['-c', '-l', '2', '-m', 'recall.500'],
        "recall.1000": ['-c', '-l', '2', '-m', 'recall.1000'],
    },
    "dl20-passage": {
        "recip_rank": ['-c', '-l', '2', '-m', 'recip_rank'],
        "ndcg_cut.10": ['-c', '-m', 'ndcg_cut.10'],
        "map": ['-c', '-l', '2', '-m', 'map'],
        "recall.100": ['-c', '-l', '2', '-m', 'recall.100'],
        "recall.500": ['-c', '-l', '2', '-m', 'recall.500'],
        "recall.1000": ['-c', '-l', '2', '-m', 'recall.1000'],
    },
    "msmarco-doc-dev": {
        "recip_rank_100": ['-c', '-M', '100', '-m', 'recip_rank'],
        "recall.1000": ['-c', '-m', 'recall.1000'],
    },
    "dl19-doc": {
        "map_100": ['-c', '-M', '100', '-m', 'map'],
        "ndcg_cut.10": ['-c', '-m', 'ndcg_cut.10'],
        "recall.1000": ['-c', '-m', 'recall.1000'],
        "recall.500": ['-c', '-m', 'recall.500'],
        "recall.100": ['-c', '-m', 'recall.100'],
        "recip_rank": ['-c', '-m', 'recip_rank'],
    },
    "dl20-doc": {
        "map_100": ['-c', '-M', '100', '-m', 'map'],
        "map": ['-c', '-m', 'map'],
        "ndcg_cut.10": ['-c', '-m', 'ndcg_cut.10'],
        "recall.1000": ['-c', '-m', 'recall.1000'],
        "recall.500": ['-c', '-m', 'recall.500'],
        "recall.100": ['-c', '-m', 'recall.100'],
        "recip_rank": ['-c', '-m', 'recip_rank'],
    },
}


def evaluate_trec(qrels, res, metric_args_mapping):
    """ all_trecs, """
    results = OrderedDict()
    for metric, args in metric_args_mapping.items():
        command = args + [qrels, res]
        output = trec_eval(command)
        result = re.findall(r'{0}\s+all.+\d+'.format(args[-1]), output)[0].split('\t')[2].strip()
        results[metric] = float(result)
    return results


def evaluate_trec_per_query(qrels, res, metric_args_mapping):
    """ all_trecs, """
    results = OrderedDict()
    for metric, args in metric_args_mapping.items():
        command = args + ['-q', qrels, res]
        output = trec_eval(command)
        curr_res = re.findall(r'{0}\s+\t\d+.+\d+'.format(metric), output)
        curr_res = list(map(lambda x: float(x.split('\t')[-1]), curr_res))
        results[metric] = curr_res
    return results


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


def trec_eval(args):
    script_path = download_evaluation_script('trec_eval')
    cmd_prefix = ['java', '-jar', script_path]

    # Option to discard non-judged hits in run file
    judged_docs_only = ''
    judged_result = []
    cutoffs = []

    if '-remove-unjudged' in args:
        judged_docs_only = args.pop(args.index('-remove-unjudged'))

    if any([i.startswith('judged.') for i in args]):
        # Find what position the arg is in.
        idx = [i.startswith('judged.') for i in args].index(True)
        cutoffs = args.pop(idx)
        cutoffs = list(map(int, cutoffs[7:].split(',')))
        # Get rid of the '-m' before the 'judged.xxx' option
        args.pop(idx - 1)

    temp_file = ''

    if len(args) > 1:
        if not os.path.exists(args[-2]):
            args[-2] = get_qrels_file(args[-2])
        if os.path.exists(args[-1]):
            # Convert run to trec if it's on msmarco
            with open(args[-1]) as f:
                first_line = f.readline()
            if 'Q0' not in first_line:
                temp_file = tempfile.NamedTemporaryFile(delete=False).name
                print('msmarco run detected. Converting to trec...')
                run = pd.read_csv(args[-1], delim_whitespace=True, header=None, names=['query_id', 'doc_id', 'rank'])
                run['score'] = 1 / run['rank']
                run.insert(1, 'Q0', 'Q0')
                run['name'] = 'TEMPRUN'
                run.to_csv(temp_file, sep='\t', header=None, index=None)
                args[-1] = temp_file

        run = pd.read_csv(args[-1], delim_whitespace=True, header=None)
        qrels = pd.read_csv(args[-2], delim_whitespace=True, header=None)

        # cast doc_id column as string
        run[0] = run[0].astype(str)
        qrels[0] = qrels[0].astype(str)

        # Discard non-judged hits
        if judged_docs_only:
            if not temp_file:
                temp_file = tempfile.NamedTemporaryFile(delete=False).name
            judged_indexes = pd.merge(run[[0, 2]].reset_index(), qrels[[0, 2]], on=[0, 2])['index']
            run = run.loc[judged_indexes]
            run.to_csv(temp_file, sep='\t', header=None, index=None)
            args[-1] = temp_file
        # Measure judged@cutoffs
        for cutoff in cutoffs:
            run_cutoff = run.groupby(0).head(cutoff)
            judged = len(pd.merge(run_cutoff[[0, 2]], qrels[[0, 2]], on=[0, 2])) / len(run_cutoff)
            metric_name = f'judged_{cutoff}'
            judged_result.append(f'{metric_name:22}\tall\t{judged:.4f}')
        cmd = cmd_prefix + args
    else:
        cmd = cmd_prefix

    shell = platform.system() == "Windows"
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               shell=shell)
    stdout, stderr = process.communicate()
    if stderr:
        raise RuntimeError(stderr.decode("utf-8"))

    if temp_file:
        os.remove(temp_file)

    for judged in judged_result:
        print(judged)

    return stdout.decode("utf-8").rstrip()


def evaluate(
        candidate_name: str = None,
        path_to_candidate: str = None,
        reference_name: str = None,
        path_to_reference: str = None,
        topic_name: str = None,
        print_result: bool = False,
) -> Mapping[str, float]:
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

    metric_args_mapping = EVAL_ARGS[topic_name]

    if not os.path.exists(topic_name):
        topic_name = get_qrels_file(topic_name)

    # metrics = ['recip_rank', 'ndcg_cut_10', 'map', 'recall_100', 'recall_500', 'recall_1000']
    if path_to_reference is None:
        result = evaluate_trec(topic_name, path_to_candidate, metric_args_mapping)
        if print_result:
            print(tabulate({
                key: [value]
                for key, value in result.items()
            }, headers='keys', tablefmt='fancy_grid'))
    else:
        result = tt_test(topic_name, path_to_candidate, path_to_reference, metric_args_mapping)
        result = pd.DataFrame(result)
        if print_result:
            pd.set_option('display.max_columns', 100)
            pd.set_option('display.width', 100)
            print(result)

    return result


if __name__ == '__main__':
    CLI(evaluate)
