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
import platform
import re
import subprocess
import tempfile
from collections import OrderedDict
from typing import List, Mapping

import pandas as pd
from jsonargparse import CLI
from pyserini.search import get_qrels_file
from pyserini.util import download_evaluation_script
from scipy import stats
from tabulate import tabulate

from source import DEFAULT_CACHE_DIR

EVAL_NAME_MAPPING = {
    "msmarco-passage-dev-subset": "msmarco-passage-dev-subset",
    "dev-passage": "msmarco-passage-dev-subset",
    "dl19-passage": "dl19-passage",
    "dl20-passage": "dl20-passage",
    "msmarco-doc-dev": "msmarco-doc-dev",
    "dev-doc": "msmarco-doc-dev",
    "dl19-doc": "dl19-doc",
    "dl20-doc": "dl20-doc",
}

TREC_EVAL_SCRIPT_PATH = os.path.join(DEFAULT_CACHE_DIR, "eval", "trec_eval-9.0.7", "trec_eval")


def evaluate_trec(qrels, res, metrics):
    """ all_trecs, """
    command = ['-c', '-l', '2', '-m', 'all_trec', qrels, res]
    if 'doc' in qrels:
        command = ['-c', '-m', 'all_trec', '-q', qrels, res]
    output = trec_eval(command)

    metrics_val = []
    for metric in metrics:
        curr_res = re.findall(r'{0}\s+all.+\d+'.format(metric), output)[0].split('\t')[2].strip()
        metrics_val.append(float(curr_res))

    return OrderedDict(zip(metrics, metrics_val))


def evaluate_trec_per_query(qrels, res, metrics):
    """ all_trecs, """

    command = ['-c', '-l', '2', '-m', 'all_trec', '-q', qrels, res]
    if 'doc' in qrels:
        command = ['-c', '-m', 'all_trec', '-q', qrels, res]
    output = trec_eval(command)

    metrics_val = []
    for metric in metrics:
        curr_res = re.findall(r'{0}\s+\t\d+.+\d+'.format(metric), output)
        curr_res = list(map(lambda x: float(x.split('\t')[-1]), curr_res))
        metrics_val.append(curr_res)

    return OrderedDict(zip(metrics, metrics_val))


def tt_test(qrels, res1, res2, metrics):
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
        metrics: List[str] = None,
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

    if not os.path.exists(topic_name):
        topic_name = get_qrels_file(topic_name)

    if metrics is None:
        metrics = ['recip_rank', 'ndcg_cut_10', 'map', 'recall_100', 'recall_500', 'recall_1000']
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
