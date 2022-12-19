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
import sys

from jsonargparse import CLI

from source import DEFAULT_CACHE_DIR

ARG_DICT = {
    'msmarco-passage-dev-subset': ['-c', '-M', '10', '-m', 'recip_rank', '-m', 'recall.1000', 'msmarco-passage-dev-subset'],
    'dl19-passage': ['-c', '-l', '2', '-m', 'map', '-m', 'ndcg_cut.10', '-l', '2', '-m', 'recall.1000', 'dl19-passage'],
    'dl20': ['-c', '-l', '2', '-m', 'map', '-m', 'ndcg_cut.10', '-l', '2', '-m', 'recall.1000', 'dl20-passage'],
}


def evaluate(
        candidate_name: str = None,
        path_to_candidate: str = None,
        topic_name: str = None,
):
    run_path = os.path.join(DEFAULT_CACHE_DIR, "runs")

    if candidate_name is not None:
        if path_to_candidate is not None:
            raise ValueError("Can not specify both candidate_name and path_to_candidate")
        else:
            path_to_candidate = os.path.join(run_path, candidate_name)
    elif path_to_candidate is None:
        raise ValueError("At least specify candidate_name or path_to_candidate")

    sys.argv = [sys.argv[0]] + ARG_DICT[topic_name] + [path_to_candidate]
    # noinspection PyUnresolvedReferences
    from pyserini.eval import trec_eval
    # Import operation will execute the evaluation inside


if __name__ == '__main__':
    CLI(evaluate)
