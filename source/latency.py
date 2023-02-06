# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      latency.py
@Author:    Rosenberg
@Date:      2023/1/23 19:03 
@Documentation: 
    ...
"""
import cProfile
import json
import os
import pstats
from multiprocessing import cpu_count
from typing import Callable, List, Mapping, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jsonargparse import CLI
from pyserini.query_iterator import get_query_iterator, TopicsFormat

from source import DEFAULT_CACHE_DIR
from source.search import search
from source.utils import QUERY_NAME_MAPPING
from source.utils.dense import faiss_main
from source.utils.sparse import lucene_main

FAISS_BASELINES = {
    'DistilBERT-KD-TASB': {
        'index': 'msmarco-passage-distilbert-dot-tas_b-b256-bf',
        'encoder': 'sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco',
        'output': os.path.join(DEFAULT_CACHE_DIR, 'runs', 'run.msmarco-v1-passage.distilbert-kd-tasb-otf.dl19.txt'),
    },
    'TCT-ColBERT': {
        'index': 'msmarco-passage-tct_colbert-bf',
        'encoder': 'castorini/tct_colbert-msmarco',
        'output': os.path.join(DEFAULT_CACHE_DIR, 'runs', 'run.msmarco-v1-passage.tct_colbert-v2-hnp-otf.dl19.txt'),
    },
    'TCT-ColBERTv2': {
        'index': 'msmarco-passage-tct_colbert-v2-hnp-bf',
        'encoder': 'castorini/tct_colbert-v2-hnp-msmarco',
        'output': os.path.join(DEFAULT_CACHE_DIR, 'runs', 'run.msmarco-v1-passage.tct_colbert-v2-hnp-otf.dl19.txt'),
    },
    'ANCE': {
        'index': 'msmarco-passage-ance-bf',
        'encoder': 'castorini/ance-msmarco-passage',
        'output': os.path.join(DEFAULT_CACHE_DIR, 'runs', 'run.msmarco-v1-passage.ance-otf.dl19.txt'),
    },
    'DistilBERT-KD-TASB PRF': {
        'index': 'msmarco-passage-distilbert-dot-tas_b-b256-bf',
        'encoder': 'sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco',
        'output': os.path.join(DEFAULT_CACHE_DIR, 'runs', 'run.msmarco-v1-passage.distilbert-kd-tasb-otf.dl19.txt'),
        'prf_depth': 3,
        'prf_method': 'avg',
    },
    'TCT-ColBERT PRF': {
        'index': 'msmarco-passage-tct_colbert-bf',
        'encoder': 'castorini/tct_colbert-msmarco',
        'output': os.path.join(DEFAULT_CACHE_DIR, 'runs', 'run.msmarco-v1-passage.tct_colbert-v2-hnp-otf.dl19.txt'),
        'prf_depth': 3,
        'prf_method': 'avg',
    },
    'TCT-ColBERTv2 PRF': {
        'index': 'msmarco-passage-tct_colbert-v2-hnp-bf',
        'encoder': 'castorini/tct_colbert-v2-hnp-msmarco',
        'output': os.path.join(DEFAULT_CACHE_DIR, 'runs', 'run.msmarco-v1-passage.tct_colbert-v2-hnp-otf.dl19.txt'),
        'prf_depth': 3,
        'prf_method': 'avg',
    },
    'ANCE PRF': {
        'index': 'msmarco-passage-ance-bf',
        'encoder': 'castorini/ance-msmarco-passage',
        'output': os.path.join(DEFAULT_CACHE_DIR, 'runs', 'run.msmarco-v1-passage.ance-otf.dl19.txt'),
        'prf_depth': 3,
        'prf_method': 'avg',
    }
}
LUCENE_BASELINES = {
    'BM25': {
        'index': 'msmarco-v1-passage-slim',
        'bm25': True,
        'output': 'run.msmarco-v1-passage.bm25-default.dl19.txt'
    },
    'BM25+RM3': {
        'index': 'msmarco-v1-passage-full',
        'bm25': True,
        'rm3': True,
        'output': 'run.msmarco-v1-passage.bm25-rm3-default.dl19.txt'
    },
    'docT5query': {
        'index': 'msmarco-v1-passage-d2q-t5',
        'bm25': True,
        'output': 'run.msmarco-v1-passage.bm25-d2q-t5-default.dl19.txt'
    },
    'uniCOIL': {
        'index': 'msmarco-v1-passage-unicoil',
        'encoder': 'castorini/unicoil-msmarco-passage',
        'impact': True,
        'output': 'run.msmarco-v1-passage.unicoil-otf.dl19.txt'
    },
}
PRF_BASELINES = {
    'BM25+RM3',
    'DistilBERT-KD-TASB PRF',
    'TCT-ColBERT PRF',
    'TCT-ColBERTv2 PRF',
    'ANCE PRF',
}


def measure(
        num_repeat: int,
        metric_name: str,
        search: Callable,
        key_filter: Callable,
):
    metric, latencies = 0, list()
    for i in range(num_repeat):
        with cProfile.Profile() as profile:
            metrics: Mapping[str, float] = search()
            stats = pstats.Stats(profile)
            stats.strip_dirs()

            search_func_key = None
            for key in stats.stats.keys():
                if key_filter(key):
                    search_func_key = key

            latency = stats.stats[search_func_key][3]
            metric = metrics[metric_name]
            latencies.append(latency)

    latencies = sorted(latencies)
    return metric, latencies[num_repeat // 2]


def latency(
        topic_name: str = 'msmarco-passage-dev-subset',
        pseudo_name: str = 'msmarco_v1_passage_doc2query-t5_expansions_5',
        pseudo_index_dir: str = None,
        num_pseudo_queries: int = 8,
        num_pseudo_return_hits: int = 1000,
        pseudo_encoder_name: Union[str, List[str]] = "lucene",
        doc_index: Union[str, List[str]] = 'msmarco-v1-passage-full',
        max_passage: bool = False,
        max_passage_hits: int = 1000,
        num_repeat: int = 5,
        metric_name: str = "MAP",
        threads: int = cpu_count(),
        batch_size: int = cpu_count(),
        device: str = "cpu",
):
    """

    :param topic_name: Name of topics.
    :param pseudo_name: index name of the candidate pseudo queries
    :param pseudo_index_dir: index path to the candidate pseudo queries.
    :param num_pseudo_queries: Default num_pseudo_queries or set num_pseudo_queries.
    :param num_pseudo_return_hits: Default num_pseudo_return_hits or set num_pseudo_return_hits.
    :param pseudo_encoder_name: Path to query encoder pytorch checkpoint or hgf encoder model name
    :param doc_index: the index of the candidate documents
    :param max_passage: Select only max passage from document.
    :param max_passage_hits: Final number of hits when selecting only max passage.
    :param num_repeat: num of times for repeat measure latency.
    :param metric_name: name of metric to be demonstrated.
    :param threads: maximum threads to use during search
    :param batch_size: batch size used for the batch search.
    :param device: the device the whole search procedure will on
    """

    output_path = os.path.join(DEFAULT_CACHE_DIR, "runs", "hyper")
    os.makedirs(output_path, exist_ok=True)

    query_iterator = get_query_iterator(QUERY_NAME_MAPPING[topic_name], TopicsFormat.DEFAULT)
    query_length = len(query_iterator)

    statistics = dict()

    baselines = {**LUCENE_BASELINES, **FAISS_BASELINES}
    for name, kargs in baselines.items():
        kargs['topic_name'] = topic_name
        kargs['batch_size'] = batch_size
        kargs['threads'] = threads
        kargs['print_result'] = False

        if name in FAISS_BASELINES:
            search_func = lambda: faiss_main(device=device, **kargs)[-1]
        elif name in LUCENE_BASELINES:
            search_func = lambda: lucene_main(**kargs)[-1]
        else:
            raise RuntimeError(f"Unexpected {name}")

        metric, latency = measure(
            num_repeat, metric_name,
            search=search_func,
            key_filter=lambda key: ("faiss.py" in key and 'faiss_search' in key) or ("lucene.py" in key and 'lucene_search' in key),
        )

        statistics[name] = [metric, latency / query_length]

    kargs = {
        "topic_name": topic_name,
        "pseudo_name": pseudo_name,
        "pseudo_index_dir": pseudo_index_dir,
        "pseudo_encoder_name": pseudo_encoder_name,
        "num_pseudo_queries": num_pseudo_queries,
        "num_pseudo_return_hits": num_pseudo_return_hits,
        "doc_index": doc_index,
        "threads": threads,
        "batch_size": batch_size,
        "max_passage": max_passage,
        "max_passage_hits": max_passage_hits,
        "print_result": False,
    }

    metric, latency = measure(
        num_repeat, metric_name,
        search=lambda: search(**kargs)[-1],
        key_filter=lambda key: "pseudo.py" in key and 'batch_search' in key,
    )
    statistics["Ours"] = [metric, latency / query_length]

    with open(os.path.join(output_path, "statistics.latency.json"), "w") as f:
        f.write(json.dumps(statistics, indent=4, sort_keys=True))

    data, i = pd.DataFrame(columns=[metric_name, "Latency(s/query)", "Architecture"]), 0
    for name, (map, latency) in statistics.items():
        if name in PRF_BASELINES:
            arc = "Pseudo Relevance Feedback"
        elif name in FAISS_BASELINES:
            arc = "Dense Retrieval"
        elif name in LUCENE_BASELINES:
            arc = "Sparse Retrieval"
        else:
            arc = name

        data.loc[i] = [map, latency, arc]
        i += 1

    sns.scatterplot(data=data, x=metric_name, y="Latency(s/query)", hue="Architecture", style="Architecture")
    for name, (map, latency) in statistics.items():
        plt.text(map - .008, latency + .005, name)

    plt.savefig(os.path.join(output_path, f"latency.pdf"))
    plt.show()


if __name__ == '__main__':
    CLI(latency)
