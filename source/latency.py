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
from source.search import QUERY_NAME_MAPPING, search
from source.utils.faiss import FAISS_BASELINES, faiss_main
from source.utils.lucene import LUCENE_BASELINES, lucene_main


def measure(
        num_repeat: int,
        metric_name: str,
        search: Callable,
        key_filter: Callable,
):
    metric, total_latency = 0, 0
    for i in range(num_repeat):
        with cProfile.Profile() as profile:
            metrics: Mapping[str, float] = search()
            stats = pstats.Stats(profile)
            stats.strip_dirs()

            search_func_key = None
            for key in stats.stats.keys():
                if key_filter(key):
                    search_func_key = key

            total_latency = stats.stats[search_func_key][3]
            metric = metrics[metric_name]
            total_latency += total_latency
    return metric, total_latency / num_repeat


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

    statistics = {
        "Ours(GPU)": [0., 0.],
        "Ours(CPU)": [0., 0.]
    }

    baselines = FAISS_BASELINES.copy()
    baselines.update(LUCENE_BASELINES)

    for name, kargs in baselines.items():
        statistics[name] = [0., 0.]
        kargs['topic_name'] = topic_name
        if name in FAISS_BASELINES:
            search_func = lambda: faiss_main(print_result=False, **kargs)[-1]
        elif name in LUCENE_BASELINES:
            search_func = lambda: lucene_main(print_result=False, **kargs)[-1]
        else:
            raise RuntimeError(f"Unexpected {name}")

        metric, total_latency = measure(
            num_repeat, metric_name,
            search=search_func,
            key_filter=lambda key: ("faiss.py" in key and 'faiss_search' in key) or ("lucene.py" in key and 'lucene_search' in key),
        )

        statistics[name][0] = metric
        statistics[name][1] = total_latency / query_length

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

    metric, total_latency = measure(
        num_repeat, metric_name,
        search=lambda: search(device=device, **kargs)[-1],
        key_filter=lambda key: "pseudo.py" in key and 'batch_search' in key,
    )
    statistics["Ours(GPU)"][0] = metric
    statistics["Ours(GPU)"][1] = total_latency / query_length

    metric, total_latency = measure(
        num_repeat, metric_name,
        search=lambda: search(device='cpu', **kargs)[-1],
        key_filter=lambda key: "pseudo.py" in key and 'batch_search' in key,
    )
    statistics["Ours(CPU)"][0] = metric
    statistics["Ours(CPU)"][1] = total_latency / query_length

    with open(os.path.join(output_path, "statistics.latency.json"), "w") as f:
        f.write(json.dumps(statistics, indent=4, sort_keys=True))

    data, i = pd.DataFrame(columns=[metric_name, "Latency(s/query)", "Architecture"]), 0
    for name, (map, total_latency) in statistics.items():
        if name in FAISS_BASELINES:
            arc = "Dense Retrieval"
        elif name in LUCENE_BASELINES:
            arc = "Sparse Retrieval"
        else:
            arc = name

        data.loc[i] = [map, total_latency, arc]
        i += 1

    sns.scatterplot(data=data, x=metric_name, y="Latency(s/query)", hue="Architecture", style="Architecture")
    for name, (map, total_latency) in statistics.items():
        plt.text(map - .008, total_latency + .005, name)

    plt.savefig(os.path.join(output_path, f"latency.pdf"))
    plt.show()


if __name__ == '__main__':
    CLI(latency)
