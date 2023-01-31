# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      hyper.py
@Author:    Rosenberg
@Date:      2023/1/14 18:56 
@Documentation: 
    ...
"""
import cProfile
import json
import os
import pstats
from collections import OrderedDict
from multiprocessing import cpu_count
from typing import List, Mapping, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jsonargparse import CLI
from pyserini.query_iterator import get_query_iterator, TopicsFormat
from tqdm import tqdm

from source import DEFAULT_CACHE_DIR
from source.search import QUERY_NAME_MAPPING, search

PARAMETER_NAME_MAPPING = {
    "num_pseudo_queries": "Number of pseudo-queries",
    "num_pseudo_return_hits": "Number of returned pseudo hits",
}


def hyper(
        topic_name: str = 'msmarco-passage-dev-subset',
        pseudo_name: str = 'msmarco_v1_passage_doc2query-t5_expansions_5',
        pseudo_index_dir: str = None,
        num_pseudo_queries: int = 8,
        num_pseudo_queries_min: int = None,
        num_pseudo_queries_max: int = None,
        num_pseudo_queries_step: int = 1,
        num_pseudo_return_hits: int = 1000,
        num_pseudo_return_hits_min: int = None,
        num_pseudo_return_hits_max: int = None,
        num_pseudo_return_hits_step: int = 50,
        pseudo_encoder_name: Union[str, List[str]] = "lucene",
        doc_index: Union[str, List[str]] = 'msmarco-v1-passage-full',
        max_passage: bool = False,
        max_passage_hits: int = 1000,
        num_repeat_latency: int = 5,
        threads: int = cpu_count(),
        batch_size: int = cpu_count(),
        device: str = "cpu",
):
    """

    :param topic_name: Name of topics.
    :param pseudo_name: index name of the candidate pseudo queries
    :param pseudo_index_dir: index path to the candidate pseudo queries.
    :param num_pseudo_queries: Default num_pseudo_queries or set num_pseudo_queries.
    :param num_pseudo_queries_min:
        When both specify the num_pseudo_queries_min and num_pseudo_queries_max,
        conduct hyperparameter experiment from num_pseudo_queries_min to num_pseudo_queries_max.
    :param num_pseudo_queries_step: When num_pseudo_queries is in range, step for this range.
    :param num_pseudo_queries_max: See num_pseudo_queries_min
    :param num_pseudo_return_hits: Default num_pseudo_return_hits or set num_pseudo_return_hits.
    :param num_pseudo_return_hits_min: Same to num_pseudo_queries with only num_pseudo_return_hits.
    :param num_pseudo_return_hits_max: Same to num_pseudo_queries with only num_pseudo_return_hits.
    :param num_pseudo_return_hits_step: Same to num_pseudo_queries with only num_pseudo_return_hits.
    :param pseudo_encoder_name: Path to query encoder pytorch checkpoint or hgf encoder model name
    :param doc_index: the index of the candidate documents
    :param max_passage: Select only max passage from document.
    :param max_passage_hits: Final number of hits when selecting only max passage.
    :param num_repeat_latency: num of times for repeat measure latency.
    :param threads: maximum threads to use during search
    :param batch_size: batch size used for the batch search.
    :param device: the device the whole search procedure will on
    """

    parameter_range, parameter_name = None, None
    if num_pseudo_queries_min is not None and num_pseudo_queries_max is not None:
        parameter_range = range(num_pseudo_queries_max, num_pseudo_queries_min, -num_pseudo_queries_step)
        parameter_name = "num_pseudo_queries"
    if num_pseudo_return_hits_min is not None and num_pseudo_return_hits_max is not None:
        if parameter_range is not None:
            ValueError(
                "Should not specify both num_pseudo_queries and num_pseudo_return_hits in range."
                "Conduct experiment on one hyperparameter only."
            )
        parameter_range = range(num_pseudo_return_hits_max, num_pseudo_return_hits_min, -num_pseudo_return_hits_step)
        parameter_name = "num_pseudo_return_hits"
    if parameter_name is None:
        raise ValueError(
            "Specify num_pseudo_queries or num_pseudo_return_hits in range."
            "None of them is specified now."
        )

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
        "device": device,
        "max_passage": max_passage,
        "max_passage_hits": max_passage_hits,
    }

    output_path = os.path.join(DEFAULT_CACHE_DIR, "runs", "hyper")
    os.makedirs(output_path, exist_ok=True)

    query_iterator = get_query_iterator(QUERY_NAME_MAPPING[topic_name], TopicsFormat.DEFAULT)
    query_length = len(query_iterator)

    statistics, latencies = OrderedDict(), OrderedDict()
    for param in tqdm(parameter_range):
        kargs[parameter_name] = param
        latencies[param] = list()

        for i in range(num_repeat_latency):
            with cProfile.Profile() as profile:
                metrics: Mapping[str, float] = search(**kargs)[2]
                stats = pstats.Stats(profile)
                stats.strip_dirs()

                search_func_key = None
                for key in stats.stats.keys():
                    if "pseudo.py" in key and 'batch_search' in key:
                        search_func_key = key

                latencies[param].append(stats.stats[search_func_key][3] / query_length)
                statistics[param] = metrics

    statistics_name = f"statistics.{parameter_name}-{parameter_range.start}-{parameter_range.stop}-{-parameter_range.step}"
    with open(os.path.join(output_path, f"{statistics_name}.json"), "w") as f:
        f.write(json.dumps(statistics, indent=4, sort_keys=True))

    parameter_name = PARAMETER_NAME_MAPPING[parameter_name]

    data, i = pd.DataFrame(columns=[parameter_name, "metric", "Value"]), 0
    for parameter, metrics in statistics.items():
        for metric, value in metrics.items():
            data.loc[i] = [parameter, metric, value]
            i += 1
    ax = sns.lineplot(data, x=parameter_name, y="Value", hue="metric", style="metric", markers=True, dashes=False)

    data, i = pd.DataFrame(columns=[parameter_name, "Latency(s/query)"]), 0
    for parameter, latency in latencies.items():
        for l in latency:
            data.loc[i] = [parameter, l]
            i += 1
    ax2 = ax.twinx()
    sns.lineplot(data, x=parameter_name, y="Latency(s/query)", color='grey', ax=ax2)

    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    ax2.legend(handles + ax2.get_lines(), labels + ['Latency'])

    plt.savefig(os.path.join(output_path, f"{statistics_name}.pdf"))
    plt.show()


if __name__ == '__main__':
    CLI(hyper)
