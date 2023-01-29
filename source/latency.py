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
from typing import List, Mapping, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jsonargparse import CLI
from pyserini.query_iterator import get_query_iterator, TopicsFormat

from source import DEFAULT_CACHE_DIR
from source.search import QUERY_NAME_MAPPING, search
from source.utils.faiss import FAISS_BASELINES, faiss_main
from source.utils.lucene import LUCENE_BASELINES, lucene_main


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
    :param num_pseudo_return_hits: Default num_pseudo_return_hits or set num_pseudo_return_hits.
    :param pseudo_encoder_name: Path to query encoder pytorch checkpoint or hgf encoder model name
    :param doc_index: the index of the candidate documents
    :param max_passage: Select only max passage from document.
    :param max_passage_hits: Final number of hits when selecting only max passage.
    :param num_repeat_latency: num of times for repeat measure latency.
    :param threads: maximum threads to use during search
    :param batch_size: batch size used for the batch search.
    :param device: the device the whole search procedure will on
    """

    output_path = os.path.join(DEFAULT_CACHE_DIR, "runs", "hyper")
    os.makedirs(output_path, exist_ok=True)

    query_iterator = get_query_iterator(QUERY_NAME_MAPPING[topic_name], TopicsFormat.DEFAULT)
    query_length = len(query_iterator)

    statistics = {"Ours": [0, 0]}
    for i in range(num_repeat_latency):
        with cProfile.Profile() as profile:
            metrics: Mapping[str, float] = search(
                topic_name=topic_name,
                pseudo_name=pseudo_name,
                pseudo_index_dir=pseudo_index_dir,
                pseudo_encoder_name=pseudo_encoder_name,
                num_pseudo_queries=num_pseudo_queries,
                num_pseudo_return_hits=num_pseudo_return_hits,
                doc_index=doc_index,
                threads=threads,
                batch_size=batch_size,
                device=device,
                max_passage=max_passage,
                max_passage_hits=max_passage_hits
            )[2]
            stats = pstats.Stats(profile)
            stats.strip_dirs()

            search_func_key = None
            for key in stats.stats.keys():
                if "pseudo.py" in key and 'batch_search' in key:
                    search_func_key = key

            latency = stats.stats[search_func_key][3] / query_length
            statistics["Ours"][0] = metrics['map']
            statistics["Ours"][1] += latency
    statistics["Ours"][1] /= num_repeat_latency

    baselines = FAISS_BASELINES.copy()
    baselines.update(LUCENE_BASELINES)

    for name, kargs in baselines.items():
        statistics[name] = [0, 0]
        for i in range(num_repeat_latency):
            with cProfile.Profile() as profile:
                kargs['topic_name'] = topic_name
                if name in FAISS_BASELINES:
                    metrics: Mapping[str, float] = faiss_main(**kargs)[-1]
                elif name in LUCENE_BASELINES:
                    metrics: Mapping[str, float] = lucene_main(**kargs)[-1]

                stats = pstats.Stats(profile)
                stats.strip_dirs()

                search_func_key = None
                for key in stats.stats.keys():
                    if ("faiss.py" in key and 'faiss_search' in key) or ("lucene.py" in key and 'lucene_search' in key):
                        search_func_key = key

                latency = stats.stats[search_func_key][3] / query_length
                statistics[name][0] = metrics['map']
                statistics[name][1] += latency
        statistics[name][1] /= num_repeat_latency

    with open(os.path.join(output_path, "statistics.latency.json"), "w") as f:
        f.write(json.dumps(statistics, indent=4, sort_keys=True))

    data, i = pd.DataFrame(columns=["map", "latency"]), 0
    for _, (map, latency) in statistics.items():
        data.loc[i] = [map, latency]
        i += 1

    sns.scatterplot(data=data, x="map", y="latency")
    for name, (map, latency) in statistics.items():
        plt.text(map - .008, latency + .005, name)

    plt.savefig(os.path.join(output_path, f"latency.png"))
    plt.show()


if __name__ == '__main__':
    CLI(latency)
