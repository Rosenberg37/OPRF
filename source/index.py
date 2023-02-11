# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      index.py
@Author:    Rosenberg
@Date:      2023/1/14 18:56 
@Documentation: 
    ...
"""
import json
import os
import subprocess
from multiprocessing import cpu_count
from typing import List, Union

from jsonargparse import CLI
from tqdm import tqdm

from source import DEFAULT_CACHE_DIR
from source.hyper import draw
from source.search import search

NUM_PSEUDO_INDEX = [1, 5, 10, 20, 40, 80]
NUM_PSEUDO_INDEX_MAPPING = {
    1: '1',
    5: '5',
    10: '10',
    20: '20',
    40: '40',
    80: '-1',
}
NUMBER_OF_DOCUMENT = 8841823


def index(
        topic_name: str = 'msmarco-passage-dev-subset',
        pseudo_name_predix: str = 'msmarco_v1_passage_doc2query-t5_expansions',
        num_pseudo_index: List[int] = None,
        pseudo_encoder_name: Union[str, List[str]] = "lucene",
        pseudo_doc_index: Union[str, List[str]] = 'msmarco-v1-passage-full',
        max_passage: bool = False,
        max_passage_hits: int = 1000,
        metrics: List[str] = None,
        threads: int = cpu_count(),
        batch_size: int = cpu_count(),
        device: str = "cpu",
):
    """

    :param topic_name: Name of topics.
    :param pseudo_name_predix: index name of the candidate pseudo queries
    :param num_pseudo_index: number of queries returned by each index in pseudo index.
    :param pseudo_encoder_name: Path to query encoder pytorch checkpoint or hgf encoder model name
    :param pseudo_doc_index: the index of the candidate documents
    :param max_passage: Select only max passage from document.
    :param max_passage_hits: Final number of hits when selecting only max passage.
    :param metrics: metrics that play evaluation on.
    :param threads: maximum threads to use during search
    :param batch_size: batch size used for the batch search.
    :param device: the device the whole search procedure will on
    """

    if num_pseudo_index is None:
        num_pseudo_index = NUM_PSEUDO_INDEX

    kargs = {
        "topic_name": topic_name,
        "pseudo_name": pseudo_name_predix,
        "pseudo_encoder_name": pseudo_encoder_name,
        "pseudo_doc_index": pseudo_doc_index,
        "threads": threads,
        "batch_size": batch_size,
        "device": device,
        "max_passage": max_passage,
        "max_passage_hits": max_passage_hits,
        "metrics": metrics,
    }

    output_path = os.path.join(DEFAULT_CACHE_DIR, "runs", "hyper")
    os.makedirs(output_path, exist_ok=True)

    statistics, lines = dict(), dict()
    for param in tqdm(num_pseudo_index):
        pseudo_name = f"{pseudo_name_predix}_{NUM_PSEUDO_INDEX_MAPPING[param]}"
        kargs["pseudo_name"] = pseudo_name

        metrics = search(**kargs)[-1]
        statistics[param] = metrics

        out = subprocess.getoutput("wc -l " + os.path.join(DEFAULT_CACHE_DIR, "runs", "pseudo_queries", pseudo_name, f"{pseudo_name}.json"))
        lines[param] = int(out.split()[0]) / NUMBER_OF_DOCUMENT

    statistics_name = f"statistics.index_num_returned_pseudo_queries"
    with open(os.path.join(output_path, f"{statistics_name}.json"), "w") as f:
        f.write(json.dumps(statistics, indent=4, sort_keys=True))

    draw(
        parameter_name="Number of generated pseudo-queries per query",
        statistics=statistics,
        second_statistics=lines,
        second_name="Pseudo-queries per document",
        output_path=os.path.join(output_path, f"{statistics_name}.pdf")
    )


if __name__ == '__main__':
    CLI(index)
