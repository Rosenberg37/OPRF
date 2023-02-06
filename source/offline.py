# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      offline.py
@Author:    Rosenberg
@Date:      2023/1/16 10:27
@Documentation:
    ...
"""
import json
import os
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jsonargparse import CLI
from pyserini.query_iterator import get_query_iterator, TopicsFormat
from tqdm import tqdm

from source import BatchSearchResult, DEFAULT_CACHE_DIR
from source.utils.sparse import LuceneBatchSearcher

TOPIC_NAME_MAPPING = {
    "dev": 'msmarco-passage-dev-subset',
    'msmarco-passage-dev-subset': 'msmarco-passage-dev-subset',
    "dl19-passage": "dl19-passage",
    "dl19-doc": "dl19-doc",
    "dl20-passage": "dl20",
    "dl20-doc": "dl20",
    "dl20": "dl20",
}


def main(
        topic_name: str = 'msmarco-passage-dev-subset',
        pseudo_name: str = 'msmarco_v1_passage_doc2query-t5_expansions_5',
        pseudo_index_dir: str = None,
        num_pseudo_queries: int = 8,
        query_rm3: bool = False,
        query_rocchio: bool = False,
        query_rocchio_use_negative: bool = False,
        threads: int = cpu_count(),
        batch_size: int = cpu_count(),
):
    """

    :param topic_name: Name of topics.
    :param pseudo_name: index name of the candidate pseudo queries
    :param pseudo_index_dir: index path to the candidate pseudo queries.
    :param num_pseudo_queries: Default num_pseudo_queries or set num_pseudo_queries.
    :param query_rm3: whether the rm3 algorithm used for the first stage search.
    :param query_rocchio: whether the rocchio algorithm used for the first stage search.
    :param query_rocchio_use_negative: whether the rocchio algorithm with negative used for the first stage search.
    :param threads: maximum threads to use during search
    :param batch_size: batch size used for the batch search.
    """
    if pseudo_name is not None:
        if pseudo_index_dir is not None:
            raise ValueError("Can not specify both pseudo_name and pseudo_index")
        else:
            pseudo_index_dir = os.path.join(DEFAULT_CACHE_DIR, 'indexes', pseudo_name)
    elif pseudo_index_dir is None:
        raise ValueError("At least specify pseudo_name or pseudo_index")

    # Initialize
    topic_name = TOPIC_NAME_MAPPING[topic_name]
    query_iterator = get_query_iterator(topic_name, TopicsFormat.DEFAULT)

    searcher_query = LuceneBatchSearcher(
        pseudo_index_dir,
        rm3=query_rm3,
        rocchio=query_rocchio,
        rocchio_use_negative=query_rocchio_use_negative,
    )

    # Collect data
    statistics, query_num = dict(), 0
    batch_queries_ids = dict()
    all_query_hits = set()
    for index, (query_id, text) in enumerate(tqdm(query_iterator)):
        batch_queries_ids[str(query_id)] = text

        if (index + 1) % batch_size == 0 or index == len(query_iterator.topics) - 1:
            batch_qids, batch_queries = zip(*batch_queries_ids.items())
            batch_query_hits: BatchSearchResult = searcher_query.batch_search(
                batch_queries, batch_qids,
                k=num_pseudo_queries,
                threads=threads,
                add_query_to_pseudo=False,
            )

            for hits in batch_query_hits.values():
                all_query_hits.update([hit.docid for hit in hits])

            query_num += len(batch_queries_ids)
            statistics[query_num] = len(all_query_hits)

            batch_queries_ids.clear()

    # Output result
    output_path = os.path.join(DEFAULT_CACHE_DIR, "runs", "offline")
    os.makedirs(output_path, exist_ok=True)

    statistics_name = f"statistics.{topic_name}"
    with open(os.path.join(output_path, f"{statistics_name}.json"), "w") as f:
        f.write(json.dumps(statistics, indent=4, sort_keys=True))

    # Draw figure
    data, i = pd.DataFrame(columns=["query_num", "pseudo_num"]), 0
    for query_num, pseudo_num in statistics.items():
        data.loc[i] = [query_num, pseudo_num]
        i += 1

    sns.relplot(data, x="query_num", y="pseudo_num", markers=True, kind="line")
    plt.show()
    plt.savefig(os.path.join(output_path, f"{statistics_name}.png"))


if __name__ == '__main__':
    CLI(main)
