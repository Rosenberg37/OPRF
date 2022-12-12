# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      search.py
@Author:    Rosenberg
@Date:      2022/12/10 9:17 
@Documentation: 
    ...
"""
import os.path
from multiprocessing import cpu_count

from jsonargparse import CLI
from pyserini.query_iterator import get_query_iterator, TopicsFormat
from tqdm import tqdm

from pprf.pseudo_search import PseudoQuerySearcher

CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'pprf')


def main(
        topic_name: str = 'msmarco-passage-dev-subset',
        doc_index: str = 'msmarco-v1-passage-full',
        encoder: str = "lucene",
        pseudo_name: str = 'msmarco_v1_passage_doc2query-t5_expansions_5',
        pseudo_index: str = None,
        num_pseudo_queries: int = 2,
        threads: int = cpu_count(),
        batch_size: int = cpu_count(),
        device: str = "cpu",
        num_return_hits: int = 1000,
        output_path: str = os.path.join(CACHE_DIR, "runs"),
):
    if pseudo_name is not None:
        if pseudo_index is not None:
            raise ValueError("Can not specify both pseudo_name and pseudo_index")
        else:
            pseudo_index = os.path.join(CACHE_DIR, 'indexes', pseudo_name)
    elif pseudo_index is None:
        raise ValueError("At least specify pseudo_name or pseudo_index")

    query_iterator = get_query_iterator(topic_name, TopicsFormat.DEFAULT)
    topics_length = len(query_iterator.topics)
    searcher = PseudoQuerySearcher(pseudo_index, doc_index, encoder, device=device)

    output_path = os.path.join(CACHE_DIR, output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(os.path.join(output_path, f"{encoder.split('/')[-1]}_{num_pseudo_queries}.txt"), 'w') as f:
        batch_queries = list()
        batch_queries_ids = list()
        for index, (query_id, text) in enumerate(tqdm(query_iterator)):
            batch_queries_ids.append(str(query_id))
            batch_queries.append(text)
            if (index + 1) % batch_size == 0 or index == topics_length - 1:
                batch_hits = searcher.batch_search(
                    batch_queries,
                    batch_queries_ids,
                    num_pseudo_queries=num_pseudo_queries,
                    num_return_hits=num_return_hits,
                    threads=threads
                )

                batch_queries_ids.clear()
                batch_queries.clear()
            else:
                continue

            for topic, hits in batch_hits:
                for i, hit in enumerate(hits):
                    f.write(f'{topic}\t{hit[1]}\t{i + 1}\n')

            batch_hits.clear()


if __name__ == '__main__':
    CLI(main)
