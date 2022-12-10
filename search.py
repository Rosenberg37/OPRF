# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      search.py
@Author:    Rosenberg
@Date:      2022/12/10 9:17 
@Documentation: 
    ...
"""
from multiprocessing import cpu_count

from jsonargparse import CLI
from pyserini.query_iterator import get_query_iterator, TopicsFormat
from tqdm import tqdm

from pprf.pseudo_query_search import PseudoQuerySearcher


def main(
        pseudo_index: str = './indexes/msmarco_v1_passage_doc2query-t5_expansions',
        doc_index: str = 'msmarco-v1-passage-full',
        topic_name: str = 'msmarco-passage-dev-subset',
        output_file_path: str = './runs/two_stage_search.txt',
        num_pseudo_queries: int = 2,
        num_return_hits: int = 1000,
        doc_searcher: str = 'sparse',
        threads: int = cpu_count(),
        batch_size: int = cpu_count(),
        device: str = "cuda"
):
    query_iterator = get_query_iterator(topic_name, TopicsFormat.DEFAULT)
    topics_length = len(query_iterator.topics)
    searcher = PseudoQuerySearcher(pseudo_index, doc_index, doc_searcher, device=device)

    with open(output_file_path, 'w') as f:
        batch_queries = list()
        batch_queries_ids = list()
        for index, (query_id, text) in tqdm(enumerate(tqdm(query_iterator))):
            batch_queries_ids.append(str(query_id))
            batch_queries.append(text)
            if (index + 1) % batch_size == 0 or index == topics_length - 1:
                batch_hits = searcher.batch_search(batch_queries, batch_queries_ids, num_pseudo_queries=num_pseudo_queries, num_return_hits=num_return_hits, threads=threads)

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
