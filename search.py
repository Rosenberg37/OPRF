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
from pyserini.output_writer import get_output_writer, OutputFormat
from pyserini.query_iterator import get_query_iterator, TopicsFormat
from tqdm import tqdm

from pprf.eval import evaluate
from pprf.pseudo_search import PseudoQuerySearcher

CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'pprf')


def main(
        topic_name: str = 'msmarco-passage-dev-subset',
        query_rm3: bool = False,
        query_rocchio: bool = False,
        query_rocchio_use_negative: bool = False,
        pseudo_name: str = 'msmarco_v1_passage_doc2query-t5_expansions_5',
        pseudo_index_dir: str = None,
        num_pseudo_queries: int = 2,
        add_query_to_pseudo: bool = False,
        pseudo_encoder_name: str = "lucene",
        pseudo_prf_depth: int = 0,
        pseudo_prf_method: str = 'avg',
        pseudo_rocchio_alpha: float = 0.9,
        pseudo_rocchio_beta: float = 0.1,
        pseudo_rocchio_gamma: float = 0.1,
        pseudo_rocchio_topk: int = 3,
        pseudo_rocchio_bottomk: int = 0,
        pseudo_sparse_index: str = None,
        pseudo_tokenizer: str = None,
        pseudo_ance_prf_encoder: str = None,
        doc_index: str = 'msmarco-v1-passage-full',
        num_return_hits: int = 1000,
        threads: int = cpu_count(),
        batch_size: int = cpu_count(),
        device: str = "cpu",
        output_path: str = os.path.join(CACHE_DIR, "runs"),
        output_format: str = OutputFormat.TREC.value,
        do_eval: bool = True,
):
    """
    
    :param topic_name: 
    :param query_rm3: 
    :param query_rocchio: 
    :param query_rocchio_use_negative: 
    :param pseudo_name: 
    :param pseudo_index_dir: 
    :param num_pseudo_queries: 
    :param add_query_to_pseudo: 
    :param pseudo_encoder_name: 
    :param pseudo_prf_depth: 
    :param pseudo_prf_method: 
    :param pseudo_rocchio_alpha: 
    :param pseudo_rocchio_beta: 
    :param pseudo_rocchio_gamma: 
    :param pseudo_rocchio_topk: 
    :param pseudo_rocchio_bottomk: 
    :param pseudo_sparse_index: 
    :param pseudo_tokenizer: 
    :param pseudo_ance_prf_encoder: 
    :param doc_index: 
    :param num_return_hits: 
    :param threads: 
    :param batch_size: 
    :param device: 
    :param output_path: 
    :param output_format: 
    :param do_eval: 
    """
    if pseudo_name is not None:
        if pseudo_index_dir is not None:
            raise ValueError("Can not specify both pseudo_name and pseudo_index")
        else:
            pseudo_index_dir = os.path.join(CACHE_DIR, 'indexes', pseudo_name)
    elif pseudo_index_dir is None:
        raise ValueError("At least specify pseudo_name or pseudo_index")

    query_iterator = get_query_iterator(topic_name, TopicsFormat.DEFAULT)
    topics = query_iterator.topics
    topics_length = len(query_iterator.topics)
    searcher = PseudoQuerySearcher(
        pseudo_index_dir, doc_index,
        query_rm3=query_rm3,
        query_rocchio=query_rocchio,
        query_rocchio_use_negative=query_rocchio_use_negative,
        pseudo_encoder_name=pseudo_encoder_name,
        pseudo_prf_depth=pseudo_prf_depth,
        pseudo_prf_method=pseudo_prf_method,
        pseudo_rocchio_alpha=pseudo_rocchio_alpha,
        pseudo_rocchio_beta=pseudo_rocchio_beta,
        pseudo_rocchio_gamma=pseudo_rocchio_gamma,
        pseudo_rocchio_topk=pseudo_rocchio_topk,
        pseudo_rocchio_bottomk=pseudo_rocchio_bottomk,
        pseudo_sparse_index=pseudo_sparse_index,
        pseudo_tokenizer=pseudo_tokenizer,
        pseudo_ance_prf_encoder=pseudo_ance_prf_encoder,
        device=device
    )

    output_path = os.path.join(CACHE_DIR, output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path = os.path.join(output_path, f"run.{pseudo_name}.{num_pseudo_queries}.{pseudo_encoder_name.split('/')[-1]}.{topic_name}.txt")
    tag = output_path[:-4]
    output_writer = get_output_writer(output_path, OutputFormat(output_format), 'w', max_hits=num_return_hits, tag=tag, topics=topics)

    with output_writer:
        batch_queries, batch_queries_ids = list(), list()
        for index, (query_id, text) in enumerate(tqdm(query_iterator)):
            batch_queries_ids.append(str(query_id))
            batch_queries.append(text)
            if (index + 1) % batch_size == 0 or index == topics_length - 1:
                batch_hits = searcher.batch_search(
                    batch_queries,
                    batch_queries_ids,
                    num_pseudo_queries=num_pseudo_queries,
                    add_query_to_pseudo=add_query_to_pseudo,
                    num_return_hits=num_return_hits,
                    threads=threads
                )

                batch_queries_ids.clear()
                batch_queries.clear()
            else:
                continue

            for topic, hits in batch_hits:
                output_writer.write(topic, hits)

            batch_hits.clear()

    if do_eval:
        evaluate(topic_name=topic_name, path_to_candidate=output_path)


if __name__ == '__main__':
    CLI(main)
