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
        aggregation: str = "softmax_sum_with_count",
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
    
    :param topic_name: Name of topics.
    :param query_rm3: whether the rm3 algorithm used for the first stage search.
    :param query_rocchio: whether the rocchio algorithm used for the first stage search.
    :param query_rocchio_use_negative: whether the rocchio algorithm with negative used for the first stage search.
    :param pseudo_name: index name of the candidate pseudo queries
    :param pseudo_index_dir: index path to the candidate pseudo queries.
    :param num_pseudo_queries: how many pseudo query used for second stage
    :param add_query_to_pseudo: whether add query into pseudo query for search
    :param pseudo_encoder_name: Path to query encoder pytorch checkpoint or hgf encoder model name
    :param pseudo_prf_depth: Specify how many passages are used for PRF, 0: Simple retrieval with no PRF, > 0: perform PRF
    :param pseudo_prf_method: Choose PRF methods, avg or rocchio
    :param pseudo_rocchio_alpha: The alpha parameter to control the contribution from the query vector
    :param pseudo_rocchio_beta: The beta parameter to control the contribution from the average vector of the positive PRF passages
    :param pseudo_rocchio_gamma: The gamma parameter to control the contribution from the average vector of the negative PRF passages
    :param pseudo_rocchio_topk: Set topk passages as positive PRF passages for rocchio
    :param pseudo_rocchio_bottomk: Set bottomk passages as negative PRF passages for rocchio, 0: do not use negatives prf passages.
    :param pseudo_sparse_index: The path to sparse index containing the passage contents
    :param pseudo_tokenizer: Path to a hgf tokenizer name or path
    :param pseudo_ance_prf_encoder: The path or name to ANCE-PRF model checkpoint
    :param aggregation: the way of aggregate hits from different pseudo queries
    :param doc_index: the index of the candidate documents
    :param num_return_hits: how many hits will be returned
    :param threads: maximum threads to use during search
    :param batch_size: batch size used for the batch search.
    :param device: the device the whole search procedure will on
    :param output_path: the path where the run file will be outputted
    :param output_format: the format where the run file will be
    :param do_eval: do evaluation step after search or not
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
        aggregation=aggregation,
        device=device
    )

    output_path = os.path.join(CACHE_DIR, output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pseudo_encoder_full_name = pseudo_encoder_name.split('/')[-1]
    if pseudo_prf_depth is not None:
        pseudo_encoder_full_name += "-" + pseudo_prf_method
    run_name = f"run.{pseudo_name}.{topic_name}.{num_pseudo_queries}.{pseudo_encoder_full_name}.{aggregation}.txt"
    output_path = os.path.join(output_path, run_name)
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
