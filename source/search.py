# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      search.py
@Author:    Rosenberg
@Date:      2022/12/10 9:17 
@Documentation: 
    ...
"""
import os
import os.path
from multiprocessing import cpu_count
from typing import List, Mapping, Tuple, Union

from jsonargparse import CLI
from pyserini.query_iterator import get_query_iterator, TopicsFormat
from tqdm import tqdm

from source import BatchSearchResult, DEFAULT_CACHE_DIR
from source.eval import EVAL_NAME_MAPPING, evaluate
from source.utils.output import OutputWriter
from source.utils.pseudo import PseudoQuerySearcher

QUERY_NAME_MAPPING = {
    "msmarco-passage-dev-subset": "msmarco-passage-dev-subset",
    "dev-passage": "msmarco-passage-dev-subset",
    "dl19-passage": "dl19-passage",
    "dl20-passage": "dl20",
    "msmarco-doc-dev": "msmarco-doc-dev",
    "dev-doc": "msmarco-doc-dev",
    "dl19-doc": "dl19-doc",
    "dl20-doc": "dl20",
}


def search(
        topic_name: str = 'msmarco-passage-dev-subset',
        query_rm3: bool = False,
        query_rocchio: bool = False,
        query_rocchio_use_negative: bool = False,
        pseudo_name: str = 'msmarco_v1_passage_doc2query-t5_expansions_-1',
        pseudo_index_dir: str = None,
        num_pseudo_queries: int = 8,
        add_query_to_pseudo: bool = False,
        num_pseudo_return_hits: int = 1000,
        pseudo_encoder_name: Union[str, List[str]] = "lucene",
        pseudo_prf_depth: int = 0,
        pseudo_prf_method: str = 'avg',
        pseudo_rocchio_alpha: float = 0.9,
        pseudo_rocchio_beta: float = 0.1,
        pseudo_rocchio_gamma: float = 0.1,
        pseudo_rocchio_topk: int = 3,
        pseudo_rocchio_bottomk: int = 0,
        doc_index: Union[str, List[str]] = 'msmarco-v1-passage-full',
        num_return_hits: int = 1000,
        max_passage: bool = False,
        max_passage_hits: int = 1000,
        max_passage_delimiter: str = '#',
        output_path: str = os.path.join(DEFAULT_CACHE_DIR, "runs"),
        reference_name: str = None,
        print_result: bool = True,
        threads: int = cpu_count(),
        batch_size: int = cpu_count(),
        device: str = "cpu",
) -> Tuple[BatchSearchResult, BatchSearchResult, Mapping[str, float]]:
    """

    :param topic_name: Name of topics.
    :param query_rm3: whether the rm3 algorithm used for the first stage search.
    :param query_rocchio: whether the rocchio algorithm used for the first stage search.
    :param query_rocchio_use_negative: whether the rocchio algorithm with negative used for the first stage search.
    :param pseudo_name: index name of the candidate pseudo queries
    :param pseudo_index_dir: index path to the candidate pseudo queries.
    :param num_pseudo_queries: how many pseudo query used for second stage
    :param add_query_to_pseudo: whether add query into pseudo query for search
    :param num_pseudo_return_hits: Number of hits to return by each pseudo query.
    :param pseudo_encoder_name: Path to query encoder pytorch checkpoint or hgf encoder model name
    :param pseudo_prf_depth: Specify how many passages are used for PRF, 0: Simple retrieval with no PRF, > 0: perform PRF
    :param pseudo_prf_method: Choose PRF methods, avg or rocchio
    :param pseudo_rocchio_alpha: The alpha parameter to control the contribution from the query vector
    :param pseudo_rocchio_beta: The beta parameter to control the contribution from the average vector of the positive PRF passages
    :param pseudo_rocchio_gamma: The gamma parameter to control the contribution from the average vector of the negative PRF passages
    :param pseudo_rocchio_topk: Set topk passages as positive PRF passages for rocchio
    :param pseudo_rocchio_bottomk: Set bottomk passages as negative PRF passages for rocchio, 0: do not use negatives prf passages.
    :param doc_index: the index of the candidate documents
    :param num_return_hits: how many hits will be returned
    :param max_passage: Select only max passage from document.
    :param max_passage_hits: Final number of hits when selecting only max passage.
    :param max_passage_delimiter: Delimiter between docid and passage id.
    :param reference_name: Reference name left for the evaluation of p-value
    :param print_result: whether print the evaluation result.
    :param threads: maximum threads to use during search
    :param batch_size: batch size used for the batch search.
    :param device: the device the whole search procedure will on
    :param output_path: the path where the run file will be outputted
    """

    if pseudo_name is not None:
        if pseudo_index_dir is not None:
            raise ValueError("Can not specify both pseudo_name and pseudo_index")
        else:
            pseudo_index_dir = os.path.join(DEFAULT_CACHE_DIR, 'indexes', pseudo_name)
    elif pseudo_index_dir is None:
        raise ValueError("At least specify pseudo_name or pseudo_index")

    searcher = PseudoQuerySearcher(
        pseudo_index_dir, doc_index,
        query_rm3=query_rm3,
        query_rocchio=query_rocchio,
        query_rocchio_use_negative=query_rocchio_use_negative,
        add_query_to_pseudo=add_query_to_pseudo,
        pseudo_encoder_name=pseudo_encoder_name,
        pseudo_prf_depth=pseudo_prf_depth,
        pseudo_prf_method=pseudo_prf_method,
        pseudo_rocchio_alpha=pseudo_rocchio_alpha,
        pseudo_rocchio_beta=pseudo_rocchio_beta,
        pseudo_rocchio_gamma=pseudo_rocchio_gamma,
        pseudo_rocchio_topk=pseudo_rocchio_topk,
        pseudo_rocchio_bottomk=pseudo_rocchio_bottomk,
        device=device
    )

    if topic_name not in QUERY_NAME_MAPPING or topic_name not in EVAL_NAME_MAPPING:
        raise ValueError(f"{topic_name} is current not supported.")
    query_iterator = get_query_iterator(QUERY_NAME_MAPPING[topic_name], TopicsFormat.DEFAULT)

    if type(pseudo_encoder_name) is str:
        pseudo_encoder_full_name = pseudo_encoder_name.split('/')[-1]
    elif type(pseudo_encoder_name) is list:
        pseudo_encoder_full_name = "hybrid"
    else:
        raise ValueError("Unexpected type of pseudo_encoder_name.")
    if pseudo_prf_depth is not None:
        pseudo_encoder_full_name += f"-{pseudo_prf_method}-{pseudo_prf_depth}"

    run_name = f"run.{pseudo_name}.{topic_name}.{num_pseudo_queries}.{pseudo_encoder_full_name}.txt"
    run_path = os.path.join(output_path, run_name)

    log_name = f"run.{pseudo_name}.{topic_name}.{num_pseudo_queries}.{pseudo_encoder_full_name}.log"
    log_path = os.path.join(output_path, "log", log_name)

    output_writer = OutputWriter(
        run_path,
        log_path=log_path,
        max_hits=num_return_hits,
        tag=output_path[:-4],
        topics=query_iterator.topics,
        use_max_passage=max_passage,
        max_passage_delimiter=max_passage_delimiter,
        max_passage_hits=max_passage_hits
    )

    batch_queries, batch_qids = list(), list()
    query_hits, pseudo_hits, queries_ids = dict(), dict(), dict()
    for index, (query_id, text) in enumerate(tqdm(query_iterator)):
        batch_queries.append(text), batch_qids.append(str(query_id))

        if (index + 1) % batch_size == 0 or index == len(query_iterator.topics) - 1:
            batch_query_hits, batch_pseudo_hits = searcher.batch_search(
                batch_queries, batch_qids,
                num_pseudo_queries=num_pseudo_queries,
                num_pseudo_return_hits=num_pseudo_return_hits,
                threads=threads,
                return_pseudo_hits=True,
            )

            query_hits.update(batch_query_hits), pseudo_hits.update(batch_pseudo_hits)
            queries_ids.update({qid: query for query, qid in zip(batch_queries, batch_qids)})
            batch_queries.clear(), batch_qids.clear()

    with output_writer:
        output_writer.write(query_hits, pseudo_hits, queries_ids)

    metrics = evaluate(
        topic_name=EVAL_NAME_MAPPING[topic_name],
        path_to_candidate=run_path,
        reference_name=reference_name,
        print_result=print_result
    )
    return query_hits, pseudo_hits, metrics


if __name__ == '__main__':
    CLI(search)
