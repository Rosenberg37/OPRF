# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      search.py
@Author:    Rosenberg
@Date:      2022/12/10 9:17 
@Documentation: 
    ...
"""
import json
import os
import os.path
from multiprocessing import cpu_count
from typing import Dict, List, Tuple, Union

from jsonargparse import CLI
from pyserini.index import IndexReader
from pyserini.output_writer import TrecWriter
from pyserini.query_iterator import get_query_iterator, TopicsFormat
from pyserini.search.lucene.__main__ import set_bm25_parameters
from tqdm import tqdm

from source import DEFAULT_CACHE_DIR
from source.eval import evaluate
from source.utils.aggregate import AGGREGATE_DICT
from source.utils.faiss import FaissBatchSearcher
from source.utils.lucene import DENSE_TO_SPARSE, LuceneBatchSearcher, SearchResult


class PseudoQuerySearcher:
    def __init__(
            self,
            pseudo_index_dir: str,
            doc_index: Union[str, List[str]],
            query_rm3: bool = False,
            query_rocchio: bool = False,
            query_rocchio_use_negative: bool = False,
            add_query_to_pseudo: bool = False,
            query_normalize_method: str = None,
            query_normalize_scale: float = 1,
            query_normalize_shift: float = 0,
            pseudo_encoder_name: Union[List[str], str] = "lucene",
            pseudo_prf_depth: int = 0,
            pseudo_prf_method: str = 'avg',
            pseudo_rocchio_alpha: float = 0.9,
            pseudo_rocchio_beta: float = 0.1,
            pseudo_rocchio_gamma: float = 0.1,
            pseudo_rocchio_topk: int = 3,
            pseudo_rocchio_bottomk: int = 0,
            pseudo_normalize_method: str = None,
            pseudo_normalize_scale: float = 1,
            pseudo_normalize_shift: float = 0,
            sparse_alpha: float = 0,
            aggregation: str = "softmax_sum_with_count",
            cache_dir: str = None,
            device: str = "cpu",
    ):
        self.searcher_pseudo = LuceneBatchSearcher(
            pseudo_index_dir,
            normalize_method=query_normalize_method,
            normalize_scale=query_normalize_scale,
            normalize_shift=query_normalize_shift,
            add_query_to_pseudo=add_query_to_pseudo
        )
        set_bm25_parameters(self.searcher_pseudo, None, 2.56, 0.59)
        if query_rm3:
            self.searcher_pseudo.set_rm3()
        if query_rocchio:
            if query_rocchio_use_negative:
                self.searcher_pseudo.set_rocchio(gamma=0.15, use_negative=True)
            else:
                self.searcher_pseudo.set_rocchio()

        self.aggregate = AGGREGATE_DICT[aggregation]

        if cache_dir is None:
            cache_dir = os.path.join(DEFAULT_CACHE_DIR, "search_cache", "doc")

        if pseudo_encoder_name == "lucene" and type(doc_index) is str:
            self.searcher_doc: LuceneBatchSearcher = LuceneBatchSearcher.from_prebuilt_index(doc_index)
        elif type(pseudo_encoder_name) is str and type(doc_index) is str:
            cache_dir = os.path.join(cache_dir, os.path.split(doc_index)[-1])
            self.searcher_doc: FaissBatchSearcher = FaissBatchSearcher(
                prebuilt_index_name=doc_index,
                encoder_name=pseudo_encoder_name,
                device=device,
                prf_depth=pseudo_prf_depth,
                prf_method=pseudo_prf_method,
                rocchio_alpha=pseudo_rocchio_alpha,
                rocchio_beta=pseudo_rocchio_beta,
                rocchio_gamma=pseudo_rocchio_gamma,
                rocchio_topk=pseudo_rocchio_topk,
                rocchio_bottomk=pseudo_rocchio_bottomk,
                normalize_method=pseudo_normalize_method,
                normalize_scale=pseudo_normalize_scale,
                normalize_shift=pseudo_normalize_shift,
                cache_dir=cache_dir
            )
        elif type(pseudo_encoder_name) is list and type(doc_index) is list:
            self.searcher_doc: List[FaissBatchSearcher] = [FaissBatchSearcher(
                prebuilt_index_name=index_name,
                encoder_name=encoder_name,
                device=device,
                prf_depth=pseudo_prf_depth,
                prf_method=pseudo_prf_method,
                rocchio_alpha=pseudo_rocchio_alpha,
                rocchio_beta=pseudo_rocchio_beta,
                rocchio_gamma=pseudo_rocchio_gamma,
                rocchio_topk=pseudo_rocchio_topk,
                rocchio_bottomk=pseudo_rocchio_bottomk,
                normalize_method=pseudo_normalize_method,
                normalize_scale=pseudo_normalize_scale,
                normalize_shift=pseudo_normalize_shift,
                cache_dir=os.path.join(cache_dir, os.path.split(index_name)[-1])
            ) for index_name, encoder_name in zip(doc_index, pseudo_encoder_name)]
        else:
            raise ValueError("Unexpected pseudo_encoder_name and doc_index")

        if sparse_alpha > 0:
            prebuilt_index = DENSE_TO_SPARSE[doc_index]
            self.sparse_alpha = sparse_alpha
            self.sparse_index_reader = IndexReader.from_prebuilt_index(prebuilt_index)
        else:
            self.sparse_alpha = 0

    def batch_search(
            self,
            batch_queries: List[str],
            batch_qids: List[str],
            num_pseudo_queries: int = 4,
            num_pseudo_return_hits: int = 1000,
            num_return_hits: int = 1000,
            return_pseudo_hits: bool = False,
            threads: int = 1,
    ) -> Union[List, Tuple[List, Dict]]:
        """Search the collection concurrently for multiple queries, using multiple threads.

        Parameters
        ----------
        batch_queries : List[str]
            List of query strings.
        batch_qids:
            List of query ids.
        num_pseudo_queries : int
            Number of buffer to return.
        num_return_hits : int
            Number of hits to return.
        num_pseudo_return_hits : int
            Number of hits to return by each pseudo query.
        threads : int
            Maximum number of threads to use.
        return_pseudo_hits:
            whether return the pseudo queries hit in the first stage.

        Returns
        -------

        """

        # Get pseudo queries
        batch_pseudo_hits = self.searcher_pseudo.batch_search(
            batch_queries, batch_qids,
            k=num_pseudo_queries,
            threads=threads
        )

        # build pseudo query to ids mapping
        pseudo_ids_texts = dict()
        for pseudo_hits in batch_pseudo_hits.values():
            for hit in pseudo_hits:
                if hit.docid not in pseudo_ids_texts:
                    pseudo_ids_texts[hit.docid] = hit.contents

        # Perform pseudo query searching
        pseudo_results = dict()
        pseudo_ids, pseudo_texts = zip(*pseudo_ids_texts.items())
        if type(self.searcher_doc) is list:  # If multi-hybrid searcher
            for searcher in self.searcher_doc:
                results = searcher.batch_search(pseudo_texts, pseudo_ids, k=num_pseudo_return_hits, threads=threads)
                for key, hits in results.items():
                    current_hits = pseudo_results.get(key, [])
                    pseudo_results[key] = current_hits + hits
        else:  # If single searcher
            pseudo_results = self.searcher_doc.batch_search(pseudo_texts, pseudo_ids, k=num_pseudo_return_hits, threads=threads)

        # If interpolation of sparse score is needed
        if self.sparse_alpha > 0:
            qid2query = {qid: query for qid, query in zip(batch_qids, batch_qids)}
            for query_id, pseudo_hits in batch_pseudo_hits.items():
                for pseudo_hit in pseudo_hits:
                    for doc_hit in pseudo_results[pseudo_hit.docid]:
                        query = qid2query[query_id]
                        sparse_score = self.sparse_index_reader.compute_query_document_score(doc_hit.docid, query)
                        doc_hit.score += self.sparse_alpha * sparse_score

        # Aggregate and generate final results
        final_results = list()
        for query_id, pseudo_hits in batch_pseudo_hits.items():
            doc_hits = dict()

            for pseudo_hit in pseudo_hits:
                pseudo_id = pseudo_hit.docid
                pseudo_score = pseudo_hit.score
                for doc_hit in pseudo_results[pseudo_id]:
                    doc_id = doc_hit.docid
                    doc_score = doc_hit.score
                    if doc_id not in doc_hits:
                        doc_hits[doc_id] = [(doc_score, pseudo_score)]
                    else:
                        doc_hits[doc_id].append((doc_score, pseudo_score))
                    # Each final document correspond to a set of pseudo queries which hit it

            for doc_id, pseudo_doc_hits in doc_hits.items():
                doc_hits[doc_id] = self.aggregate(pseudo_doc_hits)
            doc_hits = sorted([(v, k) for k, v in doc_hits.items()], reverse=True)

            if len(doc_hits) < num_return_hits:
                print(f"Warning, query of id {query_id} has less than {num_return_hits} candidate passages.")
            doc_hits = doc_hits[:num_return_hits]

            doc_hits = [SearchResult(str(idx), score, None) for score, idx in doc_hits]
            final_results.append((query_id, doc_hits))

        if return_pseudo_hits:
            return final_results, batch_pseudo_hits
        else:
            return final_results


def main(
        topic_name: str = 'msmarco-passage-dev-subset',
        query_rm3: bool = False,
        query_rocchio: bool = False,
        query_rocchio_use_negative: bool = False,
        query_normalize_method: str = None,
        query_normalize_scale: float = 1,
        query_normalize_shift: float = 0,
        pseudo_name: str = 'msmarco_v1_passage_doc2query-t5_expansions_5',
        pseudo_index_dir: str = None,
        num_pseudo_queries: int = 2,
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
        pseudo_normalize_method: str = None,
        pseudo_normalize_scale: float = 1,
        pseudo_normalize_shift: float = 0,
        sparse_alpha: float = 0,
        aggregation: str = "softmax_sum_with_count",
        doc_index: Union[str, List[str]] = 'msmarco-v1-passage-full',
        num_return_hits: int = 1000,
        threads: int = cpu_count(),
        batch_size: int = cpu_count(),
        device: str = "cpu",
        output_path: str = os.path.join(DEFAULT_CACHE_DIR, "runs"),
        do_eval: bool = True,
        log_pseudo_hits: bool = False,
):
    """

    :param topic_name: Name of topics.
    :param query_rm3: whether the rm3 algorithm used for the first stage search.
    :param query_rocchio: whether the rocchio algorithm used for the first stage search.
    :param query_rocchio_use_negative: whether the rocchio algorithm with negative used for the first stage search.
    :param query_normalize_method: way of normalize the score of pseudo query searcher
    :param query_normalize_shift: corresponding shift of normalization
    :param query_normalize_scale: corresponding scale of normalization
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
    :param pseudo_normalize_method: Way of normalize the score of document searcher
    :param pseudo_normalize_shift: corresponding shift of normalization
    :param pseudo_normalize_scale: corresponding scale of normalization
    :param sparse_alpha: alpha of sparse interpolation
    :param aggregation: the way of aggregate hits from different pseudo queries
    :param doc_index: the index of the candidate documents
    :param num_return_hits: how many hits will be returned
    :param threads: maximum threads to use during search
    :param batch_size: batch size used for the batch search.
    :param device: the device the whole search procedure will on
    :param output_path: the path where the run file will be outputted
    :param do_eval: do evaluation step after search or not
    :param log_pseudo_hits: write pseudo query hit results or not.
    """
    if pseudo_name is not None:
        if pseudo_index_dir is not None:
            raise ValueError("Can not specify both pseudo_name and pseudo_index")
        else:
            pseudo_index_dir = os.path.join(DEFAULT_CACHE_DIR, 'indexes', pseudo_name)
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
        add_query_to_pseudo=add_query_to_pseudo,
        query_normalize_method=query_normalize_method,
        query_normalize_scale=query_normalize_scale,
        query_normalize_shift=query_normalize_shift,
        pseudo_encoder_name=pseudo_encoder_name,
        pseudo_prf_depth=pseudo_prf_depth,
        pseudo_prf_method=pseudo_prf_method,
        pseudo_rocchio_alpha=pseudo_rocchio_alpha,
        pseudo_rocchio_beta=pseudo_rocchio_beta,
        pseudo_rocchio_gamma=pseudo_rocchio_gamma,
        pseudo_rocchio_topk=pseudo_rocchio_topk,
        pseudo_rocchio_bottomk=pseudo_rocchio_bottomk,
        pseudo_normalize_method=pseudo_normalize_method,
        pseudo_normalize_scale=pseudo_normalize_scale,
        pseudo_normalize_shift=pseudo_normalize_shift,
        sparse_alpha=sparse_alpha,
        aggregation=aggregation,
        device=device
    )

    output_path = os.path.join(DEFAULT_CACHE_DIR, output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if type(pseudo_encoder_name) is str:
        pseudo_encoder_full_name = pseudo_encoder_name.split('/')[-1]
        if pseudo_prf_depth is not None:
            pseudo_encoder_full_name += "-" + pseudo_prf_method
    else:
        pseudo_encoder_full_name = "hybrid"

    run_name = f"run.{pseudo_name}.{topic_name}.{num_pseudo_queries}.{pseudo_encoder_full_name}.{aggregation}.txt"
    run_path = os.path.join(output_path, run_name)
    output_writer = TrecWriter(run_path, 'w', max_hits=num_return_hits, tag=output_path[:-4], topics=topics)

    if log_pseudo_hits:
        log_name = f"run.{pseudo_name}.{topic_name}.{num_pseudo_queries}.log"
        log_path = os.path.join(output_path, "log")
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = open(os.path.join(log_path, log_name), mode='w')
    else:
        log_file = None

    with output_writer:
        batch_queries, batch_queries_ids = list(), list()
        for index, (query_id, text) in enumerate(tqdm(query_iterator)):
            batch_queries_ids.append(str(query_id))
            batch_queries.append(text)

            if (index + 1) % batch_size == 0 or index == topics_length - 1:
                batch_hits, batch_pseudo_hits = searcher.batch_search(
                    batch_queries, batch_queries_ids,
                    num_pseudo_queries=num_pseudo_queries,
                    num_pseudo_return_hits=num_pseudo_return_hits,
                    num_return_hits=num_return_hits,
                    threads=threads,
                    return_pseudo_hits=True,
                )

                for topic, hits in batch_hits:
                    output_writer.write(topic, hits)

                if log_file is not None:
                    for topic, hits in batch_pseudo_hits.items():
                        query = batch_queries[batch_queries_ids.index(topic)]
                        dump_dict = {
                            query: [{
                                "id": hit.docid,
                                "score": hit.score,
                                "contents": hit.contents
                            } for hit in hits]
                        }
                        log_file.write(json.dumps(dump_dict, indent=4) + '\n')

                batch_queries_ids.clear()
                batch_queries.clear()

    if do_eval:
        evaluate(topic_name=topic_name, path_to_candidate=run_path)


if __name__ == '__main__':
    CLI(main)
