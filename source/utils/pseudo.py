# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      pseudo.py
@Author:    Rosenberg
@Date:      2023/1/6 14:51 
@Documentation: 
    ...
"""
import os
from typing import List, Tuple, Union

import numpy as np
from scipy.special import softmax

from source import BatchSearchResult, DEFAULT_CACHE_DIR, SearchResult
from source.utils.hybrid import HybridBatchSearcher
from source.utils.sparse import LuceneBatchSearcher


class PseudoQuerySearcher:
    def __init__(
            self,
            pseudo_index_dir: str,
            pseudo_doc_index: Union[str, List[str]],
            pseudo_encoder_name: Union[List[str], str] = "lucene",
            pseudo_prf_depth: int = 0,
            pseudo_prf_method: str = 'avg',
            pseudo_rocchio_alpha: float = 0.9,
            pseudo_rocchio_beta: float = 0.1,
            pseudo_rocchio_gamma: float = 0.1,
            pseudo_rocchio_topk: int = 3,
            pseudo_rocchio_bottomk: int = 0,
            query_index: str = None,
            query_k1: float = None,
            query_b: float = None,
            query_rm3: bool = False,
            query_rocchio: bool = False,
            query_rocchio_use_negative: bool = False,
            cache_dir: str = os.path.join(DEFAULT_CACHE_DIR, "search_cache"),
            device: str = "cpu",
    ):
        self.device = device

        # Set up searcher from query to pseudo query
        self.searcher_pseudo = LuceneBatchSearcher(pseudo_index_dir)

        self.searcher_query = None
        if query_index:
            self.searcher_query = LuceneBatchSearcher(
                query_index,
                k1=query_k1,
                b=query_b,
                rm3=query_rm3,
                rocchio=query_rocchio,
                rocchio_use_negative=query_rocchio_use_negative,
            )

        # Set up searcher from pseudo query to document
        if type(pseudo_encoder_name) is str and type(pseudo_doc_index) is str:
            pseudo_doc_index = [pseudo_doc_index]
            pseudo_encoder_name = [pseudo_encoder_name]

        self.searcher_doc = HybridBatchSearcher(
            prebuilt_index_names=pseudo_doc_index,
            encoder_names=pseudo_encoder_name,
            device=device,
            prf_depth=pseudo_prf_depth,
            prf_method=pseudo_prf_method,
            rocchio_alpha=pseudo_rocchio_alpha,
            rocchio_beta=pseudo_rocchio_beta,
            rocchio_gamma=pseudo_rocchio_gamma,
            rocchio_topk=pseudo_rocchio_topk,
            rocchio_bottomk=pseudo_rocchio_bottomk,
            cache_base_dir=cache_dir,
        )

    def batch_search(
            self,
            batch_queries: List[str],
            batch_qids: List[str],
            num_pseudo_queries: int = 4,
            num_pseudo_return_hits: int = 1000,
            use_cache: bool = True,
            threads: int = 1,
    ) -> Union[BatchSearchResult, Tuple[BatchSearchResult, BatchSearchResult]]:
        """Search the collection concurrently for multiple queries, using multiple threads.

        Parameters
        ----------
        batch_queries : List[str]
            List of query strings.
        batch_qids:
            List of query ids.
        num_pseudo_queries : int
            Number of buffer to return.
        num_pseudo_return_hits: int
            Number of hits to return by each pseudo query.
        use_cache : int
            whether the cached result is used for searching
        threads : int
            Maximum number of threads to use.

        Returns
        -------

        """

        # Get pseudo queries
        batch_pseudo_hits: BatchSearchResult = self.searcher_pseudo.batch_search(
            batch_queries, batch_qids,
            k=num_pseudo_queries,
            threads=threads,
        )  # hits of query, result in pseudo query

        # build pseudo query id2text mapping
        pseudo_ids_texts = dict()
        for pseudo_hits in batch_pseudo_hits.values():
            for pseudo_hit in pseudo_hits:  # Implicitly deduplicate pseudo queries
                pseudo_ids_texts[pseudo_hit.docid] = pseudo_hit.contents

        # Perform pseudo query searching
        pseudo_ids, pseudo_texts = zip(*pseudo_ids_texts.items())
        total_doc_hits = self.searcher_doc.batch_search(
            pseudo_texts, pseudo_ids,
            k=num_pseudo_return_hits,
            threads=threads,
            use_cache=use_cache,
        )  # hits of pseudo query, result in documents

        # min-max normalization
        for key, doc_hits in total_doc_hits.items():
            max_score, min_score = -1e8, 1e8
            for doc_hit in doc_hits:
                doc_score = doc_hit.score
                if doc_score < min_score:
                    min_score = doc_score
                if doc_score > max_score:
                    max_score = doc_score

            norm = max_score - min_score
            for doc_hit in doc_hits:
                doc_hit.score = (doc_hit.score - min_score) / norm

        if self.searcher_query is not None:
            batch_doc_hits = self.searcher_query.batch_search(
                batch_queries, batch_qids,
                k=num_pseudo_return_hits,
                threads=threads,
            )

            for query, qid in zip(batch_queries, batch_qids):
                score = max(hit.score for hit in batch_pseudo_hits[qid])
                batch_pseudo_hits[qid].append(SearchResult(qid, score, query))
                total_doc_hits[qid] = batch_doc_hits[qid]

        encoder_names = self.searcher_doc.encoder_names
        encoder_names_len = len(encoder_names)

        # Aggregate and generate final results
        batch_final_hits: BatchSearchResult = dict()
        for query_id, pseudo_hits in batch_pseudo_hits.items():
            pseudo_ids, pseudo_texts, pseudo_scores = list(), list(), list()
            for pseudo_hit in pseudo_hits:
                pseudo_ids.append(pseudo_hit.docid)
                pseudo_texts.append(pseudo_hit.contents)
                pseudo_scores.append(pseudo_hit.score)

            doc_ids, doc_score_mappings = set(), list()
            for pseudo_id in pseudo_ids:
                if len(pseudo_ids) > 1 and pseudo_id == query_id:  # len(pseudo_ids) = 1 means original query directly used
                    mapping = dict()
                    for doc_hit in total_doc_hits[pseudo_id]:
                        doc_id, doc_score = doc_hit.docid, doc_hit.score
                        doc_ids.add(doc_hit.docid)
                        mapping[doc_id] = doc_score
                    doc_score_mappings.append(mapping)
                else:
                    for name in encoder_names:
                        mapping = dict()
                        for doc_hit in total_doc_hits[pseudo_id, name]:
                            doc_id, doc_score = doc_hit.docid, doc_hit.score
                            doc_ids.add(doc_hit.docid)
                            mapping[doc_id] = doc_score
                        doc_score_mappings.append(mapping)

            if self.searcher_query is not None:
                pseudo_scores = np.asarray(pseudo_scores[:-1] * encoder_names_len + [pseudo_scores[-1]])
            else:
                pseudo_scores = np.asarray(pseudo_scores).repeat(encoder_names_len)
            doc_scores = np.asarray([
                [
                    mapping.get(id, 0)
                    for id in doc_ids
                ] for mapping in doc_score_mappings
            ])

            # Aggregation
            pseudo_scores = np.expand_dims(softmax(pseudo_scores, axis=0), axis=1)
            doc_scores = np.sum(doc_scores * pseudo_scores, axis=0)

            final_hits = [SearchResult(doc_id, doc_score, None) for doc_id, doc_score in zip(doc_ids, doc_scores)]
            final_hits = sorted(final_hits, key=lambda hit: hit.score, reverse=True)
            batch_final_hits[query_id] = final_hits

        return batch_final_hits, batch_pseudo_hits
