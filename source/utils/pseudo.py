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
from source.utils.lucene import LuceneBatchSearcher


class PseudoQuerySearcher:
    def __init__(
            self,
            pseudo_index_dir: str,
            doc_index: Union[str, List[str]],
            query_rm3: bool = False,
            query_rocchio: bool = False,
            query_rocchio_use_negative: bool = False,
            add_query_to_pseudo: bool = False,
            pseudo_encoder_name: Union[List[str], str] = "lucene",
            pseudo_prf_depth: int = 0,
            pseudo_prf_method: str = 'avg',
            pseudo_rocchio_alpha: float = 0.9,
            pseudo_rocchio_beta: float = 0.1,
            pseudo_rocchio_gamma: float = 0.1,
            pseudo_rocchio_topk: int = 3,
            pseudo_rocchio_bottomk: int = 0,
            cache_dir: str = os.path.join(DEFAULT_CACHE_DIR, "search_cache"),
            device: str = "cpu",
    ):
        self.device = device

        # Set up searcher from query to pseudo query
        self.searcher_query = LuceneBatchSearcher(
            pseudo_index_dir,
            rm3=query_rm3,
            rocchio=query_rocchio,
            rocchio_use_negative=query_rocchio_use_negative,
        )
        self.add_query_to_pseudo = add_query_to_pseudo

        # Set up searcher from pseudo query to document
        if type(pseudo_encoder_name) is str and type(doc_index) is str:
            doc_index = [doc_index]
            pseudo_encoder_name = [pseudo_encoder_name]

        self.searcher_pseudo = HybridBatchSearcher(
            prebuilt_index_names=doc_index,
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
        num_pseudo_return_hits : int
            Number of hits to return by each pseudo query.
        threads : int
            Maximum number of threads to use.

        Returns
        -------

        """

        # Get pseudo queries
        batch_query_hits: BatchSearchResult = self.searcher_query.batch_search(
            batch_queries, batch_qids,
            k=num_pseudo_queries,
            threads=threads,
            add_query_to_pseudo=self.add_query_to_pseudo,
        )  # hits of query, result in pseudo query

        # build pseudo query id2text mapping
        pseudo_ids_texts = dict()
        for query_hits in batch_query_hits.values():
            for query_hit in query_hits:  # Implicitly deduplicate pseudo queries
                pseudo_ids_texts[query_hit.docid] = query_hit.contents

        # Perform pseudo query searching
        pseudo_ids, pseudo_texts = zip(*pseudo_ids_texts.items())
        total_pseudo_hits = self.searcher_pseudo.batch_search(
            pseudo_texts, pseudo_ids,
            k=num_pseudo_return_hits,
            threads=threads
        )  # hits of pseudo query, result in documents

        for key, pseudo_hits in total_pseudo_hits.items():
            max, min = -1e8, 1e8
            for hit in pseudo_hits:
                doc_score = hit.score
                if doc_score < min:
                    min = doc_score
                if doc_score > max:
                    max = doc_score

            norm = max - min
            for hit in pseudo_hits:
                hit.score = (hit.score - min) / norm

        encoder_names = self.searcher_pseudo.encoder_names
        encoder_names_len = len(encoder_names)

        # Aggregate and generate final results
        batch_final_hits: BatchSearchResult = dict()
        for query_id, query_hits in batch_query_hits.items():
            pseudo_ids, pseudo_texts, pseudo_scores = list(), list(), list()
            for query_hit in query_hits:
                pseudo_ids.append(query_hit.docid)
                pseudo_texts.append(query_hit.contents)
                pseudo_scores.append(query_hit.score)

            doc_ids, doc_score_mappings = set(), list()
            for pseudo_id in pseudo_ids:
                for name in encoder_names:
                    mapping = dict()
                    for hit in total_pseudo_hits[pseudo_id, name]:
                        doc_id, doc_score = hit.docid, hit.score
                        doc_ids.add(hit.docid)
                        mapping[doc_id] = doc_score
                    doc_score_mappings.append(mapping)

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

        return batch_final_hits, batch_query_hits
