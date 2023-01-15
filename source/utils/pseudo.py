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

import torch

from source import BatchSearchResult, DEFAULT_CACHE_DIR
from source.utils import SearchResult
from source.utils.faiss import HybridBatchSearcher
from source.utils.lucene import LuceneBatchSearcher

DENSE_TO_SPARSE = {
    "msmarco-passage-tct_colbert-v2-hnp-bf": "msmarco-v1-passage-d2q-t5-docvectors",
    "msmarco-passage-ance-bf": "msmarco-v1-passage-d2q-t5-docvectors",
}


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
            cache_dir: str = None,
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
        if cache_dir is None:
            cache_dir = os.path.join(DEFAULT_CACHE_DIR, "search_cache", "doc")

        if type(pseudo_encoder_name) is str and type(doc_index) is str:
            doc_index = [doc_index]
            pseudo_encoder_name = [pseudo_encoder_name]

        self.searcher_pseudo: HybridBatchSearcher = HybridBatchSearcher(
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
            num_return_hits: int = 1000,
            return_pseudo_hits: bool = False,
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

        # Aggregate and generate final results
        batch_final_hits: BatchSearchResult = dict()
        for i, (query_id, query_hits) in enumerate(batch_query_hits.items()):
            pseudo2score = dict()  # from pseudo query to query hit score
            doc2scores = dict()  # from document to hit scores

            for query_hit in query_hits:
                pseudo_id, pseudo_score = query_hit.docid, query_hit.score
                pseudo2score[pseudo_id] = pseudo_score

                for name, pseudo_hits in total_pseudo_hits[pseudo_id].items():
                    for pseudo_hit in pseudo_hits:
                        doc_id, doc_score = pseudo_hit.docid, pseudo_hit.score
                        if doc_id not in doc2scores:
                            doc2scores[doc_id] = {name: {pseudo_id: doc_score}}
                        elif name not in doc2scores[doc_id]:
                            doc2scores[doc_id][name] = {pseudo_id: doc_score}
                        else:
                            doc2scores[doc_id][name][pseudo_id] = doc_score

            total_pseudo_mins = {
                q_id: {
                    name: min(hit.score for hit in hits)
                    for name, hits in query_hits.items()
                } for q_id, query_hits in total_pseudo_hits.items()
            }

            doc_ids, doc_scores_matrix = list(), list()
            for doc_id, doc2scores_ in doc2scores.items():  # generate padding score list
                doc_ids.append(doc_id)
                scores_matrix = list()

                for name in tuple(total_pseudo_hits.values())[0].keys():
                    if name not in doc2scores_:
                        scores_matrix.append([
                            total_pseudo_mins[pseudo_id][name]
                            for pseudo_id in pseudo2score.keys()
                        ])
                    else:
                        scores_matrix.append([
                            doc2scores_[name].get(pseudo_id, total_pseudo_mins[pseudo_id][name])
                            for pseudo_id in pseudo2score.keys()
                        ])

                doc_scores_matrix.append(scores_matrix)

            pseudo_score = torch.as_tensor(list(pseudo2score.values()), device=self.device)
            doc_scores_matrix = torch.as_tensor(doc_scores_matrix, device=self.device)

            if doc_scores_matrix.dim() == 3:
                pseudo_score = pseudo_score.repeat(doc_scores_matrix.shape[1])
                doc_scores_matrix = doc_scores_matrix.flatten(start_dim=1)

            # Aggregation
            doc_scores_matrix = pseudo_score.unsqueeze(0) + doc_scores_matrix
            max_score = torch.max(doc_scores_matrix, dim=0, keepdim=True)[0]
            min_score = torch.min(doc_scores_matrix, dim=0, keepdim=True)[0]
            doc_scores_matrix = (doc_scores_matrix - (max_score + min_score) / 2) / (max_score - min_score)
            coefficient = torch.softmax(pseudo_score, dim=-1).unsqueeze(0)
            doc_scores_matrix = torch.sum(doc_scores_matrix * coefficient, 1)

            final_hits = [SearchResult(doc_id, doc_score.item(), None) for doc_id, doc_score in zip(doc_ids, doc_scores_matrix)]
            if len(final_hits) < num_return_hits:
                print(f"Warning, query of id {query_id} has less than {num_return_hits} candidate passages.")
            final_hits = sorted(final_hits, key=lambda hit: hit.score, reverse=True)[:num_return_hits]
            batch_final_hits[query_id] = final_hits

        # Return results
        if return_pseudo_hits:
            return batch_final_hits, batch_query_hits
        else:
            return batch_final_hits
