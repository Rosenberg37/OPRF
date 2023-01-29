# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      hybrid.py
@Author:    Rosenberg
@Date:      2023/1/29 20:05 
@Documentation: 
    ...
"""
import os
from typing import Dict, List, Union

import numpy as np

from source import SearchResult
from source.utils.faiss import FaissBatchSearcher


class HybridBatchSearcher:
    def __init__(
            self,
            prebuilt_index_names: List[str],
            encoder_names: List[str],
            device: str,
            prf_depth: int = 0,
            prf_method: str = 'avg',
            rocchio_alpha: float = 0.9,
            rocchio_beta: float = 0.1,
            rocchio_gamma: float = 0.1,
            rocchio_topk: int = 3,
            rocchio_bottomk: int = 0,
            cache_base_dir: str = None
    ):
        self.searchers = [FaissBatchSearcher(
            prebuilt_index_name=index_name,
            encoder_name=encoder_name,
            device=device,
            prf_depth=prf_depth,
            prf_method=prf_method,
            rocchio_alpha=rocchio_alpha,
            rocchio_beta=rocchio_beta,
            rocchio_gamma=rocchio_gamma,
            rocchio_topk=rocchio_topk,
            rocchio_bottomk=rocchio_bottomk,
            cache_dir=os.path.join(cache_base_dir, os.path.split(index_name)[-1])
        ) for index_name, encoder_name in zip(prebuilt_index_names, encoder_names)]
        self.encoder_names = encoder_names

    def batch_search(
            self,
            queries: Union[List[str], np.ndarray],
            q_ids: List[str],
            k: int = 10,
            threads: int = 1,
    ) -> Dict[str, Dict[str, List[SearchResult]]]:
        """

        Parameters
        ----------
        queries : Union[List[str], np.ndarray]
            List of query texts or list of query embeddings
        q_ids : List[str]
            List of corresponding query ids.
        k : int
            Number of hits to return.
        threads : int
            Maximum number of threads to use.

        Returns
        -------
        Dict[str, list[tuple]]
            return a dict contains key to list of SearchResult
        """
        final_hits = {q_id: {name: None for name in self.encoder_names} for q_id in q_ids}
        for name, searcher in zip(self.encoder_names, self.searchers):
            batch_hits = searcher.batch_search(queries, q_ids, k=k, threads=threads)
            for q_id, hits in batch_hits.items():
                final_hits[q_id][name] = hits

        return final_hits
