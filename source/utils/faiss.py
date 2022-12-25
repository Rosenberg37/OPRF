# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      faiss.py
@Author:    Rosenberg
@Date:      2022/12/19 18:32 
@Documentation: 
    ...
"""

import os
from typing import Dict, List, Union

import faiss
import numpy as np
import torch
from diskcache import Cache
from pyserini.search import AnceQueryEncoder, AutoQueryEncoder, BprQueryEncoder, DenseVectorAveragePrf, DenseVectorRocchioPrf, DkrrDprQueryEncoder, DprQueryEncoder, FaissSearcher, \
    TctColBertQueryEncoder
from pyserini.util import download_prebuilt_index

from source.utils import normalize_results, SearchResult


class AnceQueryBatchEncoder(AnceQueryEncoder):
    def batch_encode(self, queries: List[str]):
        return self.prf_batch_encode(queries)


class TctColBertQueryBatchEncoder(TctColBertQueryEncoder):
    def batch_encode(self, queries: List[str]):
        max_length = 36  # hardcode for now
        queries = ['[CLS] [Q] ' + query + '[MASK]' * max_length for query in queries]
        inputs = self.tokenizer(
            queries,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt'
        ).to(self.device)
        outputs = self.model(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state[:, 4:, :], dim=1)
        return embeddings.detach().cpu().numpy()


def init_query_encoder(encoder: str, device: str):
    encoder_class_map = {
        "dkrr": DkrrDprQueryEncoder,
        "dpr": DprQueryEncoder,
        "bpr": BprQueryEncoder,
        "tct_colbert": TctColBertQueryBatchEncoder,
        "ance": AnceQueryBatchEncoder,
        "sentence": AutoQueryEncoder,
        "auto": AutoQueryEncoder,
    }
    if encoder:
        encoder_class = None

        # determine encoder_class
        if encoder_class is not None:
            encoder_class = encoder_class_map[encoder_class]
        else:
            # if any class keyword was matched in the given encoder name,
            # use that encoder class
            for class_keyword in encoder_class_map:
                if class_keyword in encoder.lower():
                    encoder_class = encoder_class_map[class_keyword]
                    break

            # if none of the class keyword was matched,
            # use the AutoQueryEncoder
            if encoder_class is None:
                encoder_class = AutoQueryEncoder

        # prepare arguments to encoder class
        kwargs = dict(encoder_dir=encoder, device=device)
        if (encoder_class == "sentence") or ("sentence" in encoder):
            kwargs.update(dict(pooling='mean', l2_norm=True))

        return encoder_class(**kwargs)


class FaissBatchSearcher(FaissSearcher):
    def __init__(
            self,
            prebuilt_index_name: str,
            encoder_name: str,
            device: str,
            prf_depth: int = 0,
            prf_method: str = 'avg',
            rocchio_alpha: float = 0.9,
            rocchio_beta: float = 0.1,
            rocchio_gamma: float = 0.1,
            rocchio_topk: int = 3,
            rocchio_bottomk: int = 0,
            normalize_score: bool = False,
            cache_dir: str = None
    ):
        query_encoder = init_query_encoder(encoder_name, device)
        print(f'Attempting to initialize pre-built index {prebuilt_index_name}.')
        index_dir = download_prebuilt_index(prebuilt_index_name)
        print(f'Initializing {prebuilt_index_name}...')
        super().__init__(index_dir, query_encoder, prebuilt_index_name)

        # Check PRF Flag
        if prf_depth > 0:
            self.PRF_FLAG = True
            self.prf_depth = prf_depth

            if prf_method.lower() == 'avg':
                self.prfRule = DenseVectorAveragePrf()
            elif prf_method.lower() == 'rocchio':
                self.prfRule = DenseVectorRocchioPrf(
                    rocchio_alpha,
                    rocchio_beta,
                    rocchio_gamma,
                    rocchio_topk,
                    rocchio_bottomk
                )
            else:
                raise ValueError("Unexpected pseudo_prf_method.")
        else:
            self.PRF_FLAG = False

        self.cache_dir = os.path.join(cache_dir, encoder_name)
        if prf_depth > 0:
            self.cache_dir = os.path.join(cache_dir, prf_method)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        print(f"Use case at {self.cache_dir}")
        self.cache = Cache(self.cache_dir, eviction_policy='none')

        self.normalize_score = normalize_score

        # self.searcher_doc.switch_to_IVF()
        # id = device.split(':')[-1]
        # if id == 'cuda':
        #     self.searcher_doc.switch_to_gpu(0)
        # elif id != 'cpu':
        #     self.searcher_doc.switch_to_gpu(int(id))

    def batch_search(
            self,
            queries: Union[List[str], np.ndarray],
            q_ids: List[str], k: int = 10,
            threads: int = 1,
            return_vector: bool = False
    ) -> Dict[str, List[tuple]]:
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
        return_vector : bool
            Return the results with vectors

        Returns
        -------
        Dict[str, list[tuple]]
            return a dict contains key to list of SearchResult
        """

        # Read cache or leave for search
        results = dict()
        search_queries, search_q_ids = list(), list()
        for q_id, query in zip(q_ids, queries):
            result = self.cache.get(q_id, None)
            if result is not None and len(result) >= k:
                results[q_id] = result
            else:
                search_q_ids.append(q_id)
                search_queries.append(query)

        # Search for un-cased pseudo queries
        if len(search_q_ids) > 0:
            if self.PRF_FLAG:
                q_embs, prf_candidates = super().batch_search(search_queries, search_q_ids, k=self.prf_depth, return_vector=True)
                prf_embs_q = self.prfRule.get_batch_prf_q_emb(search_q_ids, q_embs, prf_candidates)
                search_results = super().batch_search(prf_embs_q, search_q_ids, k=k, threads=threads)
            else:
                q_embs = self.query_encoder.batch_encode(queries) if type(queries) is tuple else queries
                search_results = super().batch_search(q_embs, search_q_ids, k=k, threads=threads)

            for id, hits in search_results.items():
                hits = [SearchResult(hit.docid, hit.score, None) for hit in hits]
                results[id] = hits
                self.cache.set(id, hits)

        if self.normalize_score:
            results = normalize_results(results)

        return results

    def switch_to_gpu(self, id: int):
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, id, self.index)

    def switch_to_IVF(self):
        self.index = faiss.IndexIVFFlat(self.index, self.dimension, 256)
