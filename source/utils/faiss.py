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
from pyserini.search import AnceQueryEncoder, AutoQueryEncoder, DenseVectorAveragePrf, DenseVectorRocchioPrf, FaissSearcher, \
    QueryEncoder
from transformers import BertModel, BertTokenizerFast

from source.utils import SearchResult
from source.utils.lucene import LuceneBatchSearcher

IMPACT_ENCODERS = {
    "castorini/unicoil-msmarco-passage",
}


class AnceQueryBatchEncoder(AnceQueryEncoder):
    def batch_encode(self, queries: List[str]):
        return self.prf_batch_encode(queries)


class TctColBertQueryBatchEncoder(QueryEncoder):
    def __init__(self, encoder_dir: str = None, tokenizer_name: str = None,
                 encoded_query_dir: str = None, device: str = 'cpu', **kwargs):
        super().__init__(encoded_query_dir)
        if encoder_dir:
            self.device = device
            self.model = BertModel.from_pretrained(encoder_dir)
            self.model.to(self.device)
            self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name or encoder_dir)
            self.has_model = True
        if (not self.has_model) and (not self.has_encoded_query):
            raise Exception('Neither query encoder model nor encoded queries provided. Please provide at least one')

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


class AutoQueryBatchEncoder(AutoQueryEncoder):

    def batch_encode(self, queries: List[str]):
        inputs = self.tokenizer(
            queries,
            add_special_tokens=True,
            return_tensors='pt',
            truncation='only_first',
            padding='longest',
            return_token_type_ids=False,
        )

        inputs.to(self.device)
        outputs = self.model(**inputs)
        if self.pooling == "mean":
            embeddings = self._mean_pooling(outputs, inputs['attention_mask']).detach().cpu().numpy()
        else:
            embeddings = outputs[0][:, 0, :].detach().cpu().numpy()
        if self.l2_norm:
            faiss.normalize_L2(embeddings)
        return embeddings


def init_query_encoder(encoder: str, device: str):
    encoder_class_map = {
        "tct_colbert": TctColBertQueryBatchEncoder,
        "ance": AnceQueryBatchEncoder,
        "auto": AutoQueryBatchEncoder,
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
                encoder_class = AutoQueryBatchEncoder

        # prepare arguments to encoder class
        kwargs = dict(encoder_dir=encoder, device=device)
        if (encoder_class == "sentence") or ("sentence" in encoder):
            kwargs.update(dict(pooling='mean', l2_norm=True))

        return encoder_class(**kwargs)


class FaissBatchSearcher:
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
            cache_dir: str = None
    ):
        self.device = device
        self.encoder_name = encoder_name
        self.prebuilt_index_name = prebuilt_index_name
        self.query_encoder = None
        self.searcher = None

        # Check PRF Flag
        self.prf_rule = None
        if prf_depth > 0 and self.encoder_name != "lucene" and self.encoder_name not in IMPACT_ENCODERS:
            self.prf_depth = prf_depth

            if prf_method.lower() == 'avg':
                self.prf_rule = DenseVectorAveragePrf()
            elif prf_method.lower() == 'rocchio':
                self.prf_rule = DenseVectorRocchioPrf(
                    rocchio_alpha,
                    rocchio_beta,
                    rocchio_gamma,
                    rocchio_topk,
                    rocchio_bottomk
                )
            else:
                raise ValueError("Unexpected pseudo_prf_method.")

            self.cache_dir = os.path.join(cache_dir, encoder_name, prf_method)
        else:
            self.cache_dir = os.path.join(cache_dir, encoder_name)

        # Set up cache
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.cache = Cache(self.cache_dir, eviction_policy='none')

    def init_searcher(self):
        if self.encoder_name == "lucene":
            self.searcher = LuceneBatchSearcher(self.prebuilt_index_name)
        elif self.encoder_name in IMPACT_ENCODERS:
            self.searcher = LuceneBatchSearcher(self.prebuilt_index_name, impact=True, encoder_name=self.encoder_name)
        else:
            self.query_encoder = init_query_encoder(self.encoder_name, self.device)
            self.searcher = FaissSearcher.from_prebuilt_index(self.prebuilt_index_name, self.query_encoder)

    def batch_search(
            self,
            queries: Union[List[str], np.ndarray],
            q_ids: List[str],
            k: int = 10,
            threads: int = 1,
    ) -> Dict[str, List[SearchResult]]:
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

        # Read cache or leave for search
        batch_hits = dict()
        search_queries, search_q_ids = list(), list()
        for q_id, query in zip(q_ids, queries):
            result = self.cache.get(q_id, None)
            if result is not None and len(result) >= k:
                batch_hits[q_id] = result[:k]
            else:
                search_q_ids.append(q_id)
                search_queries.append(query)

        # Search for un-cased pseudo queries
        if len(search_q_ids) > 0:
            if self.searcher is None:
                self.init_searcher()

            if type(search_queries) is list and self.query_encoder is not None:
                q_embs = self.query_encoder.batch_encode(search_queries)
            else:
                q_embs = search_queries

            if self.prf_rule is not None and self.query_encoder is not None:
                q_embs, prf_candidates = self.searcher.batch_search(q_embs, search_q_ids, k=self.prf_depth, return_vector=True, threads=threads)
                prf_embs_q = self.prf_rule.get_batch_prf_q_emb(search_q_ids, q_embs, prf_candidates)
                search_hits = self.searcher.batch_search(prf_embs_q, search_q_ids, k=k, threads=threads)
            else:
                search_hits = self.searcher.batch_search(q_embs, search_q_ids, k=k, threads=threads)

            for id, hits in search_hits.items():
                hits = [SearchResult(hit.docid, hit.score, None) for hit in hits]
                batch_hits[id] = hits
                self.cache.set(id, hits)

        return batch_hits
