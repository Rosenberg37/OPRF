# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      buffer_search.py
@Author:    Rosenberg
@Date:      2022/12/6 10:42 
@Documentation: 
    ...
"""
import json
import os
import os.path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import faiss
import numpy as np
from diskcache import Cache
from math import exp
from pyserini.search import AnceQueryEncoder, AutoQueryEncoder, BprQueryEncoder, DenseSearchResult, DenseVectorAncePrf, DenseVectorAveragePrf, DenseVectorRocchioPrf, DkrrDprQueryEncoder, \
    DprQueryEncoder, FaissSearcher, PRFDenseSearchResult, TctColBertQueryEncoder
from pyserini.search.lucene import LuceneSearcher


@dataclass
class SearchResult:
    docid: str
    score: float


class FaissBatchSearcher(FaissSearcher):

    def batch_search(
            self,
            queries: Union[List[str], np.ndarray],
            q_ids: List[str],
            k: int = 10,
            threads: int = 1,
            return_vector: bool = False
    ) -> Union[Dict[str, List[tuple]], Tuple[np.ndarray, Dict[str, List[PRFDenseSearchResult]]]]:
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
        Union[Dict[str, List[DenseSearchResult]], Tuple[np.ndarray, Dict[str, List[PRFDenseSearchResult]]]]
            Either returns a dictionary holding the search results, with the query ids as keys and the
            corresponding lists of search results as the values.
            Or returns a tuple with ndarray of query vectors and a dictionary of PRF Dense Search Results with vectors
        """
        q_embs = self.query_encoder.prf_batch_encode(queries)
        faiss.omp_set_num_threads(threads)
        if return_vector:
            d, i, v = self.index.search_and_reconstruct(q_embs, k)
            return q_embs, {key: [PRFDenseSearchResult(self.docids[idx], score, vector)
                                  for score, idx, vector in zip(distances, indexes, vectors) if idx != -1]
                            for key, distances, indexes, vectors in zip(q_ids, d, i, v)}
        else:
            d, i = self.index.search(q_embs, k)
            return {key: [(score, self.docids[idx])
                          for score, idx in zip(distances, indexes) if idx != -1]
                    for key, distances, indexes in zip(q_ids, d, i)}

    def switch_to_gpu(self, id: int):
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, id, self.index)

    def switch_to_IVF(self):
        self.index = faiss.IndexIVFFlat(self.index, self.dimension, 256)


class PseudoQuerySearcher:
    DEFAULT_BUFFER_DIR = os.path.join(os.path.expanduser('~'), ".cache", "pprf")

    @staticmethod
    def init_query_encoder(encoder: str, device: str):
        encoder_class_map = {
            "dkrr": DkrrDprQueryEncoder,
            "dpr": DprQueryEncoder,
            "bpr": BprQueryEncoder,
            "tct_colbert": TctColBertQueryEncoder,
            "ance": AnceQueryEncoder,
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

    def __init__(
            self,
            pseudo_index_dir: str,
            doc_index: str,
            query_rm3: bool = False,
            query_rocchio: bool = False,
            query_rocchio_use_negative: bool = False,
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
            buffer_dir: str = None,
            device: str = "cuda:0",
    ):
        self.searcher_pseudo = LuceneSearcher(pseudo_index_dir)
        if query_rm3:
            self.searcher_pseudo.set_rm3()
        if query_rocchio:
            if query_rocchio_use_negative:
                self.searcher_pseudo.set_rocchio(gamma=0.15, use_negative=True)
            else:
                self.searcher_pseudo.set_rocchio()

        encoder_name = pseudo_encoder_name.split('/')[-1]
        if encoder_name == "lucene":
            self.pseudo_encoder = None
            self.searcher_doc: LuceneSearcher = LuceneSearcher.from_prebuilt_index(doc_index)
        else:
            self.pseudo_encoder = self.init_query_encoder(pseudo_encoder_name, device)
            if os.path.exists(doc_index):
                # create searcher from index directory
                self.searcher_doc = FaissBatchSearcher(doc_index, self.pseudo_encoder)
            else:
                # create searcher from prebuilt index name
                self.searcher_doc = FaissBatchSearcher.from_prebuilt_index(doc_index, self.pseudo_encoder)

        # self.searcher_doc.switch_to_IVF()
        # id = device.split(':')[-1]
        # if id == 'cuda':
        #     self.searcher_doc.switch_to_gpu(0)
        # elif id != 'cpu':
        #     self.searcher_doc.switch_to_gpu(int(id))

        # Check PRF Flag
        if pseudo_prf_depth > 0 and type(self.searcher_doc) == FaissSearcher:
            self.PRF_FLAG = True
            self.prf_method = pseudo_prf_method
            self.prf_depth = pseudo_prf_depth

            if pseudo_prf_method.lower() == 'avg':
                self.prfRule = DenseVectorAveragePrf()
            elif pseudo_prf_method.lower() == 'rocchio':
                self.prfRule = DenseVectorRocchioPrf(pseudo_rocchio_alpha, pseudo_rocchio_beta, pseudo_rocchio_gamma, pseudo_rocchio_topk, pseudo_rocchio_bottomk)
            # ANCE-PRF is using a new query encoder, so the input to DenseVectorAncePrf is different
            elif pseudo_prf_method.lower() == 'ance-prf' and type(self.pseudo_encoder) == AnceQueryEncoder:
                if os.path.exists(pseudo_sparse_index):
                    self.sparse_searcher = LuceneSearcher(pseudo_sparse_index)
                else:
                    self.sparse_searcher = LuceneSearcher.from_prebuilt_index(pseudo_sparse_index)
                self.prf_query_encoder = AnceQueryEncoder(encoder_dir=pseudo_ance_prf_encoder, tokenizer_name=pseudo_tokenizer, device=device)
                self.prfRule = DenseVectorAncePrf(self.prf_query_encoder, self.sparse_searcher)
        else:
            self.PRF_FLAG = False

        self.cache_dir = self.DEFAULT_BUFFER_DIR if buffer_dir is None else buffer_dir
        index_name = os.path.split(doc_index)[-1]
        self.cache_dir = os.path.join(self.cache_dir, index_name, encoder_name)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.cache = Cache(self.cache_dir, eviction_policy='none')

    def batch_search(
            self,
            batch_queries: List[str],
            batch_qids: List[str],
            num_pseudo_queries: int = 4,
            num_return_hits: int = 1000,
            threads: int = 1,
    ):
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
        threads : int
            Maximum number of threads to use.

        Returns
        -------

        """
        batch_pseudo_hits = self.searcher_pseudo.batch_search(batch_queries, batch_qids, num_pseudo_queries, threads)

        pseudo_ids_texts = dict()
        pseudo_results = dict()

        for pseudo_hits in batch_pseudo_hits.values():
            for hit in pseudo_hits:
                if hit.docid not in pseudo_results:
                    result = self.cache.get(hit.docid, None)
                    if result is not None and len(result) >= num_return_hits:
                        pseudo_results[hit.docid] = result[:num_return_hits]
                    else:
                        pseudo_ids_texts[hit.docid] = json.loads(hit.raw)['contents']

        if len(pseudo_ids_texts) > 0:
            pseudo_ids, pseudo_texts = zip(*pseudo_ids_texts.items())

            if self.PRF_FLAG:
                q_embs, prf_candidates = self.searcher_doc.batch_search(pseudo_texts, pseudo_ids, k=self.prf_depth, return_vector=True)
                # ANCE-PRF input is different, do not need query embeddings
                if self.prf_method.lower() == 'ance-prf':
                    prf_embs_q = self.prfRule.get_batch_prf_q_emb(pseudo_texts, pseudo_ids, prf_candidates)
                else:
                    prf_embs_q = self.prfRule.get_batch_prf_q_emb(pseudo_ids, q_embs, prf_candidates)
                search_results = self.searcher_doc.batch_search(prf_embs_q, pseudo_ids, k=num_return_hits, threads=threads)
                search_results = [(id_, search_results[id_]) for id_ in pseudo_ids]
            else:
                search_results = self.searcher_doc.batch_search(pseudo_texts, pseudo_ids, num_return_hits, threads)

            for pseudo_id, pseudo_doc_hits in search_results.items():
                pseudo_results[pseudo_id] = pseudo_doc_hits
                self.cache.set(pseudo_id, pseudo_doc_hits)

        final_results = list()
        for query_id, pseudo_hits in batch_pseudo_hits.items():
            doc_hits = dict()

            for pseudo_hit in pseudo_hits:
                pseudo_id = pseudo_hit.docid
                pseudo_score = exp(pseudo_hit.score)
                for doc_hit in pseudo_results[pseudo_id]:
                    doc_score, doc_id = doc_hit
                    if doc_id not in doc_hits:
                        doc_hits[doc_id] = [(doc_score, pseudo_score)]
                    else:
                        doc_hits[doc_id].append((doc_score, pseudo_score))

            for doc_id, pseudo_doc_hits in doc_hits.items():
                numerator, denominator = 0, 0
                for s in pseudo_doc_hits:
                    numerator += s[0] * s[1]
                    denominator += s[1]
                doc_hits[doc_id] = numerator / denominator * len(pseudo_doc_hits)

            doc_hits = sorted([(v, k) for k, v in doc_hits.items()], reverse=True)[:num_return_hits]
            doc_hits = [SearchResult(str(idx), score) for score, idx in doc_hits]
            final_results.append((query_id, doc_hits))

        return final_results
