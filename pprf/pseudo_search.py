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
from typing import Dict, List, Tuple, Union

import faiss
import numpy as np
from diskcache import Cache
from math import exp
from pyserini.search import AnceQueryEncoder, AutoQueryEncoder, BprQueryEncoder, DenseSearchResult, DkrrDprQueryEncoder, DprQueryEncoder, FaissSearcher, PRFDenseSearchResult, TctColBertQueryEncoder
from pyserini.search.lucene import LuceneSearcher


class FaissBatchSearcher(FaissSearcher):
    def batch_search(
            self,
            queries: List[str],
            q_ids: List[str],
            k: int = 10,
            threads: int = 1,
            return_vector: bool = False
    ) -> Union[Dict[str, List[DenseSearchResult]], Tuple[np.ndarray, Dict[str, List[PRFDenseSearchResult]]]]:
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
        D, I = self.index.search(q_embs, k)
        return {key: [DenseSearchResult(self.docids[idx], score)
                      for score, idx in zip(distances, indexes) if idx != -1]
                for key, distances, indexes in zip(q_ids, D, I)}


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
            encoder: str = "lucene",
            buffer_dir: str = None,
            device: str = "cuda:0",
    ):
        self.searcher_pseudo = LuceneSearcher(pseudo_index_dir)

        self.encoder = encoder.split('/')[-1]
        if self.encoder == "lucene":
            self.searcher_doc: LuceneSearcher = LuceneSearcher.from_prebuilt_index(doc_index)
        else:
            query_encoder = self.init_query_encoder(encoder, device)
            if os.path.exists(doc_index):
                # create searcher from index directory
                self.searcher_doc = FaissBatchSearcher(doc_index, query_encoder)
            else:
                # create searcher from prebuilt index name
                self.searcher_doc = FaissBatchSearcher.from_prebuilt_index(doc_index, query_encoder)

        self.cache_dir = self.DEFAULT_BUFFER_DIR if buffer_dir is None else buffer_dir
        index_name = os.path.split(doc_index)[-1]
        self.cache_dir = os.path.join(self.cache_dir, index_name, self.encoder)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.cache = Cache(self.cache_dir)

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

        for id_ in batch_qids:
            for hit in batch_pseudo_hits[id_]:
                result = self.cache.get(hit.docid, None)
                if result is not None and len(result) >= num_return_hits:
                    pseudo_results[hit.docid] = result[:num_return_hits]
                else:
                    pseudo_ids_texts[hit.docid] = json.loads(hit.raw)['contents']  # TODO(In-Batch pseudo deduplicate)

        if len(pseudo_ids_texts) > 0:
            pseudo_ids, pseudo_texts = zip(*pseudo_ids_texts.items())
            search_results = self.searcher_doc.batch_search(pseudo_texts, pseudo_ids, num_return_hits, threads)
            for pseudo_id, pseudo_doc_hits in search_results.items():
                pseudo_doc_hits = [{"score": hit.score, "id": hit.docid} for hit in pseudo_doc_hits]
                pseudo_results[pseudo_id] = pseudo_doc_hits
                self.cache.set(pseudo_id, pseudo_doc_hits)

        final_results = list()
        for id_ in batch_qids:
            doc_hits = dict()

            pseudo_hits = batch_pseudo_hits[id_]
            for pseudo_hit in pseudo_hits:
                pseudo_score = pseudo_hit.score
                for doc_hit in pseudo_results[pseudo_hit.docid]:
                    value = doc_hits.get(doc_hit['id'], [])
                    value += [(doc_hit['score'], pseudo_score)]
                    doc_hits[doc_hit['id']] = value

            for doc_id, pseudo_doc_hits in doc_hits.items():
                doc_hits[doc_id] = sum(s[0] * exp(s[1]) for s in pseudo_doc_hits) / sum(exp(s[1]) for s in pseudo_doc_hits) * len(pseudo_doc_hits)
            doc_hits = sorted([(v, k) for k, v in doc_hits.items()], reverse=True)[:num_return_hits]
            final_results.append((id_, doc_hits))

        return final_results
