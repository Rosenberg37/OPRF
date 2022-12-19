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
from functools import partial
from typing import Dict, List, Tuple

from diskcache import Cache
from math import exp
from pyserini.search import AnceQueryEncoder, DenseVectorAncePrf, DenseVectorAveragePrf, DenseVectorRocchioPrf
from pyserini.search.lucene import LuceneSearcher

from pprf import DEFAULT_BUFFER_DIR
from pprf.dense_search import AnceQueryBatchEncoder, FaissBatchSearcher, init_query_encoder, SearchResult


def softmax_sum(pseudo_doc_hits: List[Tuple], length_correct: bool):
    numerator, denominator = 0, 0
    for s in pseudo_doc_hits:
        exp_pseudo_score = exp(s[1])
        numerator += s[0] * exp_pseudo_score
        denominator += exp_pseudo_score
    result = numerator / denominator
    if length_correct:
        result *= len(pseudo_doc_hits)
    return result


def sum_doc(pseudo_doc_hits: List[Tuple]):
    return sum(s[0] for s in pseudo_doc_hits)


def sum_pseudo(pseudo_doc_hits: List[Tuple]):
    return sum(s[1] for s in pseudo_doc_hits)


def sum_total(pseudo_doc_hits: List[Tuple]):
    return sum(sum(s) for s in pseudo_doc_hits)


class PseudoQuerySearcher:
    DEFAULT_BUFFER_DIR = os.path.join(os.path.expanduser('~'), ".cache", "pprf")
    AGGREGATE_DICT = {
        "softmax_sum_with_count": partial(softmax_sum, length_correct=True),
        "softmax_sum": partial(softmax_sum, length_correct=False),
        "sum_doc": sum_doc,
        "sum_pseudo": sum_pseudo,
        "sum_total": sum_total,
        "count": len,
    }

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
            aggregation: str = "softmax_sum_with_count",
            buffer_dir: str = None,
            device: str = "cpu",
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
            self.pseudo_encoder = init_query_encoder(pseudo_encoder_name, device)
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
        if pseudo_prf_depth > 0 and type(self.searcher_doc) is FaissBatchSearcher:
            self.PRF_FLAG = True
            self.prf_method = pseudo_prf_method
            self.prf_depth = pseudo_prf_depth

            if pseudo_prf_method.lower() == 'avg':
                self.prfRule = DenseVectorAveragePrf()
            elif pseudo_prf_method.lower() == 'rocchio':
                self.prfRule = DenseVectorRocchioPrf(pseudo_rocchio_alpha, pseudo_rocchio_beta, pseudo_rocchio_gamma, pseudo_rocchio_topk, pseudo_rocchio_bottomk)
            # ANCE-PRF is using a new query encoder, so the input to DenseVectorAncePrf is different
            elif pseudo_prf_method.lower() == 'ance-prf' and type(self.pseudo_encoder) == AnceQueryBatchEncoder:
                if os.path.exists(pseudo_sparse_index):
                    self.sparse_searcher = LuceneSearcher(pseudo_sparse_index)
                else:
                    self.sparse_searcher = LuceneSearcher.from_prebuilt_index(pseudo_sparse_index)
                self.prf_query_encoder = AnceQueryEncoder(encoder_dir=pseudo_ance_prf_encoder, tokenizer_name=pseudo_tokenizer, device=device)
                self.prfRule = DenseVectorAncePrf(self.prf_query_encoder, self.sparse_searcher)
        else:
            self.PRF_FLAG = False

        self.aggregate = self.AGGREGATE_DICT[aggregation]

        self.cache_dir = DEFAULT_BUFFER_DIR if buffer_dir is None else buffer_dir
        index_name = os.path.split(doc_index)[-1]
        self.cache_dir = os.path.join(self.cache_dir, index_name, encoder_name)
        if pseudo_prf_depth > 0:
            self.cache_dir = os.path.join(self.cache_dir, pseudo_prf_method)

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.cache = Cache(self.cache_dir, eviction_policy='none')

    def batch_search(
            self,
            batch_queries: List[str],
            batch_qids: List[str],
            num_pseudo_queries: int = 4,
            add_query_to_pseudo: bool = False,
            num_return_hits: int = 1000,
            return_pseudo_hits: bool = False,
            threads: int = 1,
    ) -> [List, Tuple[List, Dict]]:
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
        add_query_to_pseudo: 
            Whether add self to the original pseudo queries.
        return_pseudo_hits:
            whether return the pseudo queries hit in the first stage.

        Returns
        -------

        """
        batch_pseudo_hits = self.searcher_pseudo.batch_search(batch_queries, batch_qids, num_pseudo_queries, threads)
        if add_query_to_pseudo:
            for contents, qid in zip(batch_queries, batch_qids):
                max_score = max(hit.score for hit in batch_pseudo_hits[qid])
                batch_pseudo_hits[qid].append(SearchResult(qid, max_score, contents))

        pseudo_ids_texts, pseudo_results = dict(), dict()
        for pseudo_hits in batch_pseudo_hits.values():
            for hit in pseudo_hits:
                if hit.docid not in pseudo_results:
                    result = self.cache.get(hit.docid, None)
                    if result is not None and len(result) >= num_return_hits:
                        pseudo_results[hit.docid] = result[:num_return_hits]
                    else:
                        pseudo_ids_texts[hit.docid] = json.loads(hit.raw)['contents'] if hit.contents is None else hit.contents

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
            else:
                search_results = self.searcher_doc.batch_search(pseudo_texts, pseudo_ids, num_return_hits, threads)

            for pseudo_id, pseudo_doc_hits in search_results.items():
                pseudo_doc_hits = [(hit.score, hit.docid) for hit in pseudo_doc_hits]
                pseudo_results[pseudo_id] = pseudo_doc_hits
                self.cache.set(pseudo_id, pseudo_doc_hits)

        final_results = list()
        for query_id, pseudo_hits in batch_pseudo_hits.items():
            doc_hits = dict()

            for pseudo_hit in pseudo_hits:
                pseudo_id = pseudo_hit.docid
                pseudo_score = pseudo_hit.score
                for doc_hit in pseudo_results[pseudo_id]:
                    doc_score, doc_id = doc_hit
                    if doc_id not in doc_hits:
                        doc_hits[doc_id] = [(doc_score, pseudo_score)]
                    else:
                        doc_hits[doc_id].append((doc_score, pseudo_score))
                    # Each final document correspond to a set of pseudo queries which hit it

            for doc_id, pseudo_doc_hits in doc_hits.items():
                doc_hits[doc_id] = self.aggregate(pseudo_doc_hits)
            doc_hits = sorted([(v, k) for k, v in doc_hits.items()], reverse=True)[:num_return_hits]

            doc_hits = [SearchResult(str(idx), score, None) for score, idx in doc_hits]
            final_results.append((query_id, doc_hits))

        if return_pseudo_hits:
            return final_results, batch_pseudo_hits
        else:
            return final_results
