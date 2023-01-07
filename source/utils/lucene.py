# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      lucene.py
@Author:    Rosenberg
@Date:      2022/12/24 17:08 
@Documentation: 
    ...
"""
import json
from typing import Dict, List

from pyserini.index import IndexReader
from pyserini.search import LuceneSearcher
from pyserini.search.lucene.__main__ import set_bm25_parameters
from pyserini.util import download_prebuilt_index

from source.utils import SearchResult


class LuceneBatchSearcher:
    def __init__(
            self,
            index_dir: str,
            rm3: bool = False,
            rocchio: bool = False,
            rocchio_use_negative: bool = False,
    ):
        self.searcher = LuceneSearcher(index_dir)
        set_bm25_parameters(self.searcher, None, 2.56, 0.59)

        if rm3:
            self.searcher.set_rm3()

        if rocchio:
            if rocchio_use_negative:
                self.searcher.set_rocchio(gamma=0.15, use_negative=True)
            else:
                self.searcher.set_rocchio()

    def batch_search(
            self,
            queries: List[str],
            qids: List[str],
            k: int = 10,
            threads: int = 1,
            add_query_to_pseudo: bool = False,
    ) -> Dict[str, List[SearchResult]]:
        if k <= 0:
            print("Warning, num_pseudo_queries less or equal zero, set pseudo query directly to be query.\n")

            batch_hits = dict()
            for contents, qid in zip(queries, qids):
                batch_hits[qid] = [SearchResult(qid, 1., contents)]
        else:
            batch_hits = self.searcher.batch_search(queries, qids, k, threads)
            for key, hits in batch_hits.items():
                batch_hits[key] = [
                    SearchResult(
                        hit.docid, hit.score,
                        json.loads(hit.raw)['contents'] if hit.raw else None
                    ) for hit in hits
                ]

            if add_query_to_pseudo:
                for contents, qid in zip(queries, qids):
                    query_score = max(hit.score for hit in batch_hits[qid])
                    batch_hits[qid].append(SearchResult(qid, query_score, contents))

        return batch_hits

    @classmethod
    def from_prebuilt_index(
            cls,
            prebuilt_index_name: str,
            rm3: bool = False,
            rocchio: bool = False,
            rocchio_use_negative: bool = False,
    ):
        try:
            index_dir = download_prebuilt_index(prebuilt_index_name)
        except ValueError as e:
            print(str(e))
            return None

        index_reader = IndexReader(index_dir)
        index_reader.validate(prebuilt_index_name)

        return cls(index_dir, rm3, rocchio, rocchio_use_negative)
