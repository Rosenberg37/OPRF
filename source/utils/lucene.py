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
from functools import partial
from typing import Dict, List

from pyserini.search import JQueryGenerator, LuceneSearcher

from source.utils import SearchResult
from source.utils.normalize import NORMALIZE_DICT

DENSE_TO_SPARSE = {
    "msmarco-passage-tct_colbert-v2-hnp-bf": "msmarco-v1-passage-d2q-t5-docvectors"
}


class LuceneBatchSearcher(LuceneSearcher):
    def __init__(
            self,
            index_dir: str,
            add_query_to_pseudo: bool = False,
            normalize_method: str = None,
            normalize_scale: float = 1,
            normalize_shift: float = 0
    ):
        super().__init__(index_dir)
        self.normalize = None
        if normalize_method is not None:
            self.normalize = partial(NORMALIZE_DICT[normalize_method], scale=normalize_scale, shift=normalize_shift)
        self.add_query_to_pseudo = add_query_to_pseudo

    def batch_search(
            self,
            queries: List[str],
            qids: List[str],
            k: int = 10,
            threads: int = 1,
            query_generator: JQueryGenerator = None,
            fields=None
    ) -> Dict[str, List[SearchResult]]:
        if fields is None:
            fields = dict()

        if k <= 0:
            print("Warning, num_pseudo_queries less or equal zero, set pseudo query directly to be query.\n")

            batch_hits = dict()
            for contents, qid in zip(queries, qids):
                batch_hits[qid] = [SearchResult(qid, 1., contents)]
        else:
            batch_hits = super().batch_search(queries, qids, k, threads, query_generator, fields)
            for key, hits in batch_hits.items():
                batch_hits[key] = [SearchResult(hit.docid, hit.score, json.loads(hit.raw)['contents']) for hit in hits]

            if self.add_query_to_pseudo:
                for contents, qid in zip(queries, qids):
                    query_score = sum(hit.score for hit in batch_hits[qid])
                    batch_hits[qid].append(SearchResult(qid, query_score, contents))

        if self.normalize:
            batch_hits = self.normalize(batch_hits)

        return batch_hits
