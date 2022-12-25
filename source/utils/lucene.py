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

from pyserini.search import LuceneSearcher

from source.utils import normalize_results, SearchResult


class LuceneBatchSearcher(LuceneSearcher):
    def __init__(self, index_dir: str, normalize_score: bool = False):
        super().__init__(index_dir)
        self.normalize_score = normalize_score

    def batch_search(self, *args, **kargs) -> Dict[str, List[SearchResult]]:
        results = super().batch_search(*args, **kargs)
        for key, hits in results.items():
            results[key] = [SearchResult(hit.docid, hit.score, json.loads(hit.raw)['contents']) for hit in hits]
        if self.normalize_score:
            results = normalize_results(results)
        return results
