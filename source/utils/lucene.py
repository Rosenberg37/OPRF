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
import os
from typing import List

from pyserini.search import LuceneImpactSearcher, LuceneSearcher

from source import BatchSearchResult
from source.utils import SearchResult


def set_bm25_parameters(searcher, index=None, k1=2.56, b=0.59):
    if index is not None:
        # Automatically set bm25 parameters based on known index...
        if index == 'msmarco-passage' or index == 'msmarco-passage-slim' or index == 'msmarco-v1-passage' or \
                index == 'msmarco-v1-passage-slim' or index == 'msmarco-v1-passage-full':
            # See https://github.com/castorini/anserini/blob/master/docs/regressions-msmarco-passage.md
            print('MS MARCO passage: setting k1=0.82, b=0.68')
            searcher.set_bm25(0.82, 0.68)
        elif index == 'msmarco-passage-expanded' or \
                index == 'msmarco-v1-passage-d2q-t5' or \
                index == 'msmarco-v1-passage-d2q-t5-docvectors':
            # See https://github.com/castorini/anserini/blob/master/docs/regressions-msmarco-passage-docTTTTTquery.md
            print('MS MARCO passage w/ doc2query-T5 expansion: setting k1=2.18, b=0.86')
            searcher.set_bm25(2.18, 0.86)
        elif index == 'msmarco-doc' or index == 'msmarco-doc-slim' or index == 'msmarco-v1-doc' or \
                index == 'msmarco-v1-doc-slim' or index == 'msmarco-v1-doc-full':
            # See https://github.com/castorini/anserini/blob/master/docs/regressions-msmarco-doc.md
            print('MS MARCO doc: setting k1=4.46, b=0.82')
            searcher.set_bm25(4.46, 0.82)
        elif index == 'msmarco-doc-per-passage' or index == 'msmarco-doc-per-passage-slim' or \
                index == 'msmarco-v1-doc-segmented' or index == 'msmarco-v1-doc-segmented-slim' or \
                index == 'msmarco-v1-doc-segmented-full':
            # See https://github.com/castorini/anserini/blob/master/docs/regressions-msmarco-doc-segmented.md
            print('MS MARCO doc, per passage: setting k1=2.16, b=0.61')
            searcher.set_bm25(2.16, 0.61)
        elif index == 'msmarco-doc-expanded-per-doc' or \
                index == 'msmarco-v1-doc-d2q-t5' or \
                index == 'msmarco-v1-doc-d2q-t5-docvectors':
            # See https://github.com/castorini/anserini/blob/master/docs/regressions-msmarco-doc-docTTTTTquery.md
            print('MS MARCO doc w/ doc2query-T5 (per doc) expansion: setting k1=4.68, b=0.87')
            searcher.set_bm25(4.68, 0.87)
        elif index == 'msmarco-doc-expanded-per-passage' or \
                index == 'msmarco-v1-doc-segmented-d2q-t5' or \
                index == 'msmarco-v1-doc-segmented-d2q-t5-docvectors':
            # See https://github.com/castorini/anserini/blob/master/docs/regressions-msmarco-doc-segmented-docTTTTTquery.md
            print('MS MARCO doc w/ doc2query-T5 (per passage) expansion: setting k1=2.56, b=0.59')
            searcher.set_bm25(2.56, 0.59)

    print(f'Setting BM25 parameters: k1={k1}, b={b}')
    searcher.set_bm25(k1, b)


class LuceneBatchSearcher:
    def __init__(
            self,
            index_dir: str,
            rm3: bool = False,
            rocchio: bool = False,
            rocchio_use_negative: bool = False,
            impact: bool = False,
            encoder_name: str = None,
    ):
        if impact:
            if os.path.exists(index_dir):
                self.searcher = LuceneImpactSearcher(index_dir, encoder_name)
            else:
                self.searcher = LuceneImpactSearcher.from_prebuilt_index(index_dir, encoder_name)
        else:
            if os.path.exists(index_dir):
                # create searcher from index directory
                self.searcher = LuceneSearcher(index_dir)
            else:
                # create searcher from prebuilt index name
                self.searcher = LuceneSearcher.from_prebuilt_index(index_dir)

            set_bm25_parameters(self.searcher, index_dir)

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
    ) -> BatchSearchResult:
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
                    query_score = sum(hit.score for hit in batch_hits[qid])
                    batch_hits[qid].append(SearchResult(qid, query_score, contents))

        return batch_hits
