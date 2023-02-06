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

from jsonargparse import CLI
from pyserini.analysis import JDefaultEnglishAnalyzer, JWhiteSpaceAnalyzer
from pyserini.output_writer import get_output_writer, OutputFormat
from pyserini.query_iterator import get_query_iterator, TopicsFormat
from pyserini.search import JDisjunctionMaxQueryGenerator, LuceneImpactSearcher, LuceneSearcher
from pyserini.search.lucene.__main__ import set_bm25_parameters
from pyserini.search.lucene.reranker import ClassifierType, PseudoRelevanceClassifierReranker
from tqdm import tqdm
from transformers import AutoTokenizer

from source import BatchSearchResult
from source.eval import evaluate
from source.utils import QUERY_NAME_MAPPING, SearchResult


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

            self.set_bm25_parameters(self.searcher, index_dir)

        if rm3:
            self.searcher.set_rm3()

        if rocchio:
            if rocchio_use_negative:
                self.searcher.set_rocchio(gamma=0.15, use_negative=True)
            else:
                self.searcher.set_rocchio()

    @staticmethod
    def set_bm25_parameters(searcher, index=None, k1=2.56, b=0.59):
        if index is not None:
            # Automatically set bm25 parameters based on known index...
            if index == 'msmarco-passage' or index == 'msmarco-passage-slim' or index == 'msmarco-v1-passage' or \
                    index == 'msmarco-v1-passage-slim' or index == 'msmarco-v1-passage-full':
                # See https://github.com/castorini/anserini/blob/master/docs/regressions-msmarco-passage.md
                searcher.set_bm25(0.82, 0.68)
            elif index == 'msmarco-passage-expanded' or \
                    index == 'msmarco-v1-passage-d2q-t5' or \
                    index == 'msmarco-v1-passage-d2q-t5-docvectors':
                # See https://github.com/castorini/anserini/blob/master/docs/regressions-msmarco-passage-docTTTTTquery.md
                searcher.set_bm25(2.18, 0.86)
            elif index == 'msmarco-doc' or index == 'msmarco-doc-slim' or index == 'msmarco-v1-doc' or \
                    index == 'msmarco-v1-doc-slim' or index == 'msmarco-v1-doc-full':
                # See https://github.com/castorini/anserini/blob/master/docs/regressions-msmarco-doc.md
                searcher.set_bm25(4.46, 0.82)
            elif index == 'msmarco-doc-per-passage' or index == 'msmarco-doc-per-passage-slim' or \
                    index == 'msmarco-v1-doc-segmented' or index == 'msmarco-v1-doc-segmented-slim' or \
                    index == 'msmarco-v1-doc-segmented-full':
                # See https://github.com/castorini/anserini/blob/master/docs/regressions-msmarco-doc-segmented.md
                searcher.set_bm25(2.16, 0.61)
            elif index == 'msmarco-doc-expanded-per-doc' or \
                    index == 'msmarco-v1-doc-d2q-t5' or \
                    index == 'msmarco-v1-doc-d2q-t5-docvectors':
                # See https://github.com/castorini/anserini/blob/master/docs/regressions-msmarco-doc-docTTTTTquery.md
                searcher.set_bm25(4.68, 0.87)
            elif index == 'msmarco-doc-expanded-per-passage' or \
                    index == 'msmarco-v1-doc-segmented-d2q-t5' or \
                    index == 'msmarco-v1-doc-segmented-d2q-t5-docvectors':
                # See https://github.com/castorini/anserini/blob/master/docs/regressions-msmarco-doc-segmented-docTTTTTquery.md
                searcher.set_bm25(2.56, 0.59)

        searcher.set_bm25(k1, b)

    def batch_search(
            self,
            queries: List[str],
            qids: List[str],
            k: int = 10,
            threads: int = 1,
            add_query_to_pseudo: bool = False,
    ) -> BatchSearchResult:
        if k <= 0:
            print("Warning, num_pseudo_queries less or equal zero, set pseudo query directly to be query.")

            batch_hits = dict()
            for contents, qid in zip(queries, qids):
                batch_hits[qid] = [SearchResult(qid, 1., contents)]
        else:
            threads = min(len(queries), threads)
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


def lucene_search(
        impact: bool,
        searcher,
        batch_topics,
        batch_topic_ids,
        hits,
        threads,
        fields,
        query_generator,
):
    if impact:
        return searcher.batch_search(
            batch_topics, batch_topic_ids, hits, threads, fields=fields
        )
    else:
        return searcher.batch_search(
            batch_topics, batch_topic_ids, hits, threads,
            query_generator=query_generator, fields=fields
        )


def lucene_main(
        index: str,
        topic_name: str,
        output: str,
        encoder: str = None,
        impact: bool = False,
        min_idf: int = 0,
        bm25=True,
        k1: float = None,
        b: float = None,
        rm3: bool = False,
        rocchio: bool = False,
        rocchio_use_negative: bool = False,
        qld: bool = False,
        prcl: str = None,
        vectorizer: str = None,
        r: int = 10,
        n: int = 100,
        alpha: float = 0.5,
        language: str = 'en',
        dismax: bool = False,
        tiebreaker: float = 0.0,
        stopwords: str = None,
        num_return_hits: int = 1000,
        topics_format: str = TopicsFormat.DEFAULT.value,
        output_format: str = OutputFormat.TREC.value,
        max_passage: bool = False,
        max_passage_hits: int = 100,
        max_passage_delimiter: str = '#',
        batch_size: int = 1,
        threads: int = 1,
        tokenizer: str = None,
        searcher: str = 'simple',
        remove_duplicates: bool = False,
        remove_query: bool = False,
        print_result: bool = True,
):
    query_iterator = get_query_iterator(QUERY_NAME_MAPPING[topic_name], TopicsFormat(topics_format))
    topics = query_iterator.topics

    if not impact:
        if os.path.exists(index):
            # create searcher from index directory
            searcher = LuceneSearcher(index)
        else:
            # create searcher from prebuilt index name
            searcher = LuceneSearcher.from_prebuilt_index(index)
    elif impact:
        if os.path.exists(index):
            searcher = LuceneImpactSearcher(index, encoder, min_idf)
        else:
            searcher = LuceneImpactSearcher.from_prebuilt_index(index, encoder, min_idf)

    if language != 'en':
        searcher.set_language(language)

    if not searcher:
        exit()

    search_rankers = []

    if qld:
        search_rankers.append('qld')
        searcher.set_qld()
    elif bm25:
        search_rankers.append('bm25')
        set_bm25_parameters(searcher, index, k1, b)

    if rm3:
        search_rankers.append('rm3')
        searcher.set_rm3()

    if rocchio:
        search_rankers.append('rocchio')
        if rocchio_use_negative:
            searcher.set_rocchio(gamma=0.15, use_negative=True)
        else:
            searcher.set_rocchio()

    fields = dict()
    if fields:
        fields = dict([pair.split('=') for pair in fields])

    query_generator = None
    if dismax:
        query_generator = JDisjunctionMaxQueryGenerator(tiebreaker)

    if tokenizer != None:
        analyzer = JWhiteSpaceAnalyzer()
        searcher.set_analyzer(analyzer)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    if stopwords:
        analyzer = JDefaultEnglishAnalyzer.fromArguments('porter', False, stopwords)
        searcher.set_analyzer(analyzer)

    # get re-ranker
    use_prcl = prcl and len(prcl) > 0 and alpha > 0
    if use_prcl is True:
        ranker = PseudoRelevanceClassifierReranker(
            searcher.index_dir, vectorizer, prcl, r=r, n=n, alpha=alpha)

    # build output path
    output_path = output
    if output_path is None:
        if use_prcl is True:
            clf_rankers = []
            for t in prcl:
                if t == ClassifierType.LR:
                    clf_rankers.append('lr')
                elif t == ClassifierType.SVM:
                    clf_rankers.append('svm')

            r_str = f'prcl.r_{r}'
            n_str = f'prcl.n_{n}'
            a_str = f'prcl.alpha_{alpha}'
            clf_str = 'prcl_' + '+'.join(clf_rankers)
            tokens1 = ['run', topics, '+'.join(search_rankers)]
            tokens2 = [vectorizer, clf_str, r_str, n_str, a_str]
            output_path = '.'.join(tokens1) + '-' + '-'.join(tokens2) + ".txt"
        else:
            tokens = ['run', topics, '+'.join(search_rankers), 'txt']
            output_path = '.'.join(tokens)

    tag = output_path[:-4] if output is None else 'Anserini'

    output_writer = get_output_writer(output_path, OutputFormat(output_format), 'w',
                                      max_hits=num_return_hits, tag=tag, topics=topics,
                                      use_max_passage=max_passage,
                                      max_passage_delimiter=max_passage_delimiter,
                                      max_passage_hits=max_passage_hits)

    with output_writer:
        batch_topics = list()
        batch_topic_ids = list()
        for index, (topic_id, text) in enumerate(tqdm(query_iterator, total=len(topics.keys()))):
            batch_topic_ids.append(str(topic_id))
            batch_topics.append(text)
            if (index + 1) % batch_size == 0 or index == len(topics.keys()) - 1:
                results = lucene_search(
                    impact=impact,
                    searcher=searcher,
                    batch_topics=batch_topics,
                    batch_topic_ids=batch_topic_ids,
                    hits=num_return_hits,
                    threads=threads,
                    fields=fields,
                    query_generator=query_generator
                )
                results = [(id_, results[id_]) for id_ in batch_topic_ids]
                batch_topic_ids.clear()
                batch_topics.clear()
            else:
                continue

            for topic, hits in results:
                # do rerank
                if use_prcl and len(hits) > (r + n):
                    docids = [hit.docid.strip() for hit in hits]
                    scores = [hit.score for hit in hits]
                    scores, docids = ranker.rerank(docids, scores)
                    docid_score_map = dict(zip(docids, scores))
                    for hit in hits:
                        hit.score = docid_score_map[hit.docid.strip()]

                if remove_duplicates:
                    seen_docids = set()
                    dedup_hits = []
                    for hit in hits:
                        if hit.docid.strip() in seen_docids:
                            continue
                        seen_docids.add(hit.docid.strip())
                        dedup_hits.append(hit)
                    hits = dedup_hits

                # For some test collections, a query is doc from the corpus (e.g., arguana in BEIR).
                # We want to remove the query from the results.
                if remove_query:
                    hits = [hit for hit in hits if hit.docid != topic]

                # write results
                output_writer.write(topic, hits)

            results.clear()

    metrics = evaluate(
        topic_name=topic_name,
        path_to_candidate=output_path,
        print_result=print_result,
    )
    return results, metrics


if __name__ == '__main__':
    CLI(lucene_main)
