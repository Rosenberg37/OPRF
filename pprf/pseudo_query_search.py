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
from typing import List

from diskcache import Cache
from math import exp
from pyserini.search import AnceQueryEncoder, AutoQueryEncoder, BprQueryEncoder, DkrrDprQueryEncoder, DprQueryEncoder, FaissSearcher, QueryEncoder, TctColBertQueryEncoder
from pyserini.search.lucene import LuceneSearcher

device: str = "cuda"


class PseudoQuerySearcher:
    DEFAULT_BUFFER_DIR = os.path.join(os.path.expanduser('~'), ".cache", "pprf")

    @staticmethod
    def init_query_encoder(
            encoder: str,
            tokenizer_name: str = None,
            device: str = "cuda:0",
    ):
        encoded_queries_map = {
            'msmarco-passage-dev-subset': 'tct_colbert-msmarco-passage-dev-subset',
            'dpr-nq-dev': 'dpr_multi-nq-dev',
            'dpr-nq-test': 'dpr_multi-nq-test',
            'dpr-trivia-dev': 'dpr_multi-trivia-dev',
            'dpr-trivia-test': 'dpr_multi-trivia-test',
            'dpr-wq-test': 'dpr_multi-wq-test',
            'dpr-squad-test': 'dpr_multi-squad-test',
            'dpr-curated-test': 'dpr_multi-curated-test'
        }
        encoder_class_map = {
            "dkrr": DkrrDprQueryEncoder,
            "dpr": DprQueryEncoder,
            "bpr": BprQueryEncoder,
            "tct_colbert": TctColBertQueryEncoder,
            "ance": AnceQueryEncoder,
            "sentence": AutoQueryEncoder,
            "auto": AutoQueryEncoder,
        }

        if encoder in encoded_queries_map:
            if os.path.exists(encoder):
                if 'bpr' in encoder:
                    return BprQueryEncoder(encoded_query_dir=encoder)
                else:
                    return QueryEncoder(encoder)
            else:
                if 'bpr' in encoder:
                    return BprQueryEncoder.load_encoded_queries(encoder)
                else:
                    return QueryEncoder.load_encoded_queries(encoder)

        if encoder:
            encoder_class = None

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
            kwargs = dict(encoder_dir=encoder, tokenizer_name=tokenizer_name, device=device, prefix=None)
            if (encoder_class == "sentence") or ("sentence" in encoder):
                kwargs.update(dict(pooling='mean', l2_norm=True))

            return encoder_class(**kwargs)

        raise ValueError(f'No encoded queries.')

    def __init__(
            self,
            pseudo_index_dir: str,
            doc_index: str,
            doc_searcher: str = "sparce",
            buffer_dir: str = None,
            device: str = "cuda:0"
    ):
        self.searcher_pseudo = LuceneSearcher(pseudo_index_dir)

        if doc_searcher.lower() == "sparse":
            self.searcher_doc: LuceneSearcher = LuceneSearcher.from_prebuilt_index(doc_index)
        else:
            query_encoder = self.init_query_encoder(doc_searcher, device=device)
            self.searcher_doc: FaissSearcher = FaissSearcher.from_prebuilt_index(doc_index, query_encoder)

        self.buffer_dir = self.DEFAULT_BUFFER_DIR if buffer_dir is None else buffer_dir
        pseudo_name = os.path.split(pseudo_index_dir)[-1]
        index_name = os.path.split(doc_index)[-1]
        self.buffer_dir = os.path.join(self.buffer_dir, pseudo_name, index_name)
        if not os.path.exists(self.buffer_dir):
            os.makedirs(self.buffer_dir)

        self.buffer_dir = os.path.join(self.buffer_dir, "bm25")
        self.cache = Cache(self.buffer_dir)

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
        use_buffer : bool
            Whether buffered results used.

        Returns
        -------

        """
        batch_pseudo_hits = self.searcher_pseudo.batch_search(batch_queries, batch_qids, num_pseudo_queries, threads)

        results = list()
        for id_ in batch_qids:
            pseudo_hits = batch_pseudo_hits[id_]
            query2pseudo_scores = {hit.docid: hit.score for hit in pseudo_hits}

            pseudo_ids = list()
            pseudo_texts = list()
            pseudo_results = dict()
            for hit in pseudo_hits:
                result = self.cache.get(hit.docid, None)
                if result is not None:
                    if len(result) < num_return_hits:
                        text = hit.raw.split("\"")[-2]
                        hits = self.searcher_doc.search(text, k=num_return_hits)
                        hits = [{"score": hit.score, **json.loads(hit.raw)} for hit in hits]
                        pseudo_results[hit.docid] = hits
                        self.cache.set(hit.docid, hits)
                    else:
                        pseudo_results[hit.docid] = result[:num_return_hits]
                else:
                    pseudo_ids.append(hit.docid)
                    pseudo_texts.append(hit.raw.split("\"")[-2])

            search_results = self.searcher_doc.batch_search(pseudo_texts, pseudo_ids, num_return_hits, threads)
            for pseudo_id, pseudo_hits in search_results.items():
                pseudo_hits = [{"score": hit.score, **json.loads(hit.raw)} for hit in pseudo_hits]
                pseudo_results[pseudo_id] = pseudo_hits
                self.cache.set(pseudo_id, pseudo_hits)

            doc_hits = dict()
            for pseudo_id, pseudo_hits in pseudo_results.items():
                for hit in pseudo_hits:
                    value = doc_hits.get(hit['id'], [])
                    value += [(hit['score'], query2pseudo_scores[pseudo_id])]
                    doc_hits[hit['id']] = value
            for pseudo_id, pseudo_hits in doc_hits.items():
                doc_hits[pseudo_id] = sum(s[0] * exp(s[1]) for s in pseudo_hits) / sum(exp(s[1]) for s in pseudo_hits)
            doc_hits = sorted([(v, k) for k, v in doc_hits.items()], reverse=True)[:num_return_hits]
            results.append((id_, doc_hits))

        return results
