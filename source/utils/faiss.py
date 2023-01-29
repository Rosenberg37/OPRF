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
from typing import Dict, List, Literal, Optional, Union

import faiss
import numpy as np
import torch
from diskcache import Cache
from pyserini.dsearch import BinaryDenseSearcher
from pyserini.encode import AnceQueryEncoder, PcaEncoder
from pyserini.output_writer import get_output_writer, OutputFormat
from pyserini.query_iterator import get_query_iterator, TopicsFormat
from pyserini.search import AnceQueryEncoder, AutoQueryEncoder, DenseVectorAncePrf, DenseVectorAveragePrf, DenseVectorRocchioPrf, FaissSearcher, \
    LuceneSearcher, QueryEncoder
from pyserini.search.faiss.__main__ import init_query_encoder
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast

from source import DEFAULT_CACHE_DIR
from source.eval import EVAL_NAME_MAPPING, evaluate
from source.utils import SearchResult
from source.utils.lucene import LuceneBatchSearcher

FAISS_BASELINES = {
    'DistilBERT KD TASB': {
        'threads': 16,
        'batch_size': 512,
        'index': 'msmarco-passage-distilbert-dot-tas_b-b256-bf',
        'encoder': 'sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco',
        'output': os.path.join(DEFAULT_CACHE_DIR, 'runs', 'run.msmarco-v1-passage.distilbert-kd-tasb-otf.dl19.txt'),
    },
    'TCT_ColBERTv2': {
        'threads': 16,
        'batch_size': 512,
        'index': 'msmarco-passage-tct_colbert-v2-hnp-bf',
        'encoder': 'castorini/tct_colbert-v2-hnp-msmarco',
        'output': os.path.join(DEFAULT_CACHE_DIR, 'runs', 'run.msmarco-v1-passage.tct_colbert-v2-hnp-otf.dl19.txt'),
    },
    'ANCE': {
        'threads': 16,
        'batch_size': 512,
        'index': 'msmarco-passage-ance-bf',
        'encoder': 'castorini/ance-msmarco-passage',
        'output': os.path.join(DEFAULT_CACHE_DIR, 'runs', 'run.msmarco-v1-passage.ance-otf.dl19.txt'),
    }
}

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

    @staticmethod
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

    def init_searcher(self):
        if self.encoder_name == "lucene":
            self.searcher = LuceneBatchSearcher(self.prebuilt_index_name)
        elif self.encoder_name in IMPACT_ENCODERS:
            self.searcher = LuceneBatchSearcher(self.prebuilt_index_name, impact=True, encoder_name=self.encoder_name)
        else:
            self.query_encoder = self.init_query_encoder(self.encoder_name, self.device)
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


def faiss_search(
        searcher,
        batch_topics,
        batch_topic_ids,
        PRF_FLAG: bool,
        prf_depth: int,
        prf_method: str,
        prfRule,
        hits: int,
        kwargs: dict,
        threads: int,
):
    if PRF_FLAG:
        q_embs, prf_candidates = searcher.batch_search(batch_topics, batch_topic_ids, k=prf_depth, return_vector=True, **kwargs)
        # ANCE-PRF input is different, do not need query embeddings
        if prf_method.lower() == 'ance-prf':
            prf_embs_q = prfRule.get_batch_prf_q_emb(batch_topics, batch_topic_ids, prf_candidates)
        else:
            prf_embs_q = prfRule.get_batch_prf_q_emb(batch_topic_ids, q_embs, prf_candidates)
        return searcher.batch_search(prf_embs_q, batch_topic_ids, k=hits, threads=threads, **kwargs)
    else:
        return searcher.batch_search(batch_topics, batch_topic_ids, hits, threads=threads, **kwargs)


def faiss_main(
        index: str,
        topic_name: str,
        output: str,
        hits: int = 1000,
        binary_hits: int = 1000,
        rerank: bool = False,
        topics_format: str = TopicsFormat.DEFAULT.value,
        output_format: str = OutputFormat.TREC.value,
        max_passage: bool = False,
        max_passage_hits: int = 100,
        max_passage_delimiter: str = '#',
        batch_size: int = 1,
        threads: int = 1,
        encoder_class: Optional[Literal["dkrr", "dpr", "bpr", "tct_colbert", "ance", "sentence", "auto"]] = None,
        encoder: str = None,
        tokenizer: str = None,
        encoded_queries: str = None,
        pca_model: str = None,
        device: str = 'cpu',
        query_prefix: str = None,
        searcher: str = 'simple',
        prf_depth: int = 0,
        prf_method: str = 'avg',
        rocchio_alpha: float = 0.9,
        rocchio_beta: float = 0.1,
        rocchio_gamma: float = 0.1,
        rocchio_topk: int = 3,
        rocchio_bottomk: int = 0,
        sparse_index: str = None,
        ance_prf_encoder: str = None,
):
    query_iterator = get_query_iterator(topic_name, TopicsFormat(topics_format))
    topics = query_iterator.topics

    query_encoder = init_query_encoder(
        encoder, encoder_class, tokenizer, topics, encoded_queries, device, query_prefix)
    if pca_model:
        query_encoder = PcaEncoder(query_encoder, pca_model)
    kwargs = {}
    if os.path.exists(index):
        # create searcher from index directory
        if searcher.lower() == 'bpr':
            kwargs = dict(binary_k=binary_hits, rerank=rerank)
            searcher = BinaryDenseSearcher(index, query_encoder)
        else:
            searcher = FaissSearcher(index, query_encoder)
    else:
        # create searcher from prebuilt index name
        if searcher.lower() == 'bpr':
            kwargs = dict(binary_k=binary_hits, rerank=rerank)
            searcher = BinaryDenseSearcher.from_prebuilt_index(index, query_encoder)
        else:
            searcher = FaissSearcher.from_prebuilt_index(index, query_encoder)

    if not searcher:
        exit()

    # Check PRF Flag
    prfRule = None
    if prf_depth > 0 and type(searcher) == FaissSearcher:
        PRF_FLAG = True
        if prf_method.lower() == 'avg':
            prfRule = DenseVectorAveragePrf()
        elif prf_method.lower() == 'rocchio':
            prfRule = DenseVectorRocchioPrf(rocchio_alpha, rocchio_beta, rocchio_gamma,
                                            rocchio_topk, rocchio_bottomk)
        # ANCE-PRF is using a new query encoder, so the input to DenseVectorAncePrf is different
        elif prf_method.lower() == 'ance-prf' and type(query_encoder) == AnceQueryEncoder:
            if os.path.exists(sparse_index):
                sparse_searcher = LuceneSearcher(sparse_index)
            else:
                sparse_searcher = LuceneSearcher.from_prebuilt_index(sparse_index)
            prf_query_encoder = AnceQueryEncoder(encoder_dir=ance_prf_encoder, tokenizer_name=tokenizer,
                                                 device=device)
            prfRule = DenseVectorAncePrf(prf_query_encoder, sparse_searcher)
        print(f'Running FaissSearcher with {prf_method.upper()} PRF...')
    else:
        PRF_FLAG = False

    # build output path
    output_path = output

    tag = 'Faiss'
    output_writer = get_output_writer(output_path, OutputFormat(output_format), 'w',
                                      max_hits=hits, tag=tag, topics=topics,
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
                results = faiss_search(
                    searcher=searcher,
                    batch_topics=batch_topics,
                    batch_topic_ids=batch_topic_ids,
                    PRF_FLAG=PRF_FLAG,
                    prf_depth=prf_depth,
                    prf_method=prf_method,
                    prfRule=prfRule,
                    hits=hits,
                    kwargs=kwargs,
                    threads=threads
                )
                results = [(id_, results[id_]) for id_ in batch_topic_ids]
                batch_topic_ids.clear()
                batch_topics.clear()
            else:
                continue

            for topic, hits in results:
                output_writer.write(topic, hits)

            results.clear()

    metrics = evaluate(
        topic_name=EVAL_NAME_MAPPING[topic_name],
        path_to_candidate=output_path,
        print_result=False,
    )
    return results, metrics
