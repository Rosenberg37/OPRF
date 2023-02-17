# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      cache.py
@Author:    Rosenberg
@Date:      2023/1/24 9:53 
@Documentation: 
    ...
"""
import json
import os
import subprocess
from multiprocessing import cpu_count
from typing import List, Union

from diskcache import Cache
from jsonargparse import CLI
from pyserini.search import DenseVectorAveragePrf, DenseVectorRocchioPrf, FaissSearcher
from tqdm import tqdm

from source import DEFAULT_CACHE_DIR, SearchResult
from source.utils.dense import FaissBatchSearcher, IMPACT_ENCODERS
from source.utils.sparse import LuceneBatchSearcher


class JsonlQueryIterator:
    def __init__(self, topics_path: str):
        self._lines = None
        if os.path.exists(topics_path) and topics_path.endswith('.json'):
            self.topics_path = topics_path
            self.data_file = open(topics_path, 'r')
        else:
            raise ValueError("Not a valid topics_path")

    def __iter__(self):
        for line in self.data_file:
            topic = json.loads(line)
            yield topic['id'], topic['contents']

    def __len__(self):
        out = subprocess.getoutput("wc -l %s" % self.topics_path)
        return int(out.split()[0])

    def __del__(self):
        self.data_file.close()


def cache(
        pseudo_name: str = 'msmarco_v1_passage_doc2query-t5_expansions_-1',
        pseudo_index_dir: str = None,
        num_pseudo_return_hits: int = 1000,
        encoder_name: Union[str, List[str]] = "lucene",
        prf_depth: int = 0,
        prf_method: str = 'avg',
        rocchio_alpha: float = 0.9,
        rocchio_beta: float = 0.1,
        rocchio_gamma: float = 0.1,
        rocchio_topk: int = 3,
        rocchio_bottomk: int = 0,
        prebuilt_index_name: Union[str, List[str]] = 'msmarco-v1-passage-full',
        threads: int = cpu_count(),
        batch_size: int = cpu_count(),
        device: str = "cpu",
):
    # Set up queries
    if pseudo_name is not None:
        if pseudo_index_dir is not None:
            raise ValueError("Can not specify both pseudo_name and pseudo_index")
        else:
            pseudo_index_dir = os.path.join(DEFAULT_CACHE_DIR, 'runs', 'pseudo_queries', pseudo_name, f"{pseudo_name}.json")
    elif pseudo_index_dir is None:
        raise ValueError("At least specify pseudo_name or pseudo_index")
    query_iterator = JsonlQueryIterator(pseudo_index_dir)

    # set up searcher
    if encoder_name == "lucene":
        query_encoder = None
        searcher = LuceneBatchSearcher(prebuilt_index_name)
    elif encoder_name in IMPACT_ENCODERS:
        query_encoder = None
        searcher = LuceneBatchSearcher(prebuilt_index_name, impact=True, encoder_name=encoder_name)
    else:
        query_encoder = FaissBatchSearcher.init_query_encoder(encoder_name, device)
        searcher = FaissSearcher.from_prebuilt_index(prebuilt_index_name, query_encoder)

    # Check PRF Flag
    prf_rule = None
    if prf_depth > 0 and encoder_name != "lucene" and encoder_name not in IMPACT_ENCODERS:
        prf_depth = prf_depth

        if prf_method.lower() == 'avg':
            prf_rule = DenseVectorAveragePrf()
        elif prf_method.lower() == 'rocchio':
            prf_rule = DenseVectorRocchioPrf(
                rocchio_alpha,
                rocchio_beta,
                rocchio_gamma,
                rocchio_topk,
                rocchio_bottomk
            )
        else:
            raise ValueError("Unexpected pseudo_prf_method.")

    # Set up cache
    cache = Cache(eviction_policy='none')

    print(f"Run with {threads} threads.")

    # Build cache
    batch_queries, batch_qids = list(), list()
    for index, (query_id, text) in enumerate(tqdm(query_iterator)):
        batch_queries.append(text), batch_qids.append(str(query_id))

        if (index + 1) % batch_size == 0:
            # Here meets the end of file problem, but not cause large influence for measuring time.
            if query_encoder is not None:
                q_embs = query_encoder.batch_encode(batch_queries)
            else:
                q_embs = batch_queries

            if prf_rule is not None and query_encoder is not None:
                q_embs, prf_candidates = searcher.batch_search(q_embs, batch_qids, k=prf_depth, return_vector=True, threads=threads)
                prf_embs_q = prf_rule.get_batch_prf_q_emb(batch_qids, q_embs, prf_candidates)
                search_hits = searcher.batch_search(prf_embs_q, batch_qids, k=num_pseudo_return_hits, threads=threads)
            else:
                search_hits = searcher.batch_search(q_embs, batch_qids, k=num_pseudo_return_hits, threads=threads)

            for id, hits in search_hits.items():
                cache.set(id, [SearchResult(hit.docid, hit.score, None) for hit in hits])

            batch_queries.clear(), batch_qids.clear()


if __name__ == '__main__':
    CLI(cache)
