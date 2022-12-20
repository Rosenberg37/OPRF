# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      search.py
@Author:    Rosenberg
@Date:      2022/12/10 9:17 
@Documentation: 
    ...
"""
import json
import os
import os.path
from dataclasses import dataclass
from functools import partial
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Tuple

from diskcache import Cache
from jsonargparse import CLI
from pyserini.output_writer import TrecWriter
from pyserini.query_iterator import get_query_iterator, TopicsFormat
from pyserini.search import DenseVectorAveragePrf, DenseVectorRocchioPrf, LuceneSearcher
from tqdm import tqdm

from source import DEFAULT_CACHE_DIR
from source.eval import evaluate
from source.utils import FaissBatchSearcher, init_query_encoder, max_doc, max_pseudo, max_total, softmax_sum, sum_doc, sum_pseudo, sum_total


@dataclass
class SearchResult:
    docid: str
    score: float
    contents: Optional[str]


class PseudoQuerySearcher:
    AGGREGATE_DICT = {
        "softmax_sum_with_count": partial(softmax_sum, length_correct=True),
        "softmax_sum": partial(softmax_sum, length_correct=False),
        "sum_doc": sum_doc,
        "sum_pseudo": sum_pseudo,
        "sum_total": sum_total,
        "max_doc": max_doc,
        "max_pseudo": max_pseudo,
        "max_total": max_total,
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
            self.prf_depth = pseudo_prf_depth

            if pseudo_prf_method.lower() == 'avg':
                self.prfRule = DenseVectorAveragePrf()
            elif pseudo_prf_method.lower() == 'rocchio':
                self.prfRule = DenseVectorRocchioPrf(
                    pseudo_rocchio_alpha,
                    pseudo_rocchio_beta,
                    pseudo_rocchio_gamma,
                    pseudo_rocchio_topk,
                    pseudo_rocchio_bottomk
                )
            # ANCE-PRF is using a new query encoder, so the input to DenseVectorAncePrf is different
            else:
                raise ValueError("Unexpected pseudo_prf_method.")
        else:
            self.PRF_FLAG = False

        self.aggregate = self.AGGREGATE_DICT[aggregation]

        self.cache_dir = DEFAULT_CACHE_DIR if buffer_dir is None else buffer_dir
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
            num_pseudo_return_hits: int = 1000,
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
        num_pseudo_return_hits : int
            Number of hits to return by each pseudo query.
        threads : int
            Maximum number of threads to use.
        add_query_to_pseudo:
            Whether add self to the original pseudo queries.
        return_pseudo_hits:
            whether return the pseudo queries hit in the first stage.

        Returns
        -------

        """
        # Get pseudo queries
        if num_pseudo_queries <= 0:
            print("Warning, num_pseudo_queries less or equal zero, set pseudo query directly to be query.\n")
            batch_pseudo_hits = dict()
            for contents, qid in zip(batch_queries, batch_qids):
                batch_pseudo_hits[qid] = [SearchResult(qid, 100, contents)]
        else:
            batch_pseudo_hits = self.searcher_pseudo.batch_search(batch_queries, batch_qids, num_pseudo_queries, threads)
            if add_query_to_pseudo:
                for contents, qid in zip(batch_queries, batch_qids):
                    query_score = sum(hit.score for hit in batch_pseudo_hits[qid])
                    batch_pseudo_hits[qid].append(SearchResult(qid, query_score, contents))

        # Read cache or leave for search
        pseudo_ids_texts, pseudo_results = dict(), dict()
        for pseudo_hits in batch_pseudo_hits.values():
            for hit in pseudo_hits:
                if hit.docid not in pseudo_results:
                    result = self.cache.get(hit.docid, None)
                    if result is not None and len(result) >= num_pseudo_return_hits:
                        pseudo_results[hit.docid] = result[:num_pseudo_return_hits]
                    else:
                        pseudo_ids_texts[hit.docid] = json.loads(hit.raw)['contents'] if hit.contents is None else hit.contents

        # Search for un-cased pseudo queries
        if len(pseudo_ids_texts) > 0:
            pseudo_ids, pseudo_texts = zip(*pseudo_ids_texts.items())

            if self.PRF_FLAG:
                q_embs, prf_candidates = self.searcher_doc.batch_search(pseudo_texts, pseudo_ids, k=self.prf_depth, return_vector=True)
                prf_embs_q = self.prfRule.get_batch_prf_q_emb(pseudo_ids, q_embs, prf_candidates)
                search_results = self.searcher_doc.batch_search(prf_embs_q, pseudo_ids, k=num_return_hits, threads=threads)
            else:
                search_results = self.searcher_doc.batch_search(pseudo_texts, pseudo_ids, k=num_return_hits, threads=threads)

            for pseudo_id, pseudo_doc_hits in search_results.items():
                pseudo_doc_hits = [(hit.score, hit.docid) for hit in pseudo_doc_hits]
                pseudo_results[pseudo_id] = pseudo_doc_hits
                self.cache.set(pseudo_id, pseudo_doc_hits)

        # Aggregate and generate final results
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
            doc_hits = sorted([(v, k) for k, v in doc_hits.items()], reverse=True)

            if len(doc_hits) < num_return_hits:
                print(f"Warning, query of id {query_id} has less than {num_return_hits} candidate passages.\n")
            doc_hits = doc_hits[:num_return_hits]

            doc_hits = [SearchResult(str(idx), score, None) for score, idx in doc_hits]
            final_results.append((query_id, doc_hits))

        if return_pseudo_hits:
            return final_results, batch_pseudo_hits
        else:
            return final_results


def main(
        topic_name: str = 'msmarco-passage-dev-subset',
        query_rm3: bool = False,
        query_rocchio: bool = False,
        query_rocchio_use_negative: bool = False,
        pseudo_name: str = 'msmarco_v1_passage_doc2query-t5_expansions_5',
        pseudo_index_dir: str = None,
        num_pseudo_queries: int = 2,
        add_query_to_pseudo: bool = False,
        num_pseudo_return_hits: int = 1000,
        pseudo_encoder_name: str = "lucene",
        pseudo_prf_depth: int = 0,
        pseudo_prf_method: str = 'avg',
        pseudo_rocchio_alpha: float = 0.9,
        pseudo_rocchio_beta: float = 0.1,
        pseudo_rocchio_gamma: float = 0.1,
        pseudo_rocchio_topk: int = 3,
        pseudo_rocchio_bottomk: int = 0,
        aggregation: str = "softmax_sum_with_count",
        doc_index: str = 'msmarco-v1-passage-full',
        num_return_hits: int = 1000,
        threads: int = cpu_count(),
        batch_size: int = cpu_count(),
        device: str = "cpu",
        output_path: str = os.path.join(DEFAULT_CACHE_DIR, "runs"),
        do_eval: bool = True,
):
    """

    :param topic_name: Name of topics.
    :param query_rm3: whether the rm3 algorithm used for the first stage search.
    :param query_rocchio: whether the rocchio algorithm used for the first stage search.
    :param query_rocchio_use_negative: whether the rocchio algorithm with negative used for the first stage search.
    :param pseudo_name: index name of the candidate pseudo queries
    :param pseudo_index_dir: index path to the candidate pseudo queries.
    :param num_pseudo_queries: how many pseudo query used for second stage
    :param add_query_to_pseudo: whether add query into pseudo query for search
    :param num_pseudo_return_hits: Number of hits to return by each pseudo query.
    :param pseudo_encoder_name: Path to query encoder pytorch checkpoint or hgf encoder model name
    :param pseudo_prf_depth: Specify how many passages are used for PRF, 0: Simple retrieval with no PRF, > 0: perform PRF
    :param pseudo_prf_method: Choose PRF methods, avg or rocchio
    :param pseudo_rocchio_alpha: The alpha parameter to control the contribution from the query vector
    :param pseudo_rocchio_beta: The beta parameter to control the contribution from the average vector of the positive PRF passages
    :param pseudo_rocchio_gamma: The gamma parameter to control the contribution from the average vector of the negative PRF passages
    :param pseudo_rocchio_topk: Set topk passages as positive PRF passages for rocchio
    :param pseudo_rocchio_bottomk: Set bottomk passages as negative PRF passages for rocchio, 0: do not use negatives prf passages.
    :param aggregation: the way of aggregate hits from different pseudo queries
    :param doc_index: the index of the candidate documents
    :param num_return_hits: how many hits will be returned
    :param threads: maximum threads to use during search
    :param batch_size: batch size used for the batch search.
    :param device: the device the whole search procedure will on
    :param output_path: the path where the run file will be outputted
    :param do_eval: do evaluation step after search or not
    """
    if pseudo_name is not None:
        if pseudo_index_dir is not None:
            raise ValueError("Can not specify both pseudo_name and pseudo_index")
        else:
            pseudo_index_dir = os.path.join(DEFAULT_CACHE_DIR, 'indexes', pseudo_name)
    elif pseudo_index_dir is None:
        raise ValueError("At least specify pseudo_name or pseudo_index")

    query_iterator = get_query_iterator(topic_name, TopicsFormat.DEFAULT)
    topics = query_iterator.topics
    topics_length = len(query_iterator.topics)
    searcher = PseudoQuerySearcher(
        pseudo_index_dir, doc_index,
        query_rm3=query_rm3,
        query_rocchio=query_rocchio,
        query_rocchio_use_negative=query_rocchio_use_negative,
        pseudo_encoder_name=pseudo_encoder_name,
        pseudo_prf_depth=pseudo_prf_depth,
        pseudo_prf_method=pseudo_prf_method,
        pseudo_rocchio_alpha=pseudo_rocchio_alpha,
        pseudo_rocchio_beta=pseudo_rocchio_beta,
        pseudo_rocchio_gamma=pseudo_rocchio_gamma,
        pseudo_rocchio_topk=pseudo_rocchio_topk,
        pseudo_rocchio_bottomk=pseudo_rocchio_bottomk,
        aggregation=aggregation,
        device=device
    )

    output_path = os.path.join(DEFAULT_CACHE_DIR, output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pseudo_encoder_full_name = pseudo_encoder_name.split('/')[-1]
    if pseudo_prf_depth is not None:
        pseudo_encoder_full_name += "-" + pseudo_prf_method
    run_name = f"run.{pseudo_name}.{topic_name}.{num_pseudo_queries}.{pseudo_encoder_full_name}.{aggregation}.txt"
    output_path = os.path.join(output_path, run_name)
    tag = output_path[:-4]
    output_writer = TrecWriter(output_path, 'w', max_hits=num_return_hits, tag=tag, topics=topics)

    with output_writer:
        batch_queries, batch_queries_ids = list(), list()
        for index, (query_id, text) in enumerate(tqdm(query_iterator)):
            batch_queries_ids.append(str(query_id))
            batch_queries.append(text)
            if (index + 1) % batch_size == 0 or index == topics_length - 1:
                batch_hits = searcher.batch_search(
                    batch_queries,
                    batch_queries_ids,
                    num_pseudo_queries=num_pseudo_queries,
                    add_query_to_pseudo=add_query_to_pseudo,
                    num_pseudo_return_hits=num_pseudo_return_hits,
                    num_return_hits=num_return_hits,
                    threads=threads
                )

                batch_queries_ids.clear()
                batch_queries.clear()
            else:
                continue

            for topic, hits in batch_hits:
                output_writer.write(topic, hits)

            batch_hits.clear()

    if do_eval:
        evaluate(topic_name=topic_name, path_to_candidate=output_path)


if __name__ == '__main__':
    CLI(main)
