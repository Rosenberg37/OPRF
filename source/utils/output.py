# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      output.py
@Author:    Rosenberg
@Date:      2023/1/10 11:09 
@Documentation: 
    ...
"""
import json
import os
from abc import abstractmethod
from collections import OrderedDict
from typing import List

from pyserini.search import JLuceneSearcherResult


class OutputWriter:
    def __init__(
            self,
            file_path: str,
            log_path: str = None,
            max_hits: int = 1000,
            tag: str = None,
            topics: dict = None,
            use_max_passage: bool = False,
            max_passage_delimiter: str = None,
            max_passage_hits: int = 100
    ):
        self.file_path = file_path
        self.log_path = log_path

        self.tag = tag
        self.topics = topics
        self.use_max_passage = use_max_passage
        self.max_passage_delimiter = max_passage_delimiter if use_max_passage else None
        self.max_hits = max_passage_hits if use_max_passage else max_hits
        self._file = None

    def __enter__(self):
        file_path = os.path.dirname(self.file_path)
        if file_path:
            os.makedirs(file_path, exist_ok=True)
        self._file = open(self.file_path, "w")

        if self.log_path:
            log_path = os.path.dirname(self.log_path)
            os.makedirs(log_path, exist_ok=True)
            self._log_file = open(self.log_path, "w")
        else:
            self._log_file = None

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._file.close()
        if self._log_file:
            self._log_file.close()

    def hits_iterator(self, hits: List[JLuceneSearcherResult]):
        unique_docs = set()
        rank = 1
        for hit in hits:
            if self.use_max_passage and self.max_passage_delimiter:
                docid = hit.docid.split(self.max_passage_delimiter)[0]
            else:
                docid = hit.docid.strip()

            if self.use_max_passage:
                if docid in unique_docs:
                    continue
                unique_docs.add(docid)

            yield docid, rank, hit.score, hit

            rank = rank + 1
            if rank > self.max_hits:
                break

    @abstractmethod
    def write(self, query_hits: dict, pseudo_hits: dict = None, queries_ids: dict = None):
        query_hits = OrderedDict(query_hits)
        pseudo_hits = OrderedDict(pseudo_hits)

        for topic_id, hits in query_hits.items():
            for docid, rank, score, _ in self.hits_iterator(hits):
                self._file.write(f'{topic_id} Q0 {docid} {rank} {score:.6f} {self.tag}\n')

        if self._log_file is not None:
            for topic_id, hits in pseudo_hits.items():
                query = queries_ids[topic_id] if queries_ids else topic_id
                dump_dict = {
                    query: [{
                        "id": hit.docid,
                        "score": hit.score,
                        "contents": hit.contents
                    } for hit in hits]
                }
                self._log_file.write(json.dumps(dump_dict, indent=4, sort_keys=True) + '\n')
