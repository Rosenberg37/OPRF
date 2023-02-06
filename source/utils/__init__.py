# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      __init__.py.py
@Author:    Rosenberg
@Date:      2022/12/19 18:31 
@Documentation: 
    ...
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchResult:
    docid: str
    score: float
    contents: Optional[str]


QUERY_NAME_MAPPING = {
    "msmarco-passage-dev-subset": "msmarco-passage-dev-subset",
    "dev-passage": "msmarco-passage-dev-subset",
    "dl19-passage": "dl19-passage",
    "dl20-passage": "dl20",
    "dlhard-passage": os.path.join(os.path.expanduser('~'), '.cache', 'pyserini', 'topics-and-qrels', 'topics.dl_hard.tsv'),
    "msmarco-doc-dev": "msmarco-doc-dev",
    "dev-doc": "msmarco-doc-dev",
    "dl19-doc": "dl19-doc",
    "dl20-doc": "dl20",
    "dlhard-doc": os.path.join(os.path.expanduser('~'), '.cache', 'pyserini', 'topics-and-qrels', 'topics.dl_hard.tsv'),
}
