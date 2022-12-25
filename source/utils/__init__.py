# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      __init__.py.py
@Author:    Rosenberg
@Date:      2022/12/19 18:31 
@Documentation: 
    ...
"""
from dataclasses import dataclass
from typing import Optional

from math import atan, exp, log, pi


@dataclass
class SearchResult:
    docid: str
    score: float
    contents: Optional[str]


def normalize_results(results: dict, method: str = "min_max", shift: int = 1):
    if method == "softmax":
        for hits in results.values():
            normalize = 0
            max_score = max(hit.score for hit in hits)
            for hit in hits:
                score = hit.score - max_score
                hit.score = exp(score)
                normalize += score
            for hit in hits:
                hit.score = shift + hit.score / normalize
    elif method == "min_max":
        for hits in results.values():
            max_score = max(hit.score for hit in hits)
            min_score = min(hit.score for hit in hits)
            normalize = max_score - min_score
            for hit in hits:
                hit.score = shift + (hit.score - min_score) / normalize
    elif method == "log_max":
        for hits in results.values():
            max_log = log(max(hit.score for hit in hits))
            for hit in hits:
                hit.score = shift + log(hit.score) / max_log
    elif method == "arctan":
        for hits in results.values():
            for hit in hits:
                hit.score = shift + atan(hit.score) * (2 / pi)
    else:
        raise ValueError("Unexpected method.")

    return results
