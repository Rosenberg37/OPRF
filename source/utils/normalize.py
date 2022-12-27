# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      normalize.py
@Author:    Rosenberg
@Date:      2022/12/26 12:07 
@Documentation: 
    ...
"""
from math import atan, exp, log, pi


def softmax(results: dict, scale: float = 1, shift: int = 0):
    for hits in results.values():
        normalize = 0
        max_score = max(hit.score for hit in hits)
        for hit in hits:
            score = hit.score - max_score
            hit.score = exp(score)
            normalize += score
        for hit in hits:
            hit.score = shift + hit.score / normalize * scale
    return results


def min_max(results: dict, scale: float = 1, shift: int = 0):
    for hits in results.values():
        max_score = max(hit.score for hit in hits)
        min_score = min(hit.score for hit in hits)
        normalize = max_score - min_score
        for hit in hits:
            hit.score = shift + (hit.score - min_score) / normalize * scale
    return results


def min_max_mid(results: dict, scale: float = 1, shift: int = 0):
    for hits in results.values():
        max_score = max(hit.score for hit in hits)
        min_score = min(hit.score for hit in hits)
        mid_score = (max_score + min_score) / 2
        normalize = max_score - min_score
        for hit in hits:
            hit.score = shift + (hit.score - mid_score) / normalize * scale
    return results


def log_max(results: dict, scale: float = 1, shift: int = 0):
    for hits in results.values():
        max_log = log(max(hit.score for hit in hits))
        for hit in hits:
            hit.score = shift + log(hit.score) / max_log * scale
    return results


def arctan(results: dict, scale: float = 1, shift: int = 0):
    for hits in results.values():
        for hit in hits:
            hit.score = shift + atan(hit.score) * (2 / pi) * scale
    return results


def rank(results: dict, scale: float = 1, shift: int = 0):
    for hits in results.values():
        hits = sorted(hits, key=lambda hit: hit.score)
        for i, hit in enumerate(hits):
            hit.score = shift + (i + 1) / len(hits) * scale
    return results


NORMALIZE_DICT = {
    "softmax": softmax,
    "min_max": min_max,
    "min_max_mid": min_max_mid,
    "log_max": log_max,
    "arctan": arctan,
    "rank": rank,
}
