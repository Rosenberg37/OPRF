# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      aggregate.py
@Author:    Rosenberg
@Date:      2022/12/19 18:31 
@Documentation: 
    ...
"""
from typing import List, Tuple

from math import exp


def softmax_sum(pseudo_doc_hits: List[Tuple], length_correct: bool):
    numerator, denominator = 0, 0
    for s in pseudo_doc_hits:
        exp_pseudo_score = exp(s[1])
        numerator += s[0] * exp_pseudo_score
        denominator += exp_pseudo_score
    result = numerator / denominator
    if length_correct:
        result *= len(pseudo_doc_hits)
    return result


def sum_doc(pseudo_doc_hits: List[Tuple]):
    return sum(s[0] for s in pseudo_doc_hits)


def sum_pseudo(pseudo_doc_hits: List[Tuple]):
    return sum(s[1] for s in pseudo_doc_hits)


def sum_total(pseudo_doc_hits: List[Tuple]):
    return sum(s[0] + s[1] for s in pseudo_doc_hits)


def max_doc(pseudo_doc_hits: List[Tuple]):
    return max(s[0] for s in pseudo_doc_hits)


def max_pseudo(pseudo_doc_hits: List[Tuple]):
    return max(s[1] for s in pseudo_doc_hits)


def max_total(pseudo_doc_hits: List[Tuple]):
    return max(sum(s) for s in pseudo_doc_hits)
