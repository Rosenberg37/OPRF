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


@dataclass
class SearchResult:
    docid: str
    score: float
    contents: Optional[str]
