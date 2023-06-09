# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      __init__.py.py
@Author:    Rosenberg
@Date:      2022/12/3 14:14 
@Documentation: 
    ...
"""
import os
from typing import Dict, List

from source.utils import SearchResult

DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'pprf')
BatchSearchResult = Dict[str, List[SearchResult]]
