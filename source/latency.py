# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      latency.py
@Author:    Rosenberg
@Date:      2023/1/23 19:03 
@Documentation: 
    ...
"""
import cProfile

from jsonargparse import CLI

from source.search import search

if __name__ == '__main__':
    with cProfile.Profile() as profile:
        CLI(search)

        profile.print_stats(sort="cumtime")
