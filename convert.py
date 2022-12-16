# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      generate_pseudo_index.py
@Author:    Rosenberg
@Date:      2022/12/9 8:43 
@Documentation: 
    ...
"""

from jsonargparse import CLI

from pprf.utils import convert_dataset_to_jsonl

if __name__ == '__main__':
    CLI(convert_dataset_to_jsonl)
