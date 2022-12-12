# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      generate_pseudo_index.py
@Author:    Rosenberg
@Date:      2022/12/9 8:43 
@Documentation: 
    ...
"""
import hashlib
import json
import os.path

from datasets import load_dataset
from jsonargparse import CLI
from tqdm import tqdm

CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'pprf')


def main(
        dataset_name: str = 'castorini/msmarco_v1_passage_doc2query-t5_expansions',
        queries_num: int = 5,
        output_post_fix: str = '',
):
    if output_post_fix and output_post_fix[0] != '_':
        output_post_fix = '_' + output_post_fix

    output_file_name = f"{dataset_name.split('/')[-1]}{output_post_fix}_{queries_num}"
    output_path = os.path.join(CACHE_DIR, "runs", output_file_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    seen = set()
    dataset = load_dataset(dataset_name)['train']
    with open(f"{output_path}/{output_file_name}{output_post_fix}.json", "w") as f:
        for docid_queries in tqdm(dataset):
            docid = docid_queries['id']
            queries = docid_queries['predicted_queries']

            if queries_num != -1:
                queries = queries[:queries_num]

            for qid, query in enumerate(queries):
                query_hash = hashlib.md5(query.encode()).digest()
                if query_hash not in seen:
                    f.write(json.dumps({
                        "id": f"D{docid}#{qid}",
                        "contents": query
                    }) + '\n')
                    seen.add(query_hash)


if __name__ == '__main__':
    CLI(main)
