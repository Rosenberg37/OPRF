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
import os
from typing import List, Union

from datasets import load_dataset
from jsonargparse import CLI
from tqdm import tqdm

from source import DEFAULT_CACHE_DIR


def convert_dataset_to_jsonl(
        dataset_name: Union[str, List[str]] = 'castorini/msmarco_v1_passage_doc2query-t5_expansions',
        queries_num: int = -1,
        output_name: str = None,
):
    if output_name is None:
        if type(dataset_name) is str:
            output_name = f"{dataset_name.split('/')[-1]}_{queries_num}"
        else:
            output_name = f"multi_{queries_num}"

    output_path = os.path.join(DEFAULT_CACHE_DIR, "runs", "pseudo_queries", output_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    seen = set()
    if type(dataset_name) is str:
        dataset_name = [dataset_name]

    output_path = os.path.join(output_path, f"{output_name}.json")
    print(f"Output to {output_path}")
    with open(output_path, "w") as f:
        for name in dataset_name:
            dataset = load_dataset(name)['train']
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
    CLI(convert_dataset_to_jsonl)
