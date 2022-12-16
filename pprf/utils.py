# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      utils.py
@Author:    Rosenberg
@Date:      2022/12/3 19:02 
@Documentation: 
    ...
"""
import hashlib
import json
import os
from typing import List, Union

import torch
from datasets import load_dataset
from pyserini.index import IndexReader
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer

from pprf import CACHE_DIR


def get_raw_contents(docid: int, index_reader: IndexReader):
    return index_reader.doc_raw(str(docid)).split("\"")[-2]


def generate_pseudo_queries(
        tokenizer: PreTrainedTokenizer,
        model: torch.nn.modules,
        doc_text: Union[str, List[str]],
        max_length: int = 64,
        num_return_sequences: int = 3
) -> Union[List[List[str]], List[str]]:
    device = next(model.parameters()).device
    inputs = tokenizer(doc_text, padding=True, return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        do_sample=True,
        top_k=10,
        num_return_sequences=num_return_sequences
    )

    queries = [tokenizer.decode(outputs[i], skip_special_tokens=True) for i in range(num_return_sequences)]

    if type(doc_text) is str:
        pass
    elif type(doc_text) is list:
        queries = [queries[i:i + num_return_sequences] for i in range(len(doc_text), num_return_sequences)]
    else:
        raise TypeError("Unexpected doc_texts")

    return queries


def convert_dataset_to_jsonl(
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
