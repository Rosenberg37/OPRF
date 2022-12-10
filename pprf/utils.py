# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      utils.py
@Author:    Rosenberg
@Date:      2022/12/3 19:02 
@Documentation: 
    ...
"""
from typing import List, Union

import torch
from pyserini.index import IndexReader
from transformers.tokenization_utils import PreTrainedTokenizer


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
