# -*- coding: UTF-8 -*-
"""
@Project:   PPRF 
@File:      dense.py
@Author:    Rosenberg
@Date:      2022/12/16 16:27 
@Documentation: 
    ...
"""
from typing import Dict, List, Tuple, Union

import faiss
import numpy as np
import torch
from pyserini.search import AnceQueryEncoder, AutoQueryEncoder, BprQueryEncoder, DenseSearchResult, DkrrDprQueryEncoder, DprQueryEncoder, FaissSearcher, PRFDenseSearchResult, TctColBertQueryEncoder


class AnceQueryBatchEncoder(AnceQueryEncoder):
    def batch_encode(self, queries: List[str]):
        return self.prf_batch_encode(queries)


class TctColBertQueryBatchEncoder(TctColBertQueryEncoder):
    def batch_encode(self, queries: List[str]):
        max_length = 36  # hardcode for now
        queries = ['[CLS] [Q] ' + query + '[MASK]' * max_length for query in queries]
        inputs = self.tokenizer(
            queries,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt'
        ).to(self.device)
        outputs = self.model(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state[:, 4:, :], dim=1)
        return embeddings.detach().cpu().numpy()


class FaissBatchSearcher(FaissSearcher):

    def batch_search(
            self,
            queries: Union[List[str], np.ndarray],
            q_ids: List[str],
            k: int = 10,
            threads: int = 1,
            return_vector: bool = False
    ) -> Union[Dict[str, List[DenseSearchResult]], Tuple[np.ndarray, Dict[str, List[PRFDenseSearchResult]]]]:
        """

        Parameters
        ----------
        queries : Union[List[str], np.ndarray]
            List of query texts or list of query embeddings
        q_ids : List[str]
            List of corresponding query ids.
        k : int
            Number of hits to return.
        threads : int
            Maximum number of threads to use.
        return_vector : bool
            Return the results with vectors

        Returns
        -------
        Union[Dict[str, List[DenseSearchResult]], Tuple[np.ndarray, Dict[str, List[PRFDenseSearchResult]]]]
            Either returns a dictionary holding the search results, with the query ids as keys and the
            corresponding lists of search results as the values.
            Or returns a tuple with ndarray of query vectors and a dictionary of PRF Dense Search Results with vectors
        """
        q_embs = self.query_encoder.batch_encode(queries)
        faiss.omp_set_num_threads(threads)
        if return_vector:
            d, i, v = self.index.search_and_reconstruct(q_embs, k)
            return q_embs, {key: [PRFDenseSearchResult(self.docids[idx], score, vector)
                                  for score, idx, vector in zip(distances, indexes, vectors) if idx != -1]
                            for key, distances, indexes, vectors in zip(q_ids, d, i, v)}
        else:
            d, i = self.index.search(q_embs, k)
            return {key: [DenseSearchResult(self.docids[idx], score)
                          for score, idx in zip(distances, indexes) if idx != -1]
                    for key, distances, indexes in zip(q_ids, d, i)}

    def switch_to_gpu(self, id: int):
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, id, self.index)

    def switch_to_IVF(self):
        self.index = faiss.IndexIVFFlat(self.index, self.dimension, 256)


def init_query_encoder(encoder: str, device: str):
    encoder_class_map = {
        "dkrr": DkrrDprQueryEncoder,
        "dpr": DprQueryEncoder,
        "bpr": BprQueryEncoder,
        "tct_colbert": TctColBertQueryBatchEncoder,
        "ance": AnceQueryBatchEncoder,
        "sentence": AutoQueryEncoder,
        "auto": AutoQueryEncoder,
    }
    if encoder:
        encoder_class = None

        # determine encoder_class
        if encoder_class is not None:
            encoder_class = encoder_class_map[encoder_class]
        else:
            # if any class keyword was matched in the given encoder name,
            # use that encoder class
            for class_keyword in encoder_class_map:
                if class_keyword in encoder.lower():
                    encoder_class = encoder_class_map[class_keyword]
                    break

            # if none of the class keyword was matched,
            # use the AutoQueryEncoder
            if encoder_class is None:
                encoder_class = AutoQueryEncoder

        # prepare arguments to encoder class
        kwargs = dict(encoder_dir=encoder, device=device)
        if (encoder_class == "sentence") or ("sentence" in encoder):
            kwargs.update(dict(pooling='mean', l2_norm=True))

        return encoder_class(**kwargs)
